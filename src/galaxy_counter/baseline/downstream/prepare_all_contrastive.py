"""
Stage 1: Prepare downstream H5s for the contrastive dual-encoder baseline.

Creates:
  - downstream_mmu_{suffix}.h5
  - downstream_legacy_provabgs_{suffix}.h5
  - downstream_neighbors_{suffix}.h5
  - downstream_hsc_provabgs_{suffix}.h5

Each file stores:
  - 6 embedding sets for real model
  - 6 embedding sets for untrained model
  - 6 random embedding sets
  - labels/*
"""

import argparse
import inspect
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[4]

from galaxy_counter.baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)
from galaxy_counter.data import HSCLegacyDatasetZoom, zoom_legacy_image
from galaxy_counter.neighbors import NeighborsSimpleDataset


# MMU
HDF5_PATH_MMU = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
METADATA_PATH_MMU = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"
NUM_EXAMPLES_MMU = 8192
IMAGE_SIZE = 48
METADATA_PARAMS_MMU = [
    "EBV", "FLUX_G", "FLUX_I", "FLUX_R", "FLUX_W1", "FLUX_W2", "FLUX_W3", "FLUX_W4", "FLUX_Z",
    "SHAPE_E1", "SHAPE_E2", "SHAPE_R",
    "a_g", "a_i", "a_r", "a_y", "a_z",
    "g_cmodel_mag", "g_cmodel_magerr", "g_sdssshape_psf_shape11", "g_sdssshape_psf_shape12", "g_sdssshape_psf_shape22",
    "i_cmodel_mag", "i_cmodel_magerr", "i_extendedness_value", "i_sdssshape_psf_shape11", "i_sdssshape_psf_shape12", "i_sdssshape_psf_shape22",
    "r_cmodel_mag", "r_sdssshape_psf_shape11", "r_sdssshape_psf_shape12", "r_sdssshape_psf_shape22",
    "y_cmodel_mag", "y_cmodel_magerr", "y_extendedness_value",
    "z_cmodel_mag", "z_sdssshape_psf_shape11", "z_sdssshape_psf_shape12", "z_sdssshape_psf_shape22",
]

# Legacy ProvaBGS
OVERLAP_TRAIN_CSV = _repo_root / "data/metadata/leagcy_train_overlap_df.csv"
OVERLAP_EVAL_CSV = _repo_root / "data/metadata/leagcy_eval_overlap_df.csv"
PROCESSED_H5_LEGACY = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
FITS_TRAIN_PATH_LEGACY = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits"
FITS_EVAL_PATH_LEGACY = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_eval_v2.fits"

# HSC ProvaBGS
OVERLAP_CSV_HSC = _repo_root / "data/metadata/hsc_train_overlap_df.csv"
FITS_TRAIN_PATH_HSC = "/data/vision/billf/scratch/pablomer/data/provabgs_hsc_train_v2.fits"

# Neighbors
NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
NUM_EXAMPLES_NEIGHBORS = 4096
NEIGHBORS_SEED = 42
SHUFFLE_NEIGHBORS = True

FITS_DROP_COLS = {"rgb", "tok_image", "tok_image_hsc", "tok_spectrum_desi", "PROVABGS_MCMC", "PROVABGS_THETA_BF", "PROVABGS_LOGMSTAR"}

EMBEDDING_NAMES = (
    "hsc_encoder1",
    "hsc_encoder2",
    "legacy_encoder1",
    "legacy_encoder2",
    "hsc_legacy_encoder1",
    "hsc_legacy_encoder2",
)


class H5PairRowsDataset(torch.utils.data.Dataset):
    """HSC+Legacy pairs by row index in preprocessed H5."""

    def __init__(self, actual_h5_rows, h5_path=PROCESSED_H5_LEGACY):
        self.actual_h5_rows = np.asarray(actual_h5_rows)
        self.h5_path = h5_path
        self.norm_hsc = (0.022, 0.05)
        self.norm_legacy_zoom = (0.045, 0.078)

    def __len__(self):
        return len(self.actual_h5_rows)

    def __getitem__(self, idx):
        row = int(self.actual_h5_rows[idx])
        with h5py.File(self.h5_path, "r") as f:
            hsc_image = torch.from_numpy(f["hsc_images"][row][:]).float()
            legacy_image = torch.from_numpy(f["legacy_images"][row][:]).float()
        mean_hsc, std_hsc = self.norm_hsc
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        legacy_image = zoom_legacy_image(legacy_image)
        mean_legacy, std_legacy = self.norm_legacy_zoom
        legacy_image = (legacy_image - mean_legacy) / std_legacy
        return hsc_image, legacy_image


def _to_native_byteorder(arr):
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("=")))
    return arr


def load_fits_scalar_columns(path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        keep = [c for c in d.columns.names if (c not in FITS_DROP_COLS) and (d[c].ndim == 1)]
        df = pd.DataFrame({c: _to_native_byteorder(d[c].copy()) for c in keep})
    for c in df.columns:
        if df[c].dtype == np.int8 and c.startswith("IS_"):
            df[c] = df[c].astype(bool)
    return df


def _serialize_labels_df(labels_df):
    out = {}
    for col in labels_df.columns:
        arr = labels_df[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array([str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr], dtype="S")
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        out[col] = arr
    return out


def _load_model(checkpoint_path):
    model = DualEncoderContrastiveModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    valid_keys = set(inspect.signature(DualEncoderContrastiveModule.__init__).parameters.keys())
    init_kwargs = {k: v for k, v in hparams.items() if k in valid_keys}
    model = DualEncoderContrastiveModule(**init_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _encode_hsc_legacy(model, hsc_batch, legacy_batch):
    h_g = model.encoder_galaxy(hsc_batch)
    h_i = model.encoder_instrument(hsc_batch)
    l_g = model.encoder_galaxy(legacy_batch)
    l_i = model.encoder_instrument(legacy_batch)
    return h_g, h_i, l_g, l_i


def generate_embeddings_hsc_legacy(model, dataset, device, batch_size=256):
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    with torch.no_grad():
        for hsc_im, leg_im in loader:
            h, l = hsc_im.to(device), leg_im.to(device)
            e1, e2, e3, e4 = _encode_hsc_legacy(model, h, l)
            hsc_1.append(e1.cpu())
            hsc_2.append(e2.cpu())
            leg_1.append(e3.cpu())
            leg_2.append(e4.cpu())
    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()


def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def _labels_from_metadata_list(metadata_list):
    if not metadata_list:
        return {}, []
    keys = list(metadata_list[0].keys())
    labels_n, valid_columns = {}, []
    for col in keys:
        try:
            vals = [m[col] for m in metadata_list]
            arr = np.array(vals)
            if arr.dtype.kind in "iuflb" and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
                valid_columns.append(col)
        except (TypeError, ValueError):
            continue
    return labels_n, valid_columns


def generate_embeddings_neighbors(model, dataset, device, batch_size=256, shuffle=False, seed=None):
    loader_kw = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_neighbors)
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        loader_kw["shuffle"] = True
        loader_kw["generator"] = g
    else:
        loader_kw["shuffle"] = False
    loader = TorchDataLoader(dataset, **loader_kw)

    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    metadata_collected = []
    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            metadata_collected.extend(meta_list)
            h, l = hsc_im.to(device), leg_im.to(device)
            e1, e2, e3, e4 = _encode_hsc_legacy(model, h, l)
            hsc_1.append(e1.cpu())
            hsc_2.append(e2.cpu())
            leg_1.append(e3.cpu())
            leg_2.append(e4.cpu())

    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return (e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()), metadata_collected


def make_random_embeddings_like(real_embeddings_dict, seed=42):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in real_embeddings_dict.items():
        out[k] = rng.standard_normal(v.shape, dtype=np.float32)
    return out


def save_downstream_h5_all_variants(path, real_dict, untrained_dict, random_dict, labels_n, label_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            f.create_dataset(name, data=real_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_untrained", data=untrained_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_random", data=random_dict[name], compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"] = real_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path}")


def _load_metadata_mmu(metadata_path, idx_list, metadata_params):
    metadata_dict = {}
    idx_to_meta = None
    with h5py.File(metadata_path, "r") as f:
        if "indices" in f:
            meta_indices = f["indices"][:]
            idx_to_meta = {int(dataset_idx): meta_idx for meta_idx, dataset_idx in enumerate(meta_indices)}
        valid = []
        for param in metadata_params:
            if param not in f:
                continue
            arr = f[param][:].astype(np.float32)
            if idx_to_meta is not None:
                arr = arr[[idx_to_meta[i] for i in idx_list]]
            else:
                arr = arr[idx_list]
            metadata_dict[param] = arr
            valid.append(param)
    return metadata_dict, valid


def _fit_labels_by_abs(overlap_df, actual_h5_rows, h5_indices, fits_df, overlap_id_col, target_id_col):
    overlap = overlap_df.copy()
    overlap[target_id_col] = overlap[overlap_id_col]
    abs_indices = h5_indices[actual_h5_rows]
    overlap_by_abs = overlap.set_index("abs_index")[target_id_col]
    target_ids = overlap_by_abs.reindex(abs_indices).values
    if pd.isna(target_ids).any():
        raise ValueError("Some abs_index entries have no matching ID in overlap CSV.")

    order_df = pd.DataFrame({target_id_col: target_ids})
    fits_cast = fits_df.copy()
    if np.issubdtype(order_df[target_id_col].dtype, np.number) and target_id_col in fits_cast.columns:
        fits_cast[target_id_col] = pd.to_numeric(fits_cast[target_id_col], errors="coerce")
    labels_df = order_df.merge(fits_cast, on=target_id_col, how="left")
    valid = ~labels_df.isna().any(axis=1)
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_mask = valid.to_numpy()
    return labels_df, kept_mask


def _run_model_variants(get_real_emb_fn, checkpoint_path, seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint_path).to(device)
    real = get_real_emb_fn(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_untrained = _load_untrained_model(checkpoint_path).to(device)
    untrained = get_real_emb_fn(model_untrained)
    del model_untrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_emb = make_random_embeddings_like(real, seed=seed)
    return real, untrained, random_emb


def prepare_mmu(checkpoint_path, output_dir, batch_size, seed, suffix):
    idx_list = list(range(NUM_EXAMPLES_MMU))
    dataset = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH_MMU, idx_list=idx_list)
    metadata_dict, label_columns = _load_metadata_mmu(METADATA_PATH_MMU, idx_list, METADATA_PARAMS_MMU)
    labels_n = {col: metadata_dict[col] for col in label_columns}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        arrs = generate_embeddings_hsc_legacy(model, dataset, device, batch_size)
        return dict(zip(EMBEDDING_NAMES, arrs))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_mmu_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_legacy_provabgs(checkpoint_path, output_dir, batch_size, seed, suffix):
    train_overlap_df = pd.read_csv(OVERLAP_TRAIN_CSV)
    eval_overlap_df = pd.read_csv(OVERLAP_EVAL_CSV)
    with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
        h5_indices = f["indices"][:]
        train_rows = np.where(np.isin(h5_indices, train_overlap_df["abs_index"]))[0]
        eval_rows = np.where(np.isin(h5_indices, eval_overlap_df["abs_index"]))[0]

    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH_LEGACY)
    fits_eval = load_fits_scalar_columns(FITS_EVAL_PATH_LEGACY)
    labels_train, kept_train = _fit_labels_by_abs(
        train_overlap_df, train_rows, h5_indices, fits_train, overlap_id_col="TARGETID", target_id_col="legacy_object_id"
    )
    labels_eval, kept_eval = _fit_labels_by_abs(
        eval_overlap_df, eval_rows, h5_indices, fits_eval, overlap_id_col="TARGETID", target_id_col="legacy_object_id"
    )
    train_rows = train_rows[kept_train]
    eval_rows = eval_rows[kept_eval]
    train_ds = H5PairRowsDataset(train_rows)
    eval_ds = H5PairRowsDataset(eval_rows)
    labels_all = pd.concat([labels_train, labels_eval], axis=0, ignore_index=True)
    labels_n = _serialize_labels_df(labels_all)
    label_columns = list(labels_all.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        e_tr = generate_embeddings_hsc_legacy(model, train_ds, device, batch_size)
        e_ev = generate_embeddings_hsc_legacy(model, eval_ds, device, batch_size)
        return {name: np.concatenate([e_tr[i], e_ev[i]], axis=0) for i, name in enumerate(EMBEDDING_NAMES)}

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_legacy_provabgs_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_hsc_provabgs(checkpoint_path, output_dir, batch_size, seed, suffix):
    overlap_df = pd.read_csv(OVERLAP_CSV_HSC)
    with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
        h5_indices = f["indices"][:]
        rows = np.where(np.isin(h5_indices, overlap_df["abs_index"]))[0]

    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH_HSC)
    labels_df, kept = _fit_labels_by_abs(
        overlap_df, rows, h5_indices, fits_train, overlap_id_col="TARGETID", target_id_col="hsc_object_id"
    )
    rows = rows[kept]
    dataset = H5PairRowsDataset(rows)
    labels_n = _serialize_labels_df(labels_df)
    label_columns = list(labels_df.columns)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _emb_fn(model):
        arrs = generate_embeddings_hsc_legacy(model, dataset, device, batch_size)
        return dict(zip(EMBEDDING_NAMES, arrs))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_hsc_provabgs_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


def prepare_neighbors(checkpoint_path, output_dir, batch_size, seed, suffix):
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_use = min(NUM_EXAMPLES_NEIGHBORS, len(full_dataset))
    dataset = Subset(full_dataset, range(n_use))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    meta_cache = {"labels_n": None, "label_columns": None}

    def _emb_fn(model):
        e_arrays, metadata_collected = generate_embeddings_neighbors(
            model, dataset, device, batch_size, shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED
        )
        if meta_cache["labels_n"] is None:
            labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
            meta_cache["labels_n"] = labels_n
            meta_cache["label_columns"] = label_columns
        return dict(zip(EMBEDDING_NAMES, e_arrays))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_neighbors_{suffix}.h5"
    save_downstream_h5_all_variants(
        out_path, real, untrained, random_emb, meta_cache["labels_n"], meta_cache["label_columns"]
    )
    return out_path


def main():
    p = argparse.ArgumentParser(description="Prepare all contrastive downstream embeddings (real + untrained + random).")
    p.add_argument("--checkpoint", required=True, help="Path to contrastive baseline checkpoint (.ckpt)")
    p.add_argument("--suffix", default=None, help="Output suffix. Default: checkpoint stem")
    p.add_argument("--output-dir", type=Path, default=_here, help="Output directory for H5 files")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", default="mmu,legacy_provabgs,neighbors,hsc_provabgs",
                   help="Comma-separated subset of: mmu,legacy_provabgs,neighbors,hsc_provabgs")
    args = p.parse_args()

    suffix = args.suffix or Path(args.checkpoint).stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]

    saved = []
    if "mmu" in datasets:
        print("[MMU]")
        saved.append(prepare_mmu(args.checkpoint, output_dir, args.batch_size, args.seed, suffix))
    if "legacy_provabgs" in datasets:
        print("[Legacy ProvaBGS]")
        saved.append(prepare_legacy_provabgs(args.checkpoint, output_dir, args.batch_size, args.seed, suffix))
    if "neighbors" in datasets:
        print("[Neighbors]")
        saved.append(prepare_neighbors(args.checkpoint, output_dir, args.batch_size, args.seed, suffix))
    if "hsc_provabgs" in datasets:
        print("[HSC ProvaBGS]")
        saved.append(prepare_hsc_provabgs(args.checkpoint, output_dir, args.batch_size, args.seed, suffix))

    print("\nDone. Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
