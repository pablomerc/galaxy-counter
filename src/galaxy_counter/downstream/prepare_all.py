"""
Single script to prepare all embeddings for downstream evaluation given one model checkpoint.

For each dataset (MMU, Legacy ProvaBGS, Neighbors), produces one H5 file containing:
  - Real embeddings (from the given checkpoint): hsc_encoder1, hsc_encoder2, ... (6 total)
  - Untrained embeddings (same architecture, random init): *_untrained (6 total)
  - Random embeddings (same shape, N(0,1)): *_random (6 total)
  - Labels (dataset-specific)

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/final/prepare_all.py --checkpoint PATH --module double_train_fm_neighbors.py --suffix zdim16_nogeom_neighbors
"""
import importlib
import os
from pathlib import Path

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[3]

import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

# ---------------------------------------------------------------------------
# Config (paths and targets from prepare_all.py)
# ---------------------------------------------------------------------------

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
FITS_TRAIN_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits"
FITS_EVAL_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_eval_v2.fits"
FITS_DROP_COLS = {"rgb", "tok_image", "tok_image_hsc", "tok_spectrum_desi", "PROVABGS_MCMC", "PROVABGS_THETA_BF", "PROVABGS_LOGMSTAR"}
NORM_DICT = {"hsc": [0.022, 0.05], "legacy": [0.023, 0.063], "legacy_zoom": [0.045, 0.078]}

# Neighbors
NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
NUM_EXAMPLES_NEIGHBORS = 4096
NEIGHBORS_SEED = 42
SHUFFLE_NEIGHBORS = True

EMBEDDING_NAMES = (
    "hsc_encoder1", "hsc_encoder2", "legacy_encoder1", "legacy_encoder2",
    "hsc_legacy_encoder1", "hsc_legacy_encoder2",
)

# ---------------------------------------------------------------------------
# Downstream prediction targets (for predict_all.py)
# These are the label columns to predict from each H5. Must exist in saved labels.
# ---------------------------------------------------------------------------

# Neighbors H5 (downstream_neighbors_{suffix}.h5)
instrument_neighbors_legacy = [
    "legacy_PSFSIZE_G", "legacy_PSFSIZE_I", "legacy_PSFSIZE_R", "legacy_PSFSIZE_Z",
    "legacy_PSFDEPTH_G", "legacy_PSFDEPTH_R", "legacy_PSFDEPTH_I", "legacy_PSFDEPTH_Z",
    "legacy_GALDEPTH_G", "legacy_GALDEPTH_R", "legacy_GALDEPTH_I", "legacy_GALDEPTH_Z",
    "legacy_NOBS_G", "legacy_NOBS_R", "legacy_NOBS_I", "legacy_NOBS_Z",
]
instrument_neighbors_hsc = [
    "hsc_i_variance_value",
    "hsc_r_variance_value",
    "hsc_z_variance_value",
]
# Derived in predict_all from hsc_*_sdssshape_psf_shape11/22 (compute_hsc_psf_seeing); must exist in neighbors H5 labels
instrument_neighbors_hsc_psf_fwhm = ["hsc_g_psf_fwhm", "hsc_i_psf_fwhm", "hsc_r_psf_fwhm", "hsc_z_psf_fwhm"]

# Legacy ProvaBGS H5 (downstream_legacy_provabgs_{suffix}.h5)
physics_provabgs = ["desi_Z", "LOG_MSTAR", "TAGE_MW", "LOG_Z_MW", "sSFR", "hsc_g_extendedness_value", "DEC"]
instrument_provabgs = []

# MMU H5 (downstream_mmu_{suffix}.h5)
physics_mmu = ["SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
instrument_mmu_legacy = ["EBV"]
instrument_mmu_hsc = ["a_g", "a_i", "a_r", "a_y", "a_z"]


# ---------------------------------------------------------------------------
# Model loading and embedding generation (shared)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path, module_filename):
    """Load double-encoder model from checkpoint."""
    module_stem = Path(module_filename).stem
    mod = importlib.import_module(f"galaxy_counter.models.{module_stem}")
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model(module_filename="double_train_fm_neighbors.py"):
    """Untrained model with same architecture (for neighbors-style; override if old dataloader)."""
    module_stem = Path(module_filename).stem
    mod = importlib.import_module(f"galaxy_counter.models.{module_stem}")
    model = mod.ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=1e-4,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.3,
        mask_center=False,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def generate_embeddings_hsc_legacy(model, dataset, device, batch_size=256):
    """Run encoder_1 and encoder_2 on HSC and Legacy; return 6 numpy arrays. No metadata."""
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            hsc_im, leg_im = batch[0], batch[1]
            h, l = hsc_im.to(device), leg_im.to(device)
            hsc_1.append(model.encoder_1(h).cpu())
            hsc_2.append(model.encoder_2(h).cpu())
            leg_1.append(model.encoder_1(l).cpu())
            leg_2.append(model.encoder_2(l).cpu())
    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()


def make_random_embeddings_like(real_embeddings_dict, seed=42):
    """Return dict of arrays same shape as real_embeddings_dict, filled with N(0,1)."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in real_embeddings_dict.items():
        out[k] = rng.standard_normal(v.shape, dtype=np.float32)
    return out


def save_downstream_h5_all_variants(path, real_dict, untrained_dict, random_dict, labels_n, label_columns):
    """Save 18 embedding datasets (6 real + 6 untrained + 6 random) and labels to HDF5."""
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
        n = real_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["num_examples"] = n
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path} (real + untrained + random embeddings + labels)")


# ---------------------------------------------------------------------------
# MMU
# ---------------------------------------------------------------------------

def _load_metadata_mmu(metadata_path, idx_list, metadata_params):
    metadata_dict = {}
    idx_to_meta = None
    with h5py.File(metadata_path, "r") as f:
        if "indices" in f:
            meta_indices = f["indices"][:]
            idx_to_meta = {int(dataset_idx): meta_idx for meta_idx, dataset_idx in enumerate(meta_indices)}
        valid = []
        for param in metadata_params:
            if param in f:
                arr = f[param][:].astype(np.float32)
                if idx_to_meta is not None:
                    arr = arr[[idx_to_meta[i] for i in idx_list]]
                else:
                    arr = arr[idx_list]
                metadata_dict[param] = arr
                valid.append(param)
    return metadata_dict, valid


def prepare_mmu(checkpoint_path, module_filename, output_dir, device, batch_size, seed, suffix):
    from data import HSCLegacyDatasetZoom
    idx_list = list(range(NUM_EXAMPLES_MMU))
    dataset = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH_MMU, idx_list=idx_list)
    metadata_dict, label_columns = _load_metadata_mmu(METADATA_PATH_MMU, idx_list, METADATA_PARAMS_MMU)
    labels_n = {col: metadata_dict[col] for col in label_columns}

    model = _load_model(checkpoint_path, module_filename).to(device)
    print("  MMU: generating real embeddings...")
    real = dict(zip(EMBEDDING_NAMES, generate_embeddings_hsc_legacy(model, dataset, device, batch_size)))
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    model_untrained = _load_untrained_model(module_filename).to(device)
    print("  MMU: generating untrained embeddings...")
    untrained = dict(zip(EMBEDDING_NAMES, generate_embeddings_hsc_legacy(model_untrained, dataset, device, batch_size)))
    del model_untrained
    torch.cuda.empty_cache() if device.type == "cuda" else None

    random_emb = make_random_embeddings_like(real, seed=seed)

    out_path = output_dir / f"downstream_mmu_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


# ---------------------------------------------------------------------------
# Legacy ProvaBGS
# ---------------------------------------------------------------------------

class LegacyProvabgsDataset(torch.utils.data.Dataset):
    """HSC+Legacy pairs by H5 row indices (same logic as prepare_legacy_provabgs)."""

    def __init__(self, actual_h5_rows, norm_dict=None):
        self.actual_h5_rows = np.asarray(actual_h5_rows)
        self.norm_dict = norm_dict or NORM_DICT

    def __len__(self):
        return len(self.actual_h5_rows)

    def __getitem__(self, idx):
        from data import zoom_legacy_image
        index = int(self.actual_h5_rows[idx])
        with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
            hsc_image = torch.from_numpy(f["hsc_images"][index][:]).float()
            legacy_image = torch.from_numpy(f["legacy_images"][index][:]).float()
        mean_hsc, std_hsc = self.norm_dict["hsc"]
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        legacy_image = zoom_legacy_image(legacy_image)
        mean_legacy_zoom, std_legacy_zoom = self.norm_dict["legacy_zoom"]
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom
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
        names = d.columns.names
        keep = [c for c in names if (c not in FITS_DROP_COLS) and (d[c].ndim == 1)]
        df = pd.DataFrame({c: _to_native_byteorder(d[c].copy()) for c in keep})
    for c in df.columns:
        if df[c].dtype == np.int8 and c.startswith("IS_"):
            df[c] = df[c].astype(bool)
    return df


def get_labels_legacy_provabgs(overlap_df, actual_h5_rows, h5_indices, fits_df):
    id_col = "legacy_object_id"
    abs_indices = h5_indices[actual_h5_rows]
    overlap_by_abs = overlap_df.set_index("abs_index")["TARGETID"]
    target_ids = overlap_by_abs.reindex(abs_indices).values
    if pd.isna(target_ids).any():
        raise ValueError("Some abs_indices have no TARGETID in overlap.")
    target_ids = [str(x) for x in target_ids]
    order_df = pd.DataFrame({id_col: target_ids})
    fits_str = fits_df.copy()
    fits_str[id_col] = fits_str[id_col].astype(str)
    labels_df = order_df.merge(fits_str, on=id_col, how="left")
    valid = ~labels_df.isna().any(axis=1)
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_mask = valid.to_numpy()
    return labels_df, kept_mask


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


def prepare_legacy_provabgs(checkpoint_path, module_filename, output_dir, device, batch_size, seed, suffix):
    train_overlap_df = pd.read_csv(OVERLAP_TRAIN_CSV)
    eval_overlap_df = pd.read_csv(OVERLAP_EVAL_CSV)
    with h5py.File(PROCESSED_H5_LEGACY, "r") as f:
        h5_indices = f["indices"][:]
        train_mask = np.isin(h5_indices, train_overlap_df["abs_index"])
        train_actual_h5_rows = np.where(train_mask)[0]
        eval_mask = np.isin(h5_indices, eval_overlap_df["abs_index"])
        eval_actual_h5_rows = np.where(eval_mask)[0]

    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH)
    fits_eval = load_fits_scalar_columns(FITS_EVAL_PATH)
    labels_train, kept_train = get_labels_legacy_provabgs(train_overlap_df, train_actual_h5_rows, h5_indices, fits_train)
    labels_eval, kept_eval = get_labels_legacy_provabgs(eval_overlap_df, eval_actual_h5_rows, h5_indices, fits_eval)
    train_actual_h5_rows = train_actual_h5_rows[kept_train]
    eval_actual_h5_rows = eval_actual_h5_rows[kept_eval]
    train_ds = LegacyProvabgsDataset(train_actual_h5_rows)
    eval_ds = LegacyProvabgsDataset(eval_actual_h5_rows)
    labels_all = pd.concat([labels_train, labels_eval], axis=0, ignore_index=True)
    labels_n = _serialize_labels_df(labels_all)
    label_columns = list(labels_all.columns)

    model = _load_model(checkpoint_path, module_filename).to(device)
    print("  Legacy ProvaBGS: generating real embeddings (train)...")
    e_tr = generate_embeddings_hsc_legacy(model, train_ds, device, batch_size)
    print("  Legacy ProvaBGS: generating real embeddings (eval)...")
    e_ev = generate_embeddings_hsc_legacy(model, eval_ds, device, batch_size)
    real = {name: np.concatenate([e_tr[i], e_ev[i]], axis=0) for i, name in enumerate(EMBEDDING_NAMES)}
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    model_untrained = _load_untrained_model(module_filename).to(device)
    print("  Legacy ProvaBGS: generating untrained embeddings (train)...")
    e_tr = generate_embeddings_hsc_legacy(model_untrained, train_ds, device, batch_size)
    print("  Legacy ProvaBGS: generating untrained embeddings (eval)...")
    e_ev = generate_embeddings_hsc_legacy(model_untrained, eval_ds, device, batch_size)
    untrained = {name: np.concatenate([e_tr[i], e_ev[i]], axis=0) for i, name in enumerate(EMBEDDING_NAMES)}
    del model_untrained
    torch.cuda.empty_cache() if device.type == "cuda" else None

    random_emb = make_random_embeddings_like(real, seed=seed)
    out_path = output_dir / f"downstream_legacy_provabgs_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


# ---------------------------------------------------------------------------
# Neighbors
# ---------------------------------------------------------------------------

def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def _labels_from_metadata_list(metadata_list):
    if not metadata_list:
        return {}, []
    keys = list(metadata_list[0].keys())
    labels_n = {}
    valid_columns = []
    for col in keys:
        try:
            vals = [m[col] for m in metadata_list]
            arr = np.array(vals)
            if arr.dtype.kind in "iuflb" and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32) if arr.dtype.kind in "f" else arr
                valid_columns.append(col)
        except (TypeError, ValueError):
            pass
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
            hsc_1.append(model.encoder_1(h).cpu())
            hsc_2.append(model.encoder_2(h).cpu())
            leg_1.append(model.encoder_1(l).cpu())
            leg_2.append(model.encoder_2(l).cpu())
    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return (e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()), metadata_collected


def prepare_neighbors(checkpoint_path, module_filename, output_dir, device, batch_size, seed, suffix):
    from galaxy_counter.neighbors import NeighborsSimpleDataset
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(NUM_EXAMPLES_NEIGHBORS, n_total)
    dataset = Subset(full_dataset, range(n_use))

    model = _load_model(checkpoint_path, module_filename).to(device)
    print("  Neighbors: generating real embeddings...")
    (e_arrays, metadata_collected) = generate_embeddings_neighbors(
        model, dataset, device, batch_size, shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED
    )
    labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
    real = dict(zip(EMBEDDING_NAMES, e_arrays))
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    model_untrained = _load_untrained_model(module_filename).to(device)
    print("  Neighbors: generating untrained embeddings...")
    (e_arrays, _) = generate_embeddings_neighbors(
        model_untrained, dataset, device, batch_size, shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED
    )
    untrained = dict(zip(EMBEDDING_NAMES, e_arrays))
    del model_untrained
    torch.cuda.empty_cache() if device.type == "cuda" else None

    random_emb = make_random_embeddings_like(real, seed=seed)
    out_path = output_dir / f"downstream_neighbors_{suffix}.h5"
    save_downstream_h5_all_variants(out_path, real, untrained, random_emb, labels_n, label_columns)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare all downstream embeddings (real + untrained + random) for one checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.ckpt)")
    p.add_argument("--module", required=True, help="Module filename, e.g. double_train_fm_neighbors.py or double_train_fm.py")
    p.add_argument("--suffix", default=None, help="Output suffix for filenames (default: from checkpoint dir name)")
    p.add_argument("--output-dir", type=Path, default=_here, help="Output directory for H5 files")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", default="mmu,legacy_provabgs,neighbors", help="Comma-separated: mmu,legacy_provabgs,neighbors")
    args = p.parse_args()

    if args.suffix is None:
        args.suffix = Path(args.checkpoint).parent.parent.name
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]

    suffix = args.suffix
    saved = []
    if "mmu" in datasets:
        print("[MMU]")
        p_mmu = prepare_mmu(args.checkpoint, args.module, output_dir, device, args.batch_size, args.seed, suffix)
        saved.append(p_mmu)
    if "legacy_provabgs" in datasets:
        print("[Legacy ProvaBGS]")
        p_leg = prepare_legacy_provabgs(args.checkpoint, args.module, output_dir, device, args.batch_size, args.seed, suffix)
        saved.append(p_leg)
    if "neighbors" in datasets:
        print("[Neighbors]")
        p_nei = prepare_neighbors(args.checkpoint, args.module, output_dir, device, args.batch_size, args.seed, suffix)
        saved.append(p_nei)

    print("\nDone. Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
