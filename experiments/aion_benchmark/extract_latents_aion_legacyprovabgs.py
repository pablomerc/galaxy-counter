"""
Extract AION embeddings for Legacy ProvaBGS overlap: same ID selection as prepare_legacy_provabgs
(overlap CSVs + preprocessed H5 indices), but load images from neighbours_v2.h5 at those same indices,
and labels from Legacy ProvaBGS FITS. Saves downstream H5 compatible with predict_aion.

Run from galaxy_model/ or aion_benchmark/:
  python aion_benchmark/extract_latents_aion_legacyprovabgs.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

# ---------------------------------------------------------------------------
# Same ID extraction as prepare_legacy_provabgs: overlap CSVs + indices from preprocessed H5
# ---------------------------------------------------------------------------
OVERLAP_TRAIN_CSV = _src / "util_notebooks" / "leagcy_train_overlap_df.csv"
OVERLAP_EVAL_CSV = _src / "util_notebooks" / "leagcy_eval_overlap_df.csv"
PROCESSED_H5_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
OUTPUT_H5 = _here / "downstream_aion_legacy_provabgs.h5"

# FITS paths for Legacy ProvaBGS labels (scalar columns only)
FITS_TRAIN_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits"
FITS_EVAL_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_eval_v2.fits"

BATCH_SIZE = 32
NUM_ENCODER_TOKENS = 1200
NUM_ENCODER_TOKENS_SINGLE = 600
DEVICE = "cuda"

FITS_DROP_COLS = {
    "rgb",
    "tok_image",
    "tok_image_hsc",
    "tok_spectrum_desi",
    "PROVABGS_MCMC",
    "PROVABGS_THETA_BF",
    "PROVABGS_LOGMSTAR",
}


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


def get_labels_in_dataloader_order(overlap_df, actual_h5_rows, h5_indices, fits_df):
    """Labels in dataloader order; returns labels_df and kept_mask (same as prepare_legacy_provabgs)."""
    id_col = "legacy_object_id"
    abs_indices = h5_indices[actual_h5_rows]
    overlap_by_abs = overlap_df.set_index("abs_index")["TARGETID"]
    target_ids = overlap_by_abs.reindex(abs_indices).values
    if pd.isna(target_ids).any():
        bad = np.where(pd.isna(target_ids))[0]
        raise ValueError(f"Some abs_indices have no TARGETID in overlap: {bad[:5]}...")
    target_ids = [str(x) for x in target_ids]
    order_df = pd.DataFrame({id_col: target_ids})
    fits_str = fits_df.copy()
    fits_str[id_col] = fits_str[id_col].astype(str)
    labels_df = order_df.merge(fits_str, on=id_col, how="left")
    valid = ~labels_df.isna().any(axis=1)
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  Proceeding with {valid.sum()}/{len(labels_df)} rows (dropping {n_missing} without FITS match).")
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_mask = valid.to_numpy()
    return labels_df, kept_mask


def _resolve_neighbor_rows(h5_indices, actual_h5_rows_filtered, neighbours_indices=None):
    """
    Return neighbour H5 row indices in the same order as actual_h5_rows_filtered.
    If neighbours_indices is provided (neighbours H5 has 'indices' dataset), map by abs_index;
    otherwise assume same row order and return actual_h5_rows_filtered.
    """
    abs_indices = h5_indices[actual_h5_rows_filtered]
    if neighbours_indices is not None:
        abs_to_row = {int(a): i for i, a in enumerate(neighbours_indices)}
        out = []
        for a in abs_indices:
            if a not in abs_to_row:
                raise ValueError(f"abs_index {a} not found in neighbours H5 'indices'")
            out.append(abs_to_row[a])
        return np.array(out)
    return np.asarray(actual_h5_rows_filtered)


def main():
    print("Device:", DEVICE)
    train_overlap_df = pd.read_csv(OVERLAP_TRAIN_CSV)
    eval_overlap_df = pd.read_csv(OVERLAP_EVAL_CSV)
    print("Legacy overlap (train):", len(train_overlap_df), "(eval):", len(eval_overlap_df))

    # Same index resolution as prepare_legacy_provabgs: use preprocessed H5 to get row indices
    with h5py.File(PROCESSED_H5_PATH, "r") as f:
        h5_indices = f["indices"][:]
        train_mask = np.isin(h5_indices, train_overlap_df["abs_index"])
        train_actual_h5_rows = np.where(train_mask)[0]
        eval_mask = np.isin(h5_indices, eval_overlap_df["abs_index"])
        eval_actual_h5_rows = np.where(eval_mask)[0]
    print(f"Train matches (preprocessed H5 rows): {train_actual_h5_rows.shape[0]}")
    print(f"Eval matches (preprocessed H5 rows): {eval_actual_h5_rows.shape[0]}")

    # FITS labels and filter to rows with FITS match
    print("Loading FITS labels (scalar columns)...")
    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH)
    fits_eval = load_fits_scalar_columns(FITS_EVAL_PATH)
    print("Aligning labels to dataloader order...")
    labels_train, kept_train = get_labels_in_dataloader_order(
        train_overlap_df, train_actual_h5_rows, h5_indices, fits_train
    )
    labels_eval, kept_eval = get_labels_in_dataloader_order(
        eval_overlap_df, eval_actual_h5_rows, h5_indices, fits_eval
    )
    train_actual_h5_rows_filtered = train_actual_h5_rows[kept_train]
    eval_actual_h5_rows_filtered = eval_actual_h5_rows[kept_eval]

    # Resolve neighbour H5 row indices (same objects, possibly different row layout)
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        neighbours_indices = f["indices"][:] if "indices" in f else None
    train_neighbour_rows = _resolve_neighbor_rows(
        h5_indices, train_actual_h5_rows_filtered, neighbours_indices
    )
    eval_neighbour_rows = _resolve_neighbor_rows(
        h5_indices, eval_actual_h5_rows_filtered, neighbours_indices
    )
    n_train, n_eval = len(train_neighbour_rows), len(eval_neighbour_rows)
    print(f"Neighbour rows: train {n_train}, eval {n_eval}")

    # Serialize labels for HDF5 (train then eval, same order as we will encode)
    labels_all = pd.concat([labels_train, labels_eval], axis=0, ignore_index=True)
    label_columns = list(labels_all.columns)
    labels_n = {}
    for col in labels_all.columns:
        arr = labels_all[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array([str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr], dtype="S")
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        labels_n[col] = arr

    # AION model and codec
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DEVICE)
    codec_manager = CodecManager(device=DEVICE)

    all_neighbour_rows = np.concatenate([train_neighbour_rows, eval_neighbour_rows])
    all_embeddings = []
    all_embeddings_mean = []
    all_embeddings_mean_legacy = []
    all_embeddings_mean_hsc = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, len(all_neighbour_rows), BATCH_SIZE), desc="Encoding"):
            end = min(start + BATCH_SIZE, len(all_neighbour_rows))
            indices = all_neighbour_rows[start:end]

            legacy_tensor = torch.from_numpy(f["images_legacy"][indices]).to(DEVICE)
            hsc_tensor = torch.from_numpy(f["images_hsc"][indices]).to(DEVICE)

            image_leg = LegacySurveyImage(
                flux=legacy_tensor,
                bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
            )
            image_hsc = HSCImage(
                flux=hsc_tensor,
                bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            )
            tokens_hsc_leg = codec_manager.encode(image_hsc, image_leg)
            tokens_leg = codec_manager.encode(image_leg)
            tokens_hsc = codec_manager.encode(image_hsc)

            with torch.no_grad():
                emb_hsc_leg = model.encode(tokens_hsc_leg, num_encoder_tokens=NUM_ENCODER_TOKENS)
                emb_leg = model.encode(tokens_leg, num_encoder_tokens=NUM_ENCODER_TOKENS_SINGLE)
                emb_hsc = model.encode(tokens_hsc, num_encoder_tokens=NUM_ENCODER_TOKENS_SINGLE)
            all_embeddings.append(emb_hsc_leg.cpu().numpy())
            all_embeddings_mean.append(emb_hsc_leg.mean(dim=1).cpu().numpy())
            all_embeddings_mean_legacy.append(emb_leg.mean(dim=1).cpu().numpy())
            all_embeddings_mean_hsc.append(emb_hsc.mean(dim=1).cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    embeddings_mean = np.concatenate(all_embeddings_mean, axis=0)
    embeddings_mean_legacy = np.concatenate(all_embeddings_mean_legacy, axis=0)
    embeddings_mean_hsc = np.concatenate(all_embeddings_mean_hsc, axis=0)

    n_total = len(labels_all)
    assert embeddings.shape[0] == n_total, f"Embeddings {embeddings.shape[0]} vs labels {n_total}"
    print("Embeddings shape:", embeddings.shape)

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="downstream_aion_legacy_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean", data=embeddings_mean, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_legacy", data=embeddings_mean_legacy, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_hsc", data=embeddings_mean_hsc, compression="gzip", compression_opts=4)
            for col in label_columns:
                f.create_dataset(
                    f"labels/{col}",
                    data=labels_n[col],
                    compression="gzip",
                    compression_opts=4,
                )
            f.attrs["num_examples"] = n_total
            f.attrs["label_columns"] = list(label_columns)
            f.attrs["embedding_shape"] = embeddings.shape
        shutil.move(tmp_path, OUTPUT_H5)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise

    print(f"Saved: {OUTPUT_H5} (embeddings, embeddings_mean, embeddings_mean_legacy, embeddings_mean_hsc, {len(label_columns)} labels)")


if __name__ == "__main__":
    main()
