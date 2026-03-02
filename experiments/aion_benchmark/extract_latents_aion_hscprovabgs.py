"""
Extract AION embeddings for HSC ProvaBGS overlap: same ID selection as prepare_hsc_provabgs
(single train overlap CSV), load images from neighbours_v2.h5 (mapped by abs_index),
and labels from HSC ProvaBGS FITS. No preprocessed H5 needed.

Run from galaxy_model/ or aion_benchmark/:
  python aion_benchmark/extract_latents_aion_hscprovabgs.py
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
# Same ID selection as prepare_hsc_provabgs: single train overlap CSV; images from neighbours H5
# ---------------------------------------------------------------------------
OVERLAP_TRAIN_CSV = _src / "downstream_evaluation" / "hsc_train_overlap_df.csv"
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
OUTPUT_H5 = _here / "downstream_aion_hsc_provabgs.h5"

# FITS path for HSC ProvaBGS labels (train only; no eval split for HSC catalog)
FITS_TRAIN_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_hsc_train_v2.fits"

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


def get_labels_from_overlap_fits(overlap_df, fits_df):
    """
    Merge overlap CSV (TARGETID, abs_index) with FITS on hsc_object_id.
    Returns labels_df (only rows with FITS match) and abs_indices_kept (same order as labels_df).
    """
    id_col = "hsc_object_id"
    overlap_df = overlap_df.copy()
    overlap_df[id_col] = overlap_df["TARGETID"].astype(np.int64)
    fits_df = fits_df.copy()
    fits_df[id_col] = fits_df[id_col].astype(np.int64)

    order_df = overlap_df[[id_col, "abs_index"]].copy()
    labels_df = order_df.merge(fits_df, on=id_col, how="left")

    valid = ~labels_df.isna().any(axis=1)
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(
            f"  Proceeding with {valid.sum()}/{len(labels_df)} rows (dropping {n_missing} without FITS match)."
        )
    abs_indices_kept = order_df.loc[valid, "abs_index"].values
    labels_df = labels_df.loc[valid].drop(columns=["abs_index"], errors="ignore").reset_index(drop=True)
    return labels_df, abs_indices_kept


def _abs_index_to_neighbor_rows(abs_indices):
    """
    Neighbours H5 row order matches abs_index: row i = abs_index i.
    Return abs_indices as the row indices to use (same order).
    """
    return np.asarray(abs_indices, dtype=np.int64)


def main():
    print("Device:", DEVICE)
    train_overlap_df = pd.read_csv(OVERLAP_TRAIN_CSV)
    print("HSC overlap (train):", len(train_overlap_df))

    # Labels: merge overlap CSV with FITS on hsc_object_id (same IDs as prepare_hsc_provabgs)
    print("Loading FITS labels (scalar columns)...")
    fits_train = load_fits_scalar_columns(FITS_TRAIN_PATH)
    print("Aligning labels to overlap order...")
    labels_train, abs_indices_kept = get_labels_from_overlap_fits(train_overlap_df, fits_train)

    # Neighbours H5 row order = abs_index (row i is abs_index i)
    train_neighbour_rows = _abs_index_to_neighbor_rows(abs_indices_kept)
    n_train = len(train_neighbour_rows)
    print(f"Neighbour rows: train {n_train}")

    # Serialize labels for HDF5
    label_columns = list(labels_train.columns)
    labels_n = {}
    for col in labels_train.columns:
        arr = labels_train[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array([str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr], dtype="S")
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        labels_n[col] = arr

    # AION model and codec
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DEVICE)
    codec_manager = CodecManager(device=DEVICE)

    all_embeddings = []
    all_embeddings_mean = []
    all_embeddings_mean_legacy = []
    all_embeddings_mean_hsc = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, len(train_neighbour_rows), BATCH_SIZE), desc="Encoding"):
            end = min(start + BATCH_SIZE, len(train_neighbour_rows))
            indices = train_neighbour_rows[start:end]

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

    n_total = len(labels_train)
    assert embeddings.shape[0] == n_total, f"Embeddings {embeddings.shape[0]} vs labels {n_total}"
    print("Embeddings shape:", embeddings.shape)

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix="downstream_aion_hsc_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            f.create_dataset(
                "embeddings_mean", data=embeddings_mean, compression="gzip", compression_opts=4
            )
            f.create_dataset(
                "embeddings_mean_legacy",
                data=embeddings_mean_legacy,
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "embeddings_mean_hsc",
                data=embeddings_mean_hsc,
                compression="gzip",
                compression_opts=4,
            )
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

    print(
        f"Saved: {OUTPUT_H5} (embeddings, embeddings_mean, embeddings_mean_legacy, embeddings_mean_hsc, {len(label_columns)} labels)"
    )


if __name__ == "__main__":
    main()
