"""
Prepare AION embeddings + ProvaBGS labels from aion_legacy_match.py output.

Takes extracted_matches.h5 (abs_index, target_id, images_hsc, images_legacy),
matches target_id to legacy_object_id in provabgs_legacysurvey_train_v2.fits,
and produces a single H5 with AION embeddings and labels (same layout as
extract_latents_aion_legacyprovabgs / predict_aion).

ID handling: we do NOT use preprocessed H5 or overlap CSV row order. We use
the match file's target_id directly and join with FITS on legacy_object_id,
so embedding row i corresponds exactly to labels row i.

Run from galaxy_model/ or aion_benchmark/:
  python aion_benchmark/aion_legacy/prepare_aion_legacy.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_aion_benchmark = _here.parent
_src = _aion_benchmark.parent
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
# Paths
# ---------------------------------------------------------------------------
MATCH_H5_PATH = _here / "extracted_matches.h5"
FITS_TRAIN_PATH = "/data/vision/billf/scratch/pablomer/data/provabgs_legacysurvey_train_v2.fits"
OUTPUT_H5 = _here / "downstream_aion_legacy_train.h5"

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


def _ensure_str(x):
    """Ensure ID is Python str for matching (h5py may store as bytes)."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").strip()
    return str(x).strip()


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


def main():
    print("Device:", DEVICE)
    if not MATCH_H5_PATH.exists():
        raise FileNotFoundError(
            f"Match output not found: {MATCH_H5_PATH}. Run aion_legacy_match.py first (from aion_legacy/ or set cwd so extracted_matches.h5 is created, then copy here or set MATCH_H5_PATH)."
        )

    # 1. Load match output: target_id and images (order = row order in H5)
    with h5py.File(MATCH_H5_PATH, "r") as f:
        target_ids_raw = f["target_id"][:]
        images_hsc = f["images_hsc"][:]
        images_legacy = f["images_legacy"][:]
    # Normalize IDs to str for matching
    target_ids = [_ensure_str(x) for x in target_ids_raw]
    n_match = len(target_ids)
    print(f"Match H5: {n_match} objects (target_id, images_hsc, images_legacy).")

    # 2. Load FITS labels (scalar columns only)
    print("Loading FITS labels (scalar columns)...")
    fits_df = load_fits_scalar_columns(FITS_TRAIN_PATH)
    id_col = "legacy_object_id"
    fits_df[id_col] = fits_df[id_col].astype(str).str.strip()

    # 3. Merge: keep only rows where target_id has a FITS match; preserve order of match H5
    order_df = pd.DataFrame({id_col: target_ids})
    labels_df = order_df.merge(fits_df, on=id_col, how="left")
    valid = ~labels_df.isna().any(axis=1)
    n_missing = (~valid).sum()
    if n_missing > 0:
        print(f"  Proceeding with {valid.sum()}/{len(labels_df)} rows (dropping {n_missing} without FITS match).")
    labels_df = labels_df[valid].reset_index(drop=True)
    kept_indices = np.where(valid)[0]

    # 4. Restrict images to rows that have labels (same order as labels_df)
    images_hsc = images_hsc[kept_indices]
    images_legacy = images_legacy[kept_indices]
    n_total = len(labels_df)
    print(f"With FITS match: {n_total} objects.")

    # 5. Serialize labels for HDF5
    label_columns = list(labels_df.columns)
    labels_n = {}
    for col in labels_df.columns:
        arr = labels_df[col].values
        if arr.dtype.kind in "OU" or arr.dtype == object:
            arr = np.array([str(x).encode("utf-8") if pd.notna(x) else b"" for x in arr], dtype="S")
        elif arr.dtype == bool:
            arr = arr.astype(np.int8)
        labels_n[col] = arr

    # 6. AION model and codec
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DEVICE)
    codec_manager = CodecManager(device=DEVICE)

    all_embeddings = []
    all_embeddings_mean = []
    all_embeddings_mean_legacy = []
    all_embeddings_mean_hsc = []

    for start in tqdm(range(0, n_total, BATCH_SIZE), desc="Encoding"):
        end = min(start + BATCH_SIZE, n_total)
        legacy_tensor = torch.from_numpy(images_legacy[start:end]).to(DEVICE)
        hsc_tensor = torch.from_numpy(images_hsc[start:end]).to(DEVICE)

        image_leg = LegacySurveyImage(
            flux=legacy_tensor,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        image_hsc_batch = HSCImage(
            flux=hsc_tensor,
            bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
        )
        tokens_hsc_leg = codec_manager.encode(image_hsc_batch, image_leg)
        tokens_leg = codec_manager.encode(image_leg)
        tokens_hsc = codec_manager.encode(image_hsc_batch)

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

    assert embeddings.shape[0] == n_total, f"Embeddings {embeddings.shape[0]} vs labels {n_total}"
    print("Embeddings shape:", embeddings.shape)

    # 7. Save single H5 (embeddings + labels)
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
