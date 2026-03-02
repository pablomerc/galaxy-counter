"""
Match metadata (preprocessed H5) to neighbours by HSC ID, extract images from neighbours,
compute AION embeddings, and save one H5 with IDs, embeddings, and labels aligned by HSC ID.

This fixes the ID alignment issue in extract_latents_aion_mmu.py, which used row indices
(0..NUM_EXAMPLES_MMU) instead of matching by HSC object ID between metadata and neighbours.

Run from galaxy_model/ or aion_benchmark/:
  python aion_benchmark/aion_mmu/aion_mmu_match_and_extract.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
OUTPUT_H5 = _here / "downstream_aion_mmu_matched.h5"

# Labels to keep (same as extract_latents_aion_mmu)
physics_mmu = ["SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
instrument_mmu_legacy = ["EBV"]
instrument_mmu_hsc = ["a_g", "a_i", "a_r", "a_y", "a_z"]
LABEL_COLUMNS_MMU = physics_mmu + instrument_mmu_legacy + instrument_mmu_hsc

BATCH_SIZE = 32
NUM_ENCODER_TOKENS = 1200
NUM_ENCODER_TOKENS_SINGLE = 600
DEVICE = "cuda"


def find_matches(metadata_path, neighbors_path):
    """
    Find rows in neighbours that (1) have object_id_hsc in metadata's hsc_object_id
    and (2) have source_type == 0. Returns neighbour row indices and matching HSC IDs
    in the same order (so matching_ids[i] is the HSC ID for neighbour row indices_in_file[i]).
    """
    with h5py.File(metadata_path, "r") as f:
        hsc_ids = f["hsc_object_id"][:]
    # Normalize to 1D and same dtype for isin
    hsc_ids = np.asarray(hsc_ids).ravel()
    if hsc_ids.dtype.kind in ("U", "S", "O"):
        hsc_ids = np.array([str(x).strip() for x in hsc_ids])
    hsc_set = set(hsc_ids)

    with h5py.File(neighbors_path, "r") as f:
        neighbor_ids = f["object_id_hsc"][:]
        source_types = f["source_type"][:]

    neighbor_ids_flat = np.asarray(neighbor_ids).ravel()
    if neighbor_ids_flat.dtype.kind in ("U", "S", "O"):
        neighbor_ids_flat = np.array([str(x).strip() for x in neighbor_ids_flat])
        mask_in_list = np.array([nid in hsc_set for nid in neighbor_ids_flat])
    else:
        mask_in_list = np.isin(neighbor_ids_flat, hsc_ids)
    mask_valid_type = (source_types == 0)
    final_mask = mask_in_list & mask_valid_type

    indices_in_file = np.where(final_mask)[0]
    # matching_ids in same order as indices_in_file (neighbour order)
    matching_ids = neighbor_ids_flat[indices_in_file]
    return indices_in_file, matching_ids, hsc_ids


def load_metadata_by_ids(metadata_path, matching_ids, label_columns, hsc_ids_from_metadata):
    """
    Load label columns for the given matching_ids. Uses hsc_ids_from_metadata to find
    the metadata row index for each ID so that labels are aligned with matching_ids order.
    """
    # Build id -> metadata row index (first occurrence if duplicates)
    id_to_row = {}
    for i, hid in enumerate(hsc_ids_from_metadata):
        key = str(hid).strip() if np.isscalar(hid) else str(hid.ravel()[0]).strip()
        if key not in id_to_row:
            id_to_row[key] = i

    meta_rows = []
    for mid in matching_ids:
        key = str(mid).strip() if np.isscalar(mid) else str(mid.ravel()[0]).strip()
        if key not in id_to_row:
            raise KeyError(f"HSC ID {key!r} not found in metadata hsc_object_id")
        meta_rows.append(id_to_row[key])
    meta_rows = np.array(meta_rows)

    labels = {}
    with h5py.File(metadata_path, "r") as f:
        for col in label_columns:
            if col not in f:
                raise ValueError(f"Metadata missing column: {col}")
            arr = f[col][:].astype(np.float32)
            labels[col] = arr[meta_rows]
    return labels


def main():
    print("Device:", DEVICE)

    # 1. Match metadata <-> neighbours by HSC ID (existing logic)
    print("Finding matches (metadata hsc_object_id vs neighbours object_id_hsc, source_type==0)...")
    indices_in_file, matching_ids, hsc_ids_meta = find_matches(METADATA_PATH, NEIGHBORS_HDF5)
    n_matches = len(indices_in_file)
    print(f"Found {n_matches} matches.")

    if n_matches == 0:
        raise RuntimeError("No matches between metadata and neighbours. Nothing to extract.")

    # 2. Load metadata for matched IDs only (by ID, not by position)
    print(f"Loading metadata for {n_matches} matched IDs...")
    labels_n = load_metadata_by_ids(METADATA_PATH, matching_ids, LABEL_COLUMNS_MMU, hsc_ids_meta)

    # 3. AION model and codec
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DEVICE)
    codec_manager = CodecManager(device=DEVICE)

    all_embeddings = []
    all_embeddings_mean = []
    all_embeddings_mean_legacy = []
    all_embeddings_mean_hsc = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, n_matches, BATCH_SIZE), desc="Encoding"):
            end = min(start + BATCH_SIZE, n_matches)
            indices = indices_in_file[start:end]

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

    assert embeddings.shape[0] == n_matches
    print("Embeddings shape:", embeddings.shape)

    # 4. Save one H5: IDs (traceable), neighbour indices, embeddings, labels — all same order
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="downstream_aion_mmu_matched_", dir=tempfile.gettempdir())
    os.close(tmp_fd)

    # Store HSC IDs as fixed-length bytes for HDF5
    matching_ids_str = np.array([str(x).strip() for x in matching_ids])
    id_dtype = f"S{max(len(s) for s in matching_ids_str)}"
    hsc_object_id_bytes = np.array([s.encode("utf-8") for s in matching_ids_str], dtype=id_dtype)

    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("hsc_object_id", data=hsc_object_id_bytes)
            f.create_dataset("neighbour_row_index", data=indices_in_file)

            f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean", data=embeddings_mean, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_legacy", data=embeddings_mean_legacy, compression="gzip", compression_opts=4)
            f.create_dataset("embeddings_mean_hsc", data=embeddings_mean_hsc, compression="gzip", compression_opts=4)

            for col in LABEL_COLUMNS_MMU:
                f.create_dataset(
                    f"labels/{col}",
                    data=labels_n[col],
                    compression="gzip",
                    compression_opts=4,
                )
            f.attrs["num_examples"] = n_matches
            f.attrs["label_columns"] = list(LABEL_COLUMNS_MMU)
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
        f"Saved: {OUTPUT_H5} (hsc_object_id, neighbour_row_index, embeddings, "
        f"embeddings_mean, embeddings_mean_legacy, embeddings_mean_hsc, labels: {LABEL_COLUMNS_MMU})"
    )


if __name__ == "__main__":
    main()
