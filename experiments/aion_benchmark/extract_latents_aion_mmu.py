"""
Extract AION embeddings for MMU: same index set as prepare_all MMU (first NUM_EXAMPLES_MMU rows
of preprocessed H5), metadata from the preprocessed metadata H5, but load images from neighbours_v2.h5
at those same indices. Saves only physics_mmu + instrument_mmu_legacy + instrument_mmu_hsc labels.

Run from galaxy_model/ or aion_benchmark/:
  python aion_benchmark/extract_latents_aion_mmu.py
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
import torch
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage

# ---------------------------------------------------------------------------
# MMU: same as prepare_all (indices from preprocessed H5, metadata from metadata H5)
# ---------------------------------------------------------------------------
PROCESSED_H5_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
METADATA_PATH_MMU = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5"
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
OUTPUT_H5 = _here / "downstream_aion_mmu.h5"

NUM_EXAMPLES_MMU = 8192

# Labels to keep (from prepare_all.py lines 94-99)
physics_mmu = ["SHAPE_E1", "SHAPE_E2", "SHAPE_R"]
instrument_mmu_legacy = ["EBV"]
instrument_mmu_hsc = ["a_g", "a_i", "a_r", "a_y", "a_z"]
LABEL_COLUMNS_MMU = physics_mmu + instrument_mmu_legacy + instrument_mmu_hsc

BATCH_SIZE = 32
NUM_ENCODER_TOKENS = 1200
NUM_ENCODER_TOKENS_SINGLE = 600
DEVICE = "cuda"


def _load_metadata_mmu(metadata_path, idx_list, metadata_params):
    """Load metadata for idx_list; map via 'indices' in file if present (same as prepare_all)."""
    metadata_dict = {}
    idx_to_meta = None
    with h5py.File(metadata_path, "r") as f:
        if "indices" in f:
            meta_indices = f["indices"][:]
            idx_to_meta = {int(dataset_idx): meta_idx for meta_idx, dataset_idx in enumerate(meta_indices)}
        for param in metadata_params:
            if param in f:
                arr = f[param][:].astype(np.float32)
                if idx_to_meta is not None:
                    arr = arr[[idx_to_meta[i] for i in idx_list]]
                else:
                    arr = arr[idx_list]
                metadata_dict[param] = arr
    return metadata_dict


def _resolve_neighbor_rows(h5_indices, idx_list, neighbours_indices=None):
    """
    Return neighbour H5 row indices in the same order as idx_list.
    If neighbours has 'indices', map by abs_index; else use idx_list as row indices.
    """
    abs_indices = h5_indices[np.asarray(idx_list)]
    if neighbours_indices is not None:
        abs_to_row = {int(a): i for i, a in enumerate(neighbours_indices)}
        out = []
        for a in abs_indices:
            if a not in abs_to_row:
                raise ValueError(f"abs_index {a} not found in neighbours H5 'indices'")
            out.append(abs_to_row[a])
        return np.array(out)
    return np.asarray(idx_list)


def main():
    print("Device:", DEVICE)
    idx_list = list(range(NUM_EXAMPLES_MMU))

    # Metadata from preprocessed metadata H5 (only the columns we care about)
    print(f"Loading MMU metadata from {METADATA_PATH_MMU}...")
    metadata_dict = _load_metadata_mmu(METADATA_PATH_MMU, idx_list, LABEL_COLUMNS_MMU)
    missing = set(LABEL_COLUMNS_MMU) - set(metadata_dict.keys())
    if missing:
        raise ValueError(f"Missing MMU label columns in metadata: {missing}")
    labels_n = {col: metadata_dict[col] for col in LABEL_COLUMNS_MMU}

    # Preprocessed H5: get abs_indices for first NUM_EXAMPLES_MMU rows
    with h5py.File(PROCESSED_H5_PATH, "r") as f:
        h5_indices = f["indices"][:]
    # Resolve neighbour H5 row indices (same objects)
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        neighbours_indices = f["indices"][:] if "indices" in f else None
    neighbour_rows = _resolve_neighbor_rows(h5_indices, idx_list, neighbours_indices)
    print(f"MMU: {NUM_EXAMPLES_MMU} examples, neighbour rows resolved")

    # AION model and codec
    model = AION.from_pretrained("polymathic-ai/aion-base").to(DEVICE)
    codec_manager = CodecManager(device=DEVICE)

    all_embeddings = []
    all_embeddings_mean = []
    all_embeddings_mean_legacy = []
    all_embeddings_mean_hsc = []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, len(neighbour_rows), BATCH_SIZE), desc="Encoding"):
            end = min(start + BATCH_SIZE, len(neighbour_rows))
            indices = neighbour_rows[start:end]

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

    assert embeddings.shape[0] == NUM_EXAMPLES_MMU
    print("Embeddings shape:", embeddings.shape)

    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5", prefix="downstream_aion_mmu_", dir=tempfile.gettempdir())
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
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
            f.attrs["num_examples"] = NUM_EXAMPLES_MMU
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

    print(f"Saved: {OUTPUT_H5} (embeddings, embeddings_mean, embeddings_mean_legacy, embeddings_mean_hsc, labels: {LABEL_COLUMNS_MMU})")


if __name__ == "__main__":
    main()
