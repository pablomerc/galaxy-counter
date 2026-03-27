"""
Combine the HDF5 shard files produced by save_neighbor_shards.py into HDF5
Virtual Dataset (VDS) files — one for training, one for validation.

A VDS is a thin index file (no data is copied).  Once created, h5py and PyTorch
read through it exactly like a regular HDF5 file, but data is served from the
underlying shards.

Usage
-----
Edit the constants in the "Configuration" block below, then run:

    python scripts/merge_neighbor_shards.py

Outputs
-------
    OUTPUT_DIR/train_neighbors.vds   — all shards except the last
    OUTPUT_DIR/val_neighbors.vds     — last shard only

Point GALAXY_COUNTER_PRECOMPUTED_H5 at the train VDS for training.
"""

import glob
import os

import h5py
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — edit these for your dataset
# ---------------------------------------------------------------------------
INPUT_DIR  = "/path/to/neighbor_batches"  # directory with shard files
OUTPUT_DIR = "/path/to/data"              # where to write the VDS files
# ---------------------------------------------------------------------------


def create_vds(files, output_path):
    """Create a Virtual Dataset that concatenates *files* along the row axis."""
    print(f"Creating VDS {output_path} from {len(files)} files ...")

    with h5py.File(files[0], "r") as f0:
        keys   = list(f0.keys())
        shapes = {k: f0[k].shape[1:] for k in keys}
        dtypes = {k: f0[k].dtype     for k in keys}

    total_rows = 0
    sources = []
    for fn in tqdm(files, desc="Scanning shards"):
        with h5py.File(fn, "r") as f:
            rows = f["targets"].shape[0]
            sources.append((fn, rows))
            total_rows += rows

    print(f"  Total samples: {total_rows}")

    layouts = {
        k: h5py.VirtualLayout(shape=(total_rows,) + shapes[k], dtype=dtypes[k])
        for k in keys
    }

    current_idx = 0
    for fn, rows in tqdm(sources, desc="Mapping VDS"):
        for k in keys:
            vsource = h5py.VirtualSource(fn, k, shape=(rows,) + shapes[k])
            layouts[k][current_idx : current_idx + rows] = vsource
        current_idx += rows

    with h5py.File(output_path, "w", libver="latest") as f:
        for k, layout in layouts.items():
            f.create_virtual_dataset(k, layout)

    print(f"  Saved {output_path}")


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "neighbors_shard_*.h5")))
    if not files:
        raise FileNotFoundError(f"No shard files found in {INPUT_DIR}")

    print(f"Found {len(files)} shards.")

    train_files = files[:-1]
    val_files   = files[-1:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_vds(train_files, os.path.join(OUTPUT_DIR, "train_neighbors.vds"))
    create_vds(val_files,   os.path.join(OUTPUT_DIR, "val_neighbors.vds"))


if __name__ == "__main__":
    main()
