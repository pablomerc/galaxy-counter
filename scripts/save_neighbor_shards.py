"""
Iterate through a NeighborsDataset and write pre-baked, pre-padded batches to
compressed HDF5 shard files.  Each shard stores ~CHUNKS_PER_FILE batches so
that later reads are purely sequential (no random-access penalty).

After running this script, combine the shards with merge_neighbor_shards.py.

Usage
-----
Edit the four constants in the "Configuration" block below, then run:

    python scripts/save_neighbor_shards.py

The script will create OUTPUT_DIR if it does not exist and write files named
    neighbors_shard_0000.h5, neighbors_shard_0001.h5, …

If the run crashes mid-way the current buffer is saved as
    neighbors_shard_XXXX_recovery.h5
so you do not lose progress.
"""

import os
import sys
import traceback
from datetime import datetime

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxy_counter.neighbors import NeighborsDataset, collate_neighbors

# ---------------------------------------------------------------------------
# Configuration — edit these for your dataset
#
# To try this on the HSC × Legacy Survey dataset this repo was built on, use:
#   SOURCE_H5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
#   OUTPUT_DIR = "/work1/jeroenaudenaert/pablomer/data/neighbor_batches"
# ---------------------------------------------------------------------------
SOURCE_H5 = "/path/to/neighbours.h5"          # HDF5 built in Step 3
OUTPUT_DIR = "/path/to/neighbor_batches"       # directory for output shards
BATCH_SIZE = 64
NUM_WORKERS = 8      # increase for faster random-read throughput
CHUNKS_PER_FILE = 50 # batches per shard (~3200 samples at batch_size=64)
MAX_NEIGHBORS = 5    # must match the value used when building SOURCE_H5
# ---------------------------------------------------------------------------


def save_shard(shard_idx, buffer, save_dir, global_max_neighbors):
    """Write a buffered list of batches to a single HDF5 shard file."""
    if not buffer:
        return

    filename = os.path.join(save_dir, f"neighbors_shard_{shard_idx:04d}.h5")

    all_targets  = torch.cat([b[0] for b in buffer], dim=0).numpy()
    all_samegals = torch.cat([b[1] for b in buffer], dim=0).numpy()

    # Pad each batch's neighbor dimension to the global maximum before concat.
    all_sameins_list = []
    all_masks_list   = []
    for b in buffer:
        neigh_tensor = b[2]  # (B, N_loc, C, H, W)
        mask_tensor  = b[3]  # (B, N_loc)
        B, N_loc, C, H, W = neigh_tensor.shape

        pad_n = global_max_neighbors - N_loc
        if pad_n > 0:
            new_neigh = torch.zeros((B, global_max_neighbors, C, H, W), dtype=neigh_tensor.dtype)
            new_neigh[:, :N_loc] = neigh_tensor
            new_mask = torch.zeros((B, global_max_neighbors), dtype=mask_tensor.dtype)
            new_mask[:, :N_loc] = mask_tensor
            all_sameins_list.append(new_neigh)
            all_masks_list.append(new_mask)
        else:
            all_sameins_list.append(neigh_tensor)
            all_masks_list.append(mask_tensor)

    all_sameins = torch.cat(all_sameins_list, dim=0).numpy()
    all_masks   = torch.cat(all_masks_list,   dim=0).numpy()

    flat_metadata = [item for b in buffer for item in b[4]]
    meta_idxs     = np.array([m["idx"]                for m in flat_metadata])
    meta_num_same = np.array([m["num_same_instrument"] for m in flat_metadata], dtype=np.int32)
    meta_survey   = np.array([m["anchor_survey"].encode("utf-8") for m in flat_metadata])

    print(f"Saving {filename} | shape: {all_targets.shape} ...")
    with h5py.File(filename, "w") as f:
        f.create_dataset("targets",        data=all_targets,  compression="lzf")
        f.create_dataset("samegals",       data=all_samegals, compression="lzf")
        f.create_dataset("sameins",        data=all_sameins,  compression="lzf")
        f.create_dataset("neighbor_masks", data=all_masks,    compression="lzf")
        f.create_dataset("meta_idx",                   data=meta_idxs)
        f.create_dataset("meta_num_same_instrument",   data=meta_num_same)
        f.create_dataset("meta_survey",                data=meta_survey)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = NeighborsDataset(hdf5_path=SOURCE_H5, max_neighbors=MAX_NEIGHBORS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_neighbors,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    buffer        = []
    shard_counter = 0

    print(f"Starting at {datetime.now().isoformat()}")
    print(f"Output: {OUTPUT_DIR} | total batches: {len(dataloader)}")

    try:
        for batch in tqdm(dataloader):
            buffer.append(batch)
            if len(buffer) >= CHUNKS_PER_FILE:
                save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)
                buffer = []
                shard_counter += 1

        if buffer:
            save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)

        total_shards = shard_counter + (1 if buffer else 0)
        print(f"Done at {datetime.now().isoformat()} | wrote {total_shards} shards.")

    except Exception as e:
        if buffer:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("Saving buffered batches to recovery shard ...", file=sys.stderr)
            save_shard(shard_counter, buffer, OUTPUT_DIR, dataset.max_neighbors)
            normal_path   = os.path.join(OUTPUT_DIR, f"neighbors_shard_{shard_counter:04d}.h5")
            recovery_path = os.path.join(OUTPUT_DIR, f"neighbors_shard_{shard_counter:04d}_recovery.h5")
            os.rename(normal_path, recovery_path)
            print(f"Recovery shard saved: {recovery_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
