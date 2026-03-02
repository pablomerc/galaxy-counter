"""
Build neighbor latent HDF5 for the contrastive dual-encoder baseline.

Outputs the same schema as galaxy_model/neighbor_search/make_latents_all.py:
  - idx
  - index_mmu
  - physics_embedding            (encoder_galaxy on HSC)
  - instrument_embedding         (encoder_instrument on HSC)
  - legacy_physics_embedding     (encoder_galaxy on Legacy)
  - legacy_instrument_embedding  (encoder_instrument on Legacy)
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

_here = Path(__file__).resolve().parent

from galaxy_counter.neighbors import NeighborsSimpleDataset
from galaxy_counter.baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)

NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
DEFAULT_MAX_EXAMPLES = 200_000


def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def _load_model(checkpoint_path: str):
    model = DualEncoderContrastiveModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def generate_embeddings(model, dataset, device, batch_size=256):
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_neighbors,
    )
    hsc_phys, hsc_inst, leg_phys, leg_inst = [], [], [], []
    metadata_collected = []

    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            metadata_collected.extend(meta_list)
            h = hsc_im.to(device)
            l = leg_im.to(device)

            hsc_phys.append(model.encoder_galaxy(h).cpu())
            hsc_inst.append(model.encoder_instrument(h).cpu())
            leg_phys.append(model.encoder_galaxy(l).cpu())
            leg_inst.append(model.encoder_instrument(l).cpu())

    hsc_phys = torch.cat(hsc_phys, dim=0).flatten(start_dim=1).numpy()
    hsc_inst = torch.cat(hsc_inst, dim=0).flatten(start_dim=1).numpy()
    leg_phys = torch.cat(leg_phys, dim=0).flatten(start_dim=1).numpy()
    leg_inst = torch.cat(leg_inst, dim=0).flatten(start_dim=1).numpy()
    return hsc_phys, hsc_inst, leg_phys, leg_inst, metadata_collected


def main():
    p = argparse.ArgumentParser(description="Build contrastive neighbor latents for kNN search.")
    p.add_argument("--checkpoint", required=True, help="Path to contrastive checkpoint (.ckpt)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output HDF5 path (default: contrastive_neighbor_latents_<suffix>.h5)")
    p.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--suffix", default=None, help="Suffix for default output filename")
    args = p.parse_args()

    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(args.max_examples, n_total)
    dataset = Subset(full_dataset, range(n_use))
    print(f"Neighbors dataset: using first {n_use} of {n_total} examples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint).to(device)
    hsc_phys, hsc_inst, leg_phys, leg_inst, metadata = generate_embeddings(
        model, dataset, device, batch_size=args.batch_size
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    index_mmu = np.array([m["index_mmu"] for m in metadata], dtype=np.int64)
    idx = np.arange(n_use, dtype=np.int64)

    if args.suffix is None:
        args.suffix = Path(args.checkpoint).parent.name
    if args.output is None:
        args.output = _here / f"contrastive_neighbor_latents_{args.suffix}.h5"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("idx", data=idx, compression="gzip", compression_opts=4)
        f.create_dataset("index_mmu", data=index_mmu, compression="gzip", compression_opts=4)
        f.create_dataset("physics_embedding", data=hsc_phys.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("instrument_embedding", data=hsc_inst.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("legacy_physics_embedding", data=leg_phys.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("legacy_instrument_embedding", data=leg_inst.astype(np.float32), compression="gzip", compression_opts=4)
        f.attrs["num_examples"] = n_use
        f.attrs["checkpoint"] = str(args.checkpoint)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
