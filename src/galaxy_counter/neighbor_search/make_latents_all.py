"""
Build a large HDF5 of physics and instrument embeddings for the neighbors dataset,
for fast kNN search in latent space.

Uses the same pretrained model and loading convention as
downstream_evaluation/final/prepare_all.py. Loads NeighborsSimpleDataset for the
first N examples (default 200k), runs encoder_1 (physics) and encoder_2 (instrument)
on both HSC and Legacy images, and saves:
  - index_mmu: row index in the original neighbours HDF5 (for loading images later)
  - idx: position in this subset (0 .. N-1)
  - physics_embedding: encoder_1(hsc), shape (N, D1)
  - instrument_embedding: encoder_2(hsc), shape (N, D2)
  - legacy_physics_embedding: encoder_1(legacy), shape (N, D1)
  - legacy_instrument_embedding: encoder_2(legacy), shape (N, D2)

Run from galaxy_model/:
  python neighbor_search/make_latents_all.py --checkpoint PATH --module double_train_fm_neighbors.py
"""
import importlib
from pathlib import Path

_here = Path(__file__).resolve().parent

import argparse
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

# Same paths as prepare_all.py
NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
DEFAULT_MAX_EXAMPLES = 200_000
DEFAULT_MODULE = "double_train_fm_neighbors.py"


def _load_model(checkpoint_path, module_filename):
    """Load double-encoder model from checkpoint (same as prepare_all.py)."""
    module_stem = Path(module_filename).stem
    mod = importlib.import_module(f"galaxy_counter.models.{module_stem}")
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def generate_physics_instrument_embeddings(model, dataset, device, batch_size=256):
    """Run encoder_1 (physics) and encoder_2 (instrument) on HSC and Legacy; return arrays + metadata."""
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_neighbors,
    )
    hsc_physics_list, hsc_instrument_list = [], []
    leg_physics_list, leg_instrument_list = [], []
    metadata_collected = []
    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            metadata_collected.extend(meta_list)
            h = hsc_im.to(device)
            l = leg_im.to(device)
            hsc_physics_list.append(model.encoder_1(h).cpu())
            hsc_instrument_list.append(model.encoder_2(h).cpu())
            leg_physics_list.append(model.encoder_1(l).cpu())
            leg_instrument_list.append(model.encoder_2(l).cpu())
    hsc_physics = torch.cat(hsc_physics_list, dim=0).flatten(start_dim=1).numpy()
    hsc_instrument = torch.cat(hsc_instrument_list, dim=0).flatten(start_dim=1).numpy()
    leg_physics = torch.cat(leg_physics_list, dim=0).flatten(start_dim=1).numpy()
    leg_instrument = torch.cat(leg_instrument_list, dim=0).flatten(start_dim=1).numpy()
    return hsc_physics, hsc_instrument, leg_physics, leg_instrument, metadata_collected


def main():
    p = argparse.ArgumentParser(
        description="Build neighbor latent file (physics + instrument embeddings) for kNN search."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.ckpt), same as prepare_all.py",
    )
    p.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"Module filename (default: {DEFAULT_MODULE})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (default: neighbor_search/neighbor_latents_<suffix>.h5)",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=f"Max examples from neighbors dataset (default: {DEFAULT_MAX_EXAMPLES})",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--suffix", default=None, help="Suffix for output filename (default: from checkpoint dir)")
    args = p.parse_args()

    from galaxy_counter.neighbors import NeighborsSimpleDataset

    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(args.max_examples, n_total)
    dataset = Subset(full_dataset, range(n_use))
    print(f"Neighbors dataset: using first {n_use} of {n_total} examples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, args.module).to(device)
    print("Generating physics (encoder_1) and instrument (encoder_2) embeddings on HSC and Legacy...")
    hsc_physics, hsc_instrument, leg_physics, leg_instrument, metadata = generate_physics_instrument_embeddings(
        model, dataset, device, batch_size=args.batch_size
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # index_mmu: row in original neighbours HDF5 (for loading images in search_neighbors.py)
    index_mmu = np.array([m["index_mmu"] for m in metadata], dtype=np.int64)
    idx = np.arange(n_use, dtype=np.int64)

    if args.suffix is None:
        args.suffix = Path(args.checkpoint).parent.parent.name
    if args.output is None:
        args.output = _here / f"neighbor_latents_{args.suffix}.h5"
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("idx", data=idx, compression="gzip", compression_opts=4)
        f.create_dataset("index_mmu", data=index_mmu, compression="gzip", compression_opts=4)
        f.create_dataset("physics_embedding", data=hsc_physics.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("instrument_embedding", data=hsc_instrument.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("legacy_physics_embedding", data=leg_physics.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("legacy_instrument_embedding", data=leg_instrument.astype(np.float32), compression="gzip", compression_opts=4)
        f.attrs["num_examples"] = n_use
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["module"] = args.module
    print(f"Saved: {args.output} (idx, index_mmu, physics/instrument/legacy_physics/legacy_instrument embeddings)")


if __name__ == "__main__":
    main()
