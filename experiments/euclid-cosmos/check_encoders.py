"""
Sanity check for encoder_1 and encoder_2 of the trained model.

Checks:
  1. Output shapes are correct
  2. No NaN / Inf values
  3. Embeddings vary across samples (not collapsed to a constant)
  4. encoder_1 and encoder_2 produce different embeddings (different weights)
  5. encoder_1: COSMOS and Euclid embeddings of the SAME galaxy are more
     similar to each other than to random other galaxies (alignment check)

Usage:
    python experiments/euclid-cosmos/check_encoders.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-....ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def cosine_sim(a, b):
    """Mean cosine similarity between paired rows of two (N, D) tensors."""
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    return (a * b).sum(dim=1).mean().item()


def report(name, emb):
    print(f"  {name}")
    print(f"    shape : {tuple(emb.shape)}")
    print(f"    mean  : {emb.mean():.5f}")
    print(f"    std   : {emb.std():.5f}")
    print(f"    min   : {emb.min():.5f}   max : {emb.max():.5f}")
    has_nan = torch.isnan(emb).any().item()
    has_inf = torch.isinf(emb).any().item()
    print(f"    NaN   : {has_nan}   Inf : {has_inf}")
    if emb.std() < 1e-6:
        print(f"    ⚠️  COLLAPSED — all embeddings nearly identical!")
    elif has_nan or has_inf:
        print(f"    ❌  INVALID values detected!")
    else:
        print(f"    ✅  OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--n-samples",  type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # --- Load model ---
    print(f"Loading: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval().to(device)
    torch.set_grad_enabled(False)

    # --- Load a batch ---
    dataset = EuclidCosmosDataset(args.h5)
    n = min(args.n_samples, len(dataset))
    indices = np.random.default_rng(42).choice(len(dataset), size=n, replace=False).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=n,
                        collate_fn=collate_fn, num_workers=2)
    euclid, cosmos, sameins, masks, _ = next(iter(loader))
    euclid  = euclid.to(device)   # (N, 1, H, W)
    cosmos  = cosmos.to(device)   # (N, 1, H, W)
    sameins = sameins.to(device)  # (N, 1, 1, H, W) — dummy neighbor

    print(f"\nBatch: {n} galaxy pairs, Euclid {tuple(euclid.shape)}, COSMOS {tuple(cosmos.shape)}\n")

    # --- Encode ---
    euc_emb1  = model.encoder_1(euclid).flatten(1)   # (N, D)
    cos_emb1  = model.encoder_1(cosmos).flatten(1)
    euc_emb2  = model.encoder_2(euclid).flatten(1)
    cos_emb2  = model.encoder_2(cosmos).flatten(1)

    # --- 1. Shape / stats / NaN checks ---
    print("=" * 55)
    print("1. Embedding statistics")
    print("=" * 55)
    report("encoder_1(Euclid)", euc_emb1)
    report("encoder_1(COSMOS)", cos_emb1)
    report("encoder_2(Euclid)", euc_emb2)
    report("encoder_2(COSMOS)", cos_emb2)

    # --- 2. encoder_1 ≠ encoder_2 ---
    print("\n" + "=" * 55)
    print("2. Are encoder_1 and encoder_2 different?")
    print("=" * 55)
    diff = (euc_emb1 - euc_emb2).abs().mean().item()
    print(f"  Mean |encoder_1(Euclid) - encoder_2(Euclid)| = {diff:.5f}")
    if diff < 1e-6:
        print("  ⚠️  Encoders produce identical outputs — weights may be shared or untrained")
    else:
        print("  ✅  Encoders produce different embeddings")

    # --- 3. encoder_1 alignment: same galaxy should be closer than random ---
    print("\n" + "=" * 55)
    print("3. encoder_1 alignment (same galaxy vs random)")
    print("=" * 55)
    sim_same   = cosine_sim(euc_emb1, cos_emb1)
    # Shuffle COSMOS embeddings to create random pairs
    perm       = torch.randperm(n)
    sim_random = cosine_sim(euc_emb1, cos_emb1[perm])
    print(f"  Cosine similarity — same galaxy  : {sim_same:.4f}")
    print(f"  Cosine similarity — random pair  : {sim_random:.4f}")
    if sim_same > sim_random:
        print(f"  ✅  encoder_1 aligns same-galaxy pairs better than random ({sim_same:.4f} > {sim_random:.4f})")
    else:
        print(f"  ⚠️  encoder_1 does NOT align same-galaxy pairs better than random")
        print(f"      This is expected early in training or if geometric loss was 0.")

    print("\nDone.")


if __name__ == "__main__":
    main()
