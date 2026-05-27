"""
Test the trained Euclid x COSMOS flow-matching model.

Loads a checkpoint, generates Euclid images from COSMOS inputs, and saves a
figure with rows of: [COSMOS input | Generated Euclid | Real Euclid].

Usage:
    python experiments/euclid-cosmos/testing.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=XX-step=XXXXX.ckpt \
        --h5 /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --out test.png \
        --n-samples 8
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/n03data/fontirro/galaxy-counter/experiments/euclid-cosmos")
sys.path.insert(0, "/n03data/fontirro/galaxy-counter/src")

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def show_image(ax, img_tensor, title=None):
    """Display a (1, H, W) or (H, W) tensor as a grayscale image."""
    img = img_tensor.squeeze().cpu().float().numpy()
    # Robust percentile scaling for display
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--h5", required=True, help="Path to euclid_cosmos_pairs.h5")
    p.add_argument("--out", default="test.png", help="Output figure path")
    p.add_argument("--n-samples", type=int, default=8, help="Number of galaxies to show")
    p.add_argument("--num-steps", type=int, default=100, help="ODE integration steps (fewer = faster, more = better quality)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    print("Model loaded.")

    # --- Load a small subset of the dataset ---
    dataset = EuclidCosmosDataset(args.h5)
    # Take n_samples evenly spaced from the dataset for variety
    indices = np.linspace(0, len(dataset) - 1, args.n_samples, dtype=int)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=args.n_samples, collate_fn=collate_fn)
    euclid_real, cosmos, sameins, masks, _ = next(iter(loader))

    euclid_real = euclid_real.to(device)   # (N, 1, H, W)
    cosmos      = cosmos.to(device)         # (N, 1, H, W)
    sameins     = sameins.to(device)        # (N, 1, 1, H, W)
    masks       = masks.to(device)          # (N, 1)

    # --- Generate Euclid images from COSMOS ---
    print(f"Generating {args.n_samples} images ({args.num_steps} ODE steps)...")
    with torch.no_grad():
        euclid_gen = model.sample(
            cond_image_samegal=cosmos,
            cond_image_sameins=sameins,
            masks=masks,
            num_steps=args.num_steps,
        )
    print("Generation done.")

    # --- Plot ---
    n = args.n_samples
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        show_image(axes[i, 0], cosmos[i],      title="COSMOS input"    if i == 0 else None)
        show_image(axes[i, 1], euclid_gen[i],  title="Generated Euclid" if i == 0 else None)
        show_image(axes[i, 2], euclid_real[i], title="Real Euclid"     if i == 0 else None)

    fig.suptitle(f"Euclid x COSMOS — Phase 1 testing\n{os.path.basename(args.checkpoint)}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")

    # --- Print basic stats ---
    mse = torch.mean((euclid_gen - euclid_real) ** 2).item()
    print(f"Mean pixel MSE (generated vs real): {mse:.6f}")


if __name__ == "__main__":
    main()
