"""
Test the trained Euclid x COSMOS flow-matching model on the held-out test set.

Loads the test indices saved by train.py, generates Euclid images from COSMOS
inputs, and reports MSE. Also saves a figure with sample rows:
    [COSMOS input | Generated Euclid | Real Euclid]

Usage:
    python experiments/euclid-cosmos/testing.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=00-step=100000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --indices    /n03data/fontirro/checkpoints/euclid-cosmos-phase1/test_indices.npy \
        --out        /n03data/fontirro/checkpoints/euclid-cosmos-phase1/test_results.png
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def show_image(ax, img_tensor, title=None):
    img = img_tensor.squeeze().cpu().float().numpy()
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--indices",    required=True, help="test_indices.npy saved by train.py")
    p.add_argument("--out",        default="test_results.png")
    p.add_argument("--n-plot",     type=int, default=8,   help="Galaxy rows to show in figure")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps",  type=int, default=100, help="ODE integration steps")
    p.add_argument("--num-workers",type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # --- Build test dataset from saved indices ---
    test_indices = np.load(args.indices)
    print(f"Test set: {len(test_indices)} galaxies")
    dataset     = EuclidCosmosDataset(args.h5)
    test_subset = Subset(dataset, test_indices.tolist())
    loader      = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # --- Run inference over the full test set ---
    all_mse = []
    plot_euclid_real, plot_euclid_gen, plot_cosmos = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (euclid_real, cosmos, sameins, masks, _) in enumerate(loader):
            euclid_real = euclid_real.to(device)
            cosmos      = cosmos.to(device)
            sameins     = sameins.to(device)
            masks       = masks.to(device)

            euclid_gen = model.sample(
                cond_image_samegal=cosmos,
                cond_image_sameins=sameins,
                masks=masks,
                num_steps=args.num_steps,
            )

            mse = ((euclid_gen - euclid_real) ** 2).mean(dim=(1, 2, 3))  # (B,)
            all_mse.append(mse.cpu())

            # Collect samples for the figure (only from first batch)
            if batch_idx == 0:
                plot_euclid_real = euclid_real.cpu()
                plot_euclid_gen  = euclid_gen.cpu()
                plot_cosmos      = cosmos.cpu()

    all_mse = torch.cat(all_mse)
    print(f"\n=== Test Results ===")
    print(f"N test galaxies : {len(all_mse)}")
    print(f"Mean MSE        : {all_mse.mean():.6f}")
    print(f"Median MSE      : {all_mse.median():.6f}")
    print(f"Std MSE         : {all_mse.std():.6f}")

    # --- Figure ---
    n = min(args.n_plot, len(plot_euclid_real))
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("COSMOS input",     fontsize=9)
    axes[0, 1].set_title("Generated Euclid", fontsize=9)
    axes[0, 2].set_title("Real Euclid",      fontsize=9)

    for i in range(n):
        show_image(axes[i, 0], plot_cosmos[i])
        show_image(axes[i, 1], plot_euclid_gen[i])
        show_image(axes[i, 2], plot_euclid_real[i])

    fig.suptitle(
        f"Test set  |  Mean MSE = {all_mse.mean():.5f}  |  N = {len(all_mse)}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {args.out}")


if __name__ == "__main__":
    main()
