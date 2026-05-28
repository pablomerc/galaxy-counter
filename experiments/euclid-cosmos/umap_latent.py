"""
UMAP visualization of the latent space from the trained Euclid x COSMOS model.

Encodes validation-set images through both encoders and plots:
  - encoder_1 (same-galaxy / physics): Euclid and COSMOS of the SAME galaxies
    should cluster together if the model learned survey-invariant features.
  - encoder_2 (same-instrument): shows instrument-specific structure.

Usage:
    python experiments/euclid-cosmos/umap_latent.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=21-step=98000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --out        /n03data/fontirro/plots_model/euclid-cosmos-phase1/umap.png \
        --n-samples  5000
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
import umap

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--out",        default="umap.png")
    p.add_argument("--indices",    default=None,
                   help="Optional .npy file of indices (e.g. test_indices.npy). "
                        "If omitted, a random sample is used.")
    p.add_argument("--n-samples",  type=int, default=5000,
                   help="Number of galaxy pairs to encode. Set to -1 to use all pairs (ignored if --indices given)")
    p.add_argument("--n-highlight", type=int, default=8,
                   help="Number of random pairs to highlight on encoder_1 plot")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    # --- Build dataset subset ---
    dataset = EuclidCosmosDataset(args.h5)
    if args.indices is not None:
        indices = np.load(args.indices).tolist()
        print(f"Using {len(indices)} indices from {args.indices}")
    else:
        n = len(dataset) if args.n_samples == -1 else min(args.n_samples, len(dataset))
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()
        print(f"Using {'all' if args.n_samples == -1 else n} samples ({n} pairs)")

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # --- Encode all images ---
    euclid_emb1_list, cosmos_emb1_list = [], []
    euclid_emb2_list, cosmos_emb2_list = [], []

    print("Encoding images...")
    with torch.no_grad():
        for euclid, cosmos, _, _, _ in loader:
            euclid = euclid.to(device)   # (B, 1, H, W)
            cosmos = cosmos.to(device)   # (B, 1, H, W)

            euclid_emb1_list.append(model.encoder_1(euclid).flatten(1).cpu())
            cosmos_emb1_list.append(model.encoder_1(cosmos).flatten(1).cpu())
            euclid_emb2_list.append(model.encoder_2(euclid).flatten(1).cpu())
            cosmos_emb2_list.append(model.encoder_2(cosmos).flatten(1).cpu())

    euclid_emb1 = torch.cat(euclid_emb1_list).numpy()
    cosmos_emb1 = torch.cat(cosmos_emb1_list).numpy()
    euclid_emb2 = torch.cat(euclid_emb2_list).numpy()
    cosmos_emb2 = torch.cat(cosmos_emb2_list).numpy()
    N = len(euclid_emb1)
    print(f"Encoded {N} pairs. Embedding dim: {euclid_emb1.shape[1]}")

    # --- UMAP ---
    umap_params = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                       metric="euclidean", random_state=args.seed)

    print("Computing UMAP for encoder_1 (same-galaxy / physics)...")
    all_emb1  = np.concatenate([euclid_emb1, cosmos_emb1], axis=0)
    umap_emb1 = umap.UMAP(**umap_params).fit_transform(all_emb1)
    euc_u1, cos_u1 = umap_emb1[:N], umap_emb1[N:]

    print("Computing UMAP for encoder_2 (same-instrument)...")
    all_emb2  = np.concatenate([euclid_emb2, cosmos_emb2], axis=0)
    umap_emb2 = umap.UMAP(**umap_params).fit_transform(all_emb2)
    euc_u2, cos_u2 = umap_emb2[:N], umap_emb2[N:]

    # --- Pick random pairs to highlight ---
    rng = np.random.default_rng(args.seed)
    pair_ids = rng.choice(N, size=min(args.n_highlight, N), replace=False)
    # Use a colormap so each pair gets a unique color
    pair_colors = plt.cm.tab10(np.linspace(0, 1, len(pair_ids)))

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    kw = dict(s=4, alpha=0.5, rasterized=True)
    ax1.scatter(euc_u1[:, 0], euc_u1[:, 1], c="steelblue",  label="Euclid VIS", **kw)
    ax1.scatter(cos_u1[:, 0], cos_u1[:, 1], c="darkorange", label="COSMOS F115W", **kw)

    # Highlight selected pairs: both Euclid and COSMOS get the same color and symbol
    for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
        label = str(k + 1)
        ax1.scatter(euc_u1[pid, 0], euc_u1[pid, 1], s=80, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        ax1.scatter(cos_u1[pid, 0], cos_u1[pid, 1], s=80, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        # Label both points with the pair number
        for x, y in [(euc_u1[pid, 0], euc_u1[pid, 1]),
                     (cos_u1[pid, 0], cos_u1[pid, 1])]:
            ax1.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                         fontsize=7, color=color, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",  markersize=6, label="Euclid VIS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=6, label="COSMOS F115W"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=8,
               markeredgecolor="black", label=f"{len(pair_ids)} highlighted pairs"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8)
    ax1.set_title("encoder_1 — same-galaxy (physics)\nEuclid & COSMOS should overlap")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    ax2.scatter(euc_u2[:, 0], euc_u2[:, 1], c="steelblue",  label="Euclid VIS", **kw)
    ax2.scatter(cos_u2[:, 0], cos_u2[:, 1], c="darkorange", label="COSMOS F115W", **kw)
    ax2.set_title("encoder_2 — same-instrument\nSurveys may separate")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.legend(markerscale=3)

    fig.suptitle(f"Latent space UMAP  |  N = {N} galaxy pairs", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
