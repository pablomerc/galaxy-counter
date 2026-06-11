"""
Generate and/or re-plot Euclid x COSMOS model samples.

Two modes
─────────
Generate  — needs --checkpoint + --h5. Runs inference, saves an HDF5 with all
            tensors, and immediately plots the figures.
Replot    — needs --data only. Loads the saved HDF5 and re-plots without
            touching the model. Useful for tweaking figure style.

Generate:
    python visualization/generate.py \
        --checkpoint .../best-....ckpt \
        --h5         .../euclid_cosmos_pairs.h5 \
        --out-dir    .../generated \
        --indices    .../test_indices.npy \
        --n-images 16 --num-samples 5 --direction both

Replot (no GPU):
    python visualization/generate.py --data .../generated/generation_data.h5 --all
    python visualization/generate.py --data .../generation_data.h5 --index 0 3 7
    python visualization/generate.py --data .../generation_data.h5 --all --direction cosmos-to-euclid
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root      = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn

# ── Colour scheme ─────────────────────────────────────────────────────────────
COLOR_INPUT  = "#d9d9d9"   # gray  — conditioning image
COLOR_OUTPUT = "#d1efff"   # blue  — generated samples + mean
COLOR_TARGET = "#d0f0c0"   # green — real target


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _show(ax, arr_hw, bg_color=None):
    if bg_color:
        ax.add_patch(mpatches.Rectangle(
            (-0.05, -0.05), 1.1, 1.1, transform=ax.transAxes,
            facecolor=bg_color, edgecolor="none", zorder=-1, clip_on=False,
        ))
    ax.imshow(_scale(arr_hw), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def _inner_label(ax, text):
    ax.text(0.5, 0.96, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            va="top", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                      alpha=0.7, linewidth=0))


def _group_header(fig, axes_row, col_start, col_end, label):
    x0 = axes_row[col_start].get_position().x0
    x1 = axes_row[col_end].get_position().x1
    y  = axes_row[col_start].get_position().y1 + 0.012
    fig.text((x0 + x1) / 2, y + 0.008, label,
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.add_artist(plt.Line2D([x0, x1], [y, y],
                              transform=fig.transFigure,
                              color="black", linewidth=1.2))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_direction(data, indices, out_path):
    """
    One figure per direction.
    Columns: Input (gray) | Sample 1…N (blue) | Mean (blue) | Real target (green)
    """
    N    = len(indices)
    S    = data["samples"].shape[1]
    ncol = 1 + S + 1 + 1

    row_h = 2.2
    head  = 0.55
    fig, axes = plt.subplots(N, ncol,
                             figsize=(2.0 * ncol, row_h * N + head),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.04, left=0.09, right=0.99,
                        bottom=0.02, top=1 - head / (row_h * N + head))

    from_s = data["from"]
    to_s   = data["to"]

    for row, idx in enumerate(indices):
        cond_arr    = data["cond"][idx, 0].numpy()
        target_arr  = data["target"][idx, 0].numpy()
        mean_arr    = data["mean"][idx, 0].numpy()

        _show(axes[row, 0], cond_arr, bg_color=COLOR_INPUT)
        _inner_label(axes[row, 0], from_s)

        for j in range(S):
            _show(axes[row, 1 + j], data["samples"][idx, j, 0].numpy(), bg_color=COLOR_OUTPUT)
            _inner_label(axes[row, 1 + j], f"Sample {j+1}")

        _show(axes[row, 1 + S], mean_arr, bg_color=COLOR_OUTPUT)
        _inner_label(axes[row, 1 + S], "Mean")

        _show(axes[row, 1 + S + 1], target_arr, bg_color=COLOR_TARGET)
        _inner_label(axes[row, 1 + S + 1], f"Real {to_s}")

        axes[row, 0].text(-0.22, 0.5, f"{from_s} → {to_s}",
                          transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9, fontweight="bold")

    fig.canvas.draw()
    _group_header(fig, axes[0], 0, 0, "Input")
    _group_header(fig, axes[0], 1, 1 + S, "Generated")
    _group_header(fig, axes[0], 1 + S + 1, 1 + S + 1, "Target")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_path}")


# ── HDF5 I/O ─────────────────────────────────────────────────────────────────

def _save_direction(h5file, key, cond, target, samples, from_s, to_s):
    grp = h5file.create_group(key)
    grp.create_dataset("cond",    data=cond.numpy(),            compression="gzip")
    grp.create_dataset("target",  data=target.numpy(),          compression="gzip")
    grp.create_dataset("samples", data=samples.numpy(),         compression="gzip")
    grp.create_dataset("mean",    data=samples.mean(1).numpy(), compression="gzip")
    grp.attrs["from_survey"] = from_s
    grp.attrs["to_survey"]   = to_s


def _load_direction(h5file, key):
    if key not in h5file:
        return None
    g = h5file[key]
    return {
        "cond":    torch.from_numpy(g["cond"][:]),
        "target":  torch.from_numpy(g["target"][:]),
        "samples": torch.from_numpy(g["samples"][:]),
        "mean":    torch.from_numpy(g["mean"][:]),
        "from":    g.attrs["from_survey"],
        "to":      g.attrs["to_survey"],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def _generate_samples(model, cond, sameins, masks, num_samples, num_steps, device):
    """Returns (N, num_samples, 1, H, W) CPU tensor."""
    N = cond.shape[0]
    out = []
    for i in range(N):
        c = cond[i:i+1].to(device).repeat(num_samples, 1, 1, 1)
        s = sameins[i:i+1].to(device).repeat(num_samples, 1, 1, 1, 1)
        m = masks[i:i+1].to(device).repeat(num_samples, 1)
        with torch.no_grad():
            gen = model.sample(cond_image_samegal=c, cond_image_sameins=s,
                               masks=m, num_steps=num_steps)
        out.append(gen.cpu().unsqueeze(0))
        print(f"  [{i+1}/{N}]", end="\r")
    print()
    return torch.cat(out, dim=0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()

    # Generate mode
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--h5",          default=None)
    p.add_argument("--out-dir",     default="generated")
    p.add_argument("--indices",     default=None,
                   help=".npy index file to draw samples from (e.g. test_indices.npy).")
    p.add_argument("--n-images",    type=int, default=16)
    p.add_argument("--num-samples", type=int, default=5,
                   help="Samples generated per galaxy.")
    p.add_argument("--num-steps",   type=int, default=100,
                   help="ODE integration steps.")

    # Replot mode
    p.add_argument("--data",    default=None,
                   help="Path to generation_data.h5. If given, skips inference.")
    p.add_argument("--index",   type=int, nargs="+",
                   help="Row indices to plot.")
    p.add_argument("--all",     action="store_true", help="Plot all rows.")

    # Shared
    p.add_argument("--direction", default="both",
                   choices=["cosmos-to-euclid", "euclid-to-cosmos", "both"])
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    # ── Replot mode ───────────────────────────────────────────────────────────
    if args.data is not None:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"File not found: {data_path}")
            sys.exit(1)

        out_dir = data_path.parent
        with h5py.File(data_path, "r") as f:
            n_total = int(f.attrs["num_images"])
            if args.all:
                indices = list(range(n_total))
            elif args.index:
                indices = args.index
            else:
                print("In replot mode specify --all or --index i1 i2 ...")
                sys.exit(1)

            idx_tag = "all" if args.all else "_".join(map(str, indices))

            if args.direction in ("cosmos-to-euclid", "both"):
                data = _load_direction(f, "cosmos_to_euclid")
                if data is None:
                    print("cosmos_to_euclid not found in file, skipping.")
                else:
                    plot_direction(data, indices,
                                   out_dir / f"cosmos_to_euclid_{idx_tag}.png")

            if args.direction in ("euclid-to-cosmos", "both"):
                data = _load_direction(f, "euclid_to_cosmos")
                if data is None:
                    print("euclid_to_cosmos not found in file, skipping.")
                else:
                    plot_direction(data, indices,
                                   out_dir / f"euclid_to_cosmos_{idx_tag}.png")
        return

    # ── Generate mode ─────────────────────────────────────────────────────────
    if not args.checkpoint or not args.h5:
        print("Provide --checkpoint and --h5 to generate, or --data to replot.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval().to(device)

    dataset = EuclidCosmosDataset(args.h5)
    pool    = np.load(args.indices) if args.indices else np.arange(len(dataset))
    rng     = np.random.default_rng(args.seed)
    chosen  = rng.choice(pool, size=min(args.n_images, len(pool)), replace=False).tolist()

    loader = DataLoader(Subset(dataset, chosen), batch_size=args.n_images,
                        shuffle=False, num_workers=2, collate_fn=collate_fn)
    euclid, cosmos, _, masks, _ = next(iter(loader))

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "generation_data.h5"
    indices   = list(range(len(chosen)))

    with h5py.File(data_path, "w") as f:
        f.create_dataset("euclid",  data=euclid.numpy(), compression="gzip")
        f.create_dataset("cosmos",  data=cosmos.numpy(), compression="gzip")
        f.create_dataset("indices", data=np.array(chosen, dtype=np.int64))
        f.attrs["num_images"]  = len(chosen)
        f.attrs["num_samples"] = args.num_samples
        f.attrs["num_steps"]   = args.num_steps
        f.attrs["seed"]        = args.seed

        if args.direction in ("cosmos-to-euclid", "both"):
            print("\nGenerating COSMOS → Euclid...")
            samples = _generate_samples(model, cosmos, cosmos.unsqueeze(1),
                                        masks, args.num_samples, args.num_steps, device)
            _save_direction(f, "cosmos_to_euclid", cosmos, euclid, samples, "COSMOS", "Euclid")
            plot_direction(_load_direction(f, "cosmos_to_euclid"),
                           indices, out_dir / "cosmos_to_euclid.png")

        if args.direction in ("euclid-to-cosmos", "both"):
            print("\nGenerating Euclid → COSMOS...")
            samples = _generate_samples(model, euclid, euclid.unsqueeze(1),
                                        masks, args.num_samples, args.num_steps, device)
            _save_direction(f, "euclid_to_cosmos", euclid, cosmos, samples, "Euclid", "COSMOS")
            plot_direction(_load_direction(f, "euclid_to_cosmos"),
                           indices, out_dir / "euclid_to_cosmos.png")

    print(f"\nData saved: {data_path}")
    print(f"Re-plot anytime: python generate.py --data {data_path} --all")


if __name__ == "__main__":
    main()
