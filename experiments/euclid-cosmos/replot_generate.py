"""
Re-plot generation results from the HDF5 saved by generate.py.
Lets you tweak the figure without re-running the model.

Usage:
    # Plot all rows for both directions
    python experiments/euclid-cosmos/replot_generate.py \
        --data /n03data/fontirro/plots_model/euclid-cosmos-phase1-v2/generated/generation_data.h5 \
        --all

    # Plot specific rows
    python experiments/euclid-cosmos/replot_generate.py \
        --data .../generation_data.h5 \
        --index 0 3 7

    # Plot only one direction
    python experiments/euclid-cosmos/replot_generate.py \
        --data .../generation_data.h5 \
        --all --direction cosmos-to-euclid
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLOR_INPUT  = "#d9d9d9"   # gray  — conditioning image
COLOR_OUTPUT = "#d1efff"   # blue  — generated samples + mean
COLOR_TARGET = "#d0f0c0"   # green — real target image


def _scale(arr):
    """Scale (H, W) array to [0, 1] using 1–99 percentile."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _show(ax, arr_hw, bg_color=None):
    if bg_color:
        rect = mpatches.Rectangle(
            (-0.05, -0.05), 1.1, 1.1,
            transform=ax.transAxes, facecolor=bg_color,
            edgecolor="none", zorder=-1, clip_on=False,
        )
        ax.add_patch(rect)
    ax.imshow(_scale(arr_hw), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def _inner_label(ax, text):
    ax.text(0.5, 0.96, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="black",
            va="top", ha="center",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                      alpha=0.7, linewidth=0))


def _group_header(fig, axes_row, col_start, col_end, label):
    """Draw an underline + label spanning columns col_start..col_end (inclusive)."""
    x0 = axes_row[col_start].get_position().x0
    x1 = axes_row[col_end].get_position().x1
    y  = axes_row[col_start].get_position().y1 + 0.012
    fig.text((x0 + x1) / 2, y + 0.008, label,
             ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.add_artist(plt.Line2D([x0, x1], [y, y],
                              transform=fig.transFigure,
                              color="black", linewidth=1.2))


def load_direction(h5file, key):
    """Load one direction group from the HDF5 file."""
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


def plot_direction(data, indices, out_path):
    """
    One figure for one direction.
    Columns: [Cond input | Sample 1 … N | Mean | Real target]
    """
    N    = len(indices)
    S    = data["samples"].shape[1]
    ncol = 1 + S + 1 + 1

    row_h = 2.2
    head  = 0.55
    fig, axes = plt.subplots(N, ncol,
                             figsize=(2.0 * ncol, row_h * N + head),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.04,
                        left=0.07, right=0.99,
                        bottom=0.02, top=1 - head / (row_h * N + head))

    from_s = data["from"]
    to_s   = data["to"]

    for row, idx in enumerate(indices):
        cond_arr    = data["cond"][idx, 0].numpy()
        target_arr  = data["target"][idx, 0].numpy()
        mean_arr    = data["mean"][idx, 0].numpy()
        sample_arrs = [data["samples"][idx, j, 0].numpy() for j in range(S)]

        _show(axes[row, 0], cond_arr,   bg_color=COLOR_INPUT)
        _inner_label(axes[row, 0], from_s)

        for j, sarr in enumerate(sample_arrs):
            _show(axes[row, 1 + j], sarr, bg_color=COLOR_OUTPUT)
            _inner_label(axes[row, 1 + j], f"Sample {j+1}")

        _show(axes[row, 1 + S], mean_arr,   bg_color=COLOR_OUTPUT)
        _inner_label(axes[row, 1 + S], "Mean")

        _show(axes[row, 1 + S + 1], target_arr, bg_color=COLOR_TARGET)
        _inner_label(axes[row, 1 + S + 1], f"Real {to_s}")

        # Row label
        axes[row, 0].text(-0.22, 0.5, f"{from_s} → {to_s}",
                          transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=9, fontweight="bold")

    # Group headers (only draw after layout is committed)
    fig.canvas.draw()
    _group_header(fig, axes[0], 0, 0, "Input")
    _group_header(fig, axes[0], 1, 1 + S, "Generated")
    _group_header(fig, axes[0], 1 + S + 1, 1 + S + 1, "Target")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      required=True, help="generation_data.h5 from generate.py")
    p.add_argument("--index",     type=int, nargs="+",
                   help="Row indices to plot (0-based within the saved batch).")
    p.add_argument("--all",       action="store_true", help="Plot all rows.")
    p.add_argument("--direction", default="both",
                   choices=["cosmos-to-euclid", "euclid-to-cosmos", "both"])
    p.add_argument("--out-dir",   default=None,
                   help="Output directory. Defaults to the same folder as --data.")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"File not found: {data_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        n_total = int(f.attrs["num_images"])

        if args.all:
            indices = list(range(n_total))
        elif args.index:
            indices = args.index
        else:
            print("Specify --all or --index i1 i2 ...")
            sys.exit(1)

        idx_tag = "all" if args.all else "_".join(map(str, indices))

        if args.direction in ("cosmos-to-euclid", "both"):
            data = load_direction(f, "cosmos_to_euclid")
            if data is None:
                print("cosmos_to_euclid not found in file, skipping.")
            else:
                out = out_dir / f"cosmos_to_euclid_{idx_tag}.png"
                plot_direction(data, indices, out)

        if args.direction in ("euclid-to-cosmos", "both"):
            data = load_direction(f, "euclid_to_cosmos")
            if data is None:
                print("euclid_to_cosmos not found in file, skipping.")
            else:
                out = out_dir / f"euclid_to_cosmos_{idx_tag}.png"
                plot_direction(data, indices, out)


if __name__ == "__main__":
    main()
