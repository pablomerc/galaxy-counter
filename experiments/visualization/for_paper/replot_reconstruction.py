"""
Quickly replot reconstruction results from saved data.
Reads from the HDF5 file created by reconstruction_pretrained_neighbor.py
and generates plots for specific example indices.
"""

import sys
from pathlib import Path
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py

# ===== Configuration =====

# Default data path
DEFAULT_DATA_PATH = Path(__file__).parent / 'reconstruction_outputs' / 'reconstruction_data.h5'

# Colors
COLOR_TARGET  = '#d0f0c0'  # Green
COLOR_INPUT   = '#d9d9d9'  # Medium Gray
COLOR_OUTPUT  = '#d1efff'  # Light Blue

# ===== Helper Functions =====

def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> torch.Tensor:
    """
    Scale a (3,H,W) tensor to (H,W,3) in [0,1] using fixed per-channel vmin/vmax.
    """
    x = x_chw[:3]  # Take first 3 channels (RGB)
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    return y.permute(1, 2, 0)


def _create_neighbor_grid(neighbors_kchw: torch.Tensor, vmin, vmax, padding=1):
    """
    Stitch up to 4 neighbors into a 2x2 grid image.
    """
    # Limit to 4 neighbors
    k = min(neighbors_kchw.shape[0], 4)
    c, h, w = neighbors_kchw.shape[1:]

    # Create canvas: (2*h + padding) x (2*w + padding)
    grid_h = 2 * h + padding
    grid_w = 2 * w + padding
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.float32)

    # Positions for 2x2 grid: (row, col)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i in range(k):
        img = _row_scale_rgb(neighbors_kchw[i], vmin, vmax).numpy() # (H, W, 3)
        row, col = positions[i]

        # Calculate start coordinates
        y_start = row * (h + padding)
        x_start = col * (w + padding)

        canvas[y_start : y_start+h, x_start : x_start+w, :] = img

    return canvas


def load_data(data_path):
    """Load reconstruction data from HDF5 file."""
    print(f"Loading data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        # Load tensors
        targets = torch.from_numpy(f['targets'][:])
        samegals = torch.from_numpy(f['samegals'][:])
        sameins = torch.from_numpy(f['sameins'][:])
        masks = torch.from_numpy(f['masks'][:])
        samples = torch.from_numpy(f['samples'][:])
        mean_samples = torch.from_numpy(f['mean_samples'][:])

        # Load metadata
        surveys = [s.decode('utf-8') if isinstance(s, bytes) else str(s)
                   for s in f['anchor_surveys'][:]]
        indices = f['indices'][:]
        num_same = f['num_same_instrument'][:]

        # Get shape info
        batch_size = f.attrs['batch_size']
        num_samples_per_example = f.attrs['num_samples']

    metadata = [
        {
            'anchor_survey': surveys[i],
            'idx': int(indices[i]),
            'num_same_instrument': int(num_same[i])
        }
        for i in range(batch_size)
    ]

    return targets, samegals, sameins, masks, samples, mean_samples, metadata


def add_inner_label(ax, text):
    """
    Add text inside the plot at the top.
    """
    ax.text(
        0.5, 0.96, text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        color='black',
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.7, linewidth=0)
    )

def add_bottom_label(ax, text):
    """
    Add small text inside the plot at the bottom center.
    """
    ax.text(
        0.5, 0.04, text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight='bold',
        color='white',
        verticalalignment='bottom',
        horizontalalignment='center',
        alpha=0.7
    )

def add_highlight(ax, color, zorder_bg):
    """
    Adds a colored background box behind the image.
    zorder_bg determines the stacking order (Gray < Blue < Green).
    """
    rect = patches.Rectangle(
        (-0.05, -0.05), 1.1, 1.1,
        transform=ax.transAxes,
        facecolor=color,
        edgecolor='none',
        zorder=zorder_bg,
        clip_on=False
    )
    ax.add_patch(rect)


def draw_group_headers(fig, axes_row):
    """
    Draws the 'Inputs', 'Target', and 'Output' supertitles and boxes.
    Updated to match new column order: Inputs | Target | Samples | Mean
    """
    y_offset_title = 0.01
    line_width = 1.5

    # --- Inputs Group (Col 0 & 1) ---
    ax_in_start = axes_row[0]
    ax_in_end   = axes_row[1]
    bbox_in_0 = ax_in_start.get_position()
    bbox_in_1 = ax_in_end.get_position()

    x_mid_in = (bbox_in_0.x0 + bbox_in_1.x1) / 2.0
    y_top = bbox_in_0.y1 + 0.015

    fig.text(x_mid_in, y_top + y_offset_title, "Inputs", ha='center', va='bottom', fontsize=12, fontweight='bold')
    line_in = plt.Line2D([bbox_in_0.x0, bbox_in_1.x1], [y_top, y_top], transform=fig.transFigure, color='black', linewidth=line_width)
    fig.add_artist(line_in)

    # --- Target Group (Col 2) ---
    ax_tgt = axes_row[2]
    bbox_tgt = ax_tgt.get_position()

    x_mid_tgt = (bbox_tgt.x0 + bbox_tgt.x1) / 2.0

    fig.text(x_mid_tgt, y_top + y_offset_title, "Target", ha='center', va='bottom', fontsize=12, fontweight='bold')
    line_tgt = plt.Line2D([bbox_tgt.x0, bbox_tgt.x1], [y_top, y_top], transform=fig.transFigure, color='black', linewidth=line_width)
    fig.add_artist(line_tgt)

    # --- Output Group (Col 3, 4, 5) ---
    # Now covers Sample 1 (3), Sample 2 (4), and Mean (5)
    ax_out_start = axes_row[3]
    ax_out_end   = axes_row[5]
    bbox_out_start = ax_out_start.get_position()
    bbox_out_end   = ax_out_end.get_position()

    x_mid_out = (bbox_out_start.x0 + bbox_out_end.x1) / 2.0

    label_out = r"Output: $p(x \mid z_{\mathrm{phy}}, \{z_{\mathrm{ins}}^i\}_{1}^N)$"
    fig.text(x_mid_out, y_top + y_offset_title, label_out, ha='center', va='bottom', fontsize=12, fontweight='bold')
    line_out = plt.Line2D([bbox_out_start.x0, bbox_out_end.x1], [y_top, y_top], transform=fig.transFigure, color='black', linewidth=line_width)
    fig.add_artist(line_out)


def plot_row(ax_row, idx, target, samegal, sameins_stack, samples, mean_sample, metadata, vmin, vmax):
    """
    Helper to plot a single row.
    New Order: SameGal | SameIns | Target | Sample 1 | Sample 2 | Mean
    """

    # Determine Labels based on Metadata
    survey_target = metadata.get('anchor_survey', 'unknown')

    if survey_target == 'hsc':
        input_label = "Legacy"
        target_label = "HSC"
    elif survey_target == 'legacy':
        input_label = "HSC"
        target_label = "Legacy"
    else:
        input_label = "?"
        target_label = "?"

    # Z-Order logic for backgrounds (Gray < Blue < Green)
    # Using negative values so they stay behind images (usually z=0)
    Z_GRAY = -3
    Z_BLUE = -2
    Z_GREEN = -1

    # 1. SameGal (Input) -> Gray
    samegal_vis = _row_scale_rgb(samegal[:3], vmin, vmax).numpy()
    ax_row[0].imshow(samegal_vis)
    ax_row[0].set_axis_off()
    add_inner_label(ax_row[0], "SameGal")
    add_bottom_label(ax_row[0], input_label)
    add_highlight(ax_row[0], COLOR_INPUT, Z_GRAY)

    # 2. SameIns Grid (Input) -> Gray
    sameins_grid_vis = _create_neighbor_grid(sameins_stack, vmin, vmax)
    ax_row[1].imshow(sameins_grid_vis)
    ax_row[1].set_axis_off()
    add_inner_label(ax_row[1], "SameIns")
    add_bottom_label(ax_row[1], target_label)
    add_highlight(ax_row[1], COLOR_INPUT, Z_GRAY)

    # 3. Target (Target) -> Green (Highest Priority)
    target_vis = _row_scale_rgb(target[:3], vmin, vmax).numpy()
    ax_row[2].imshow(target_vis)
    ax_row[2].set_axis_off()
    add_inner_label(ax_row[2], "Target")
    add_bottom_label(ax_row[2], target_label)
    add_highlight(ax_row[2], COLOR_TARGET, Z_GREEN)

    # 4. Sample 1 (Output) -> Blue
    sample1_vis = _row_scale_rgb(samples[0, :3], vmin, vmax).numpy()
    ax_row[3].imshow(sample1_vis)
    ax_row[3].set_axis_off()
    add_inner_label(ax_row[3], "Sample 1")
    add_highlight(ax_row[3], COLOR_OUTPUT, Z_BLUE)

    # 5. Sample 2 (Output) -> Blue
    sample2_vis = _row_scale_rgb(samples[1, :3], vmin, vmax).numpy()
    ax_row[4].imshow(sample2_vis)
    ax_row[4].set_axis_off()
    add_inner_label(ax_row[4], "Sample 2")
    add_highlight(ax_row[4], COLOR_OUTPUT, Z_BLUE)

    # 6. Mean (Output) -> Blue (Moved to end)
    mean_vis = _row_scale_rgb(mean_sample[:3], vmin, vmax).numpy()
    ax_row[5].imshow(mean_vis)
    ax_row[5].set_axis_off()
    add_inner_label(ax_row[5], "Mean")
    add_highlight(ax_row[5], COLOR_OUTPUT, Z_BLUE)

    # Add row label on the far left with "->" arrow notation
    row_desc = f"{input_label} $\\to$ {target_label}"
    ax_row[0].text(-0.25, 0.5, row_desc,
                     transform=ax_row[0].transAxes,
                     ha='right', va='center', fontsize=11, weight='bold')

def plot_examples(indices, targets, samegals, sameins, samples, mean_samples, metadata, output_path):
    """
    Plot reconstruction for one or more examples.
    """
    num_cols = 6
    num_rows = len(indices)

    # Tighter vertical height
    row_height = 2.1
    header_space = 0.6
    total_height = (num_rows * row_height) + header_space

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(num_cols * 2.0, total_height),
                             squeeze=False)

    for i, idx in enumerate(indices):
        target = targets[idx]
        samegal = samegals[idx]
        sameins_stack = sameins[idx]
        current_samples = samples[idx]
        mean_sample = mean_samples[idx]

        target_chw = target[:3]
        vmin = target_chw.amin(dim=(1, 2))
        vmax = target_chw.amax(dim=(1, 2))

        plot_row(axes[i], idx, target, samegal, sameins_stack, current_samples, mean_sample, metadata[idx], vmin, vmax)

    # Recalculate top fraction
    top_frac = 1.0 - (header_space / total_height)

    # Tighter spacing
    plt.subplots_adjust(
        wspace=0.05,
        hspace=0.05,
        left=0.05,
        right=0.99,
        bottom=0.05,
        top=top_frac
    )

    draw_group_headers(fig, axes[0])

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Replot reconstruction results')
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument('--info', action='store_true', help='Show dataset info only')
    parser.add_argument('--index', type=int, nargs='+', help='One or more indices to plot (e.g., --index 1 2)')
    parser.add_argument('--all', action='store_true', help='Plot all examples')
    parser.add_argument('--output', type=str, help='Custom output filename')

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    targets, samegals, sameins, masks, samples, mean_samples, metadata = load_data(data_path)

    if args.info:
        print(f"Loaded {len(targets)} examples.")
        return

    output_dir = data_path.parent
    indices = []

    if args.all:
        indices = list(range(len(targets)))
        default_name = 'reconstruction_all.png'
    elif args.index is not None:
        indices = args.index
        default_name = f'reconstruction_{"_".join(map(str, indices))}.png'
    else:
        print("Please specify --index [i1 i2 ...] or --all")
        sys.exit(1)

    output_path = args.output if args.output else output_dir / default_name

    plot_examples(indices, targets, samegals, sameins, samples, mean_samples, metadata, output_path)

if __name__ == "__main__":
    main()
