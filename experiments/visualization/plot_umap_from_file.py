"""
Generate UMAP plots from saved data (from load_pretrained_model_neighbors.py).
Tune plot appearance here without re-running the heavy UMAP pipeline.

Usage:
  python plot_umap_from_file.py
  python plot_umap_from_file.py --data path/to/umap_both_encoders_zdim16_zoom_avg_data.npz
  python plot_umap_from_file.py --stem umap_both_encoders_zdim16_zoom_avg

Data and metadata are expected in: visualization_scripts/neighbors_visualization/latent_space/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

# ---------- Tune these for quick iteration ----------
FIG_SIZE = (20, 8)
POINT_SIZE = 20
alpha = 1
# Color palette: first 2 = HSC / Legacy, next 5 = pair markers (cycle for same-shape different color)
PALETTE = [
    "#e8c4a0",  # HSC (BG_COLORS from aion_vs_ours_all.py)
    "#8eb8e8",  # Legacy (BG_COLORS from aion_vs_ours_all.py)
    "#70a845",
    "#b460bd",
    "#4aac8d",
    "#c85979",
    "#b49041",
]
COLOR_HSC = PALETTE[0]
COLOR_LEGACY = PALETTE[1]
PAIR_COLORS = PALETTE[2:]  # 5 colors for pairs
SHOW_PAIRS = True
PAIR_MARKER_SIZE = 200
PAIR_LINEWIDTHS = 3
DPI = 150
TITLE_FONTSIZE = 23   # +2 from previous
AXIS_FONTSIZE = 23    # xlabel/ylabel, +4 from previous
LEGEND_FONTSIZE = 21  # +2 from previous
TICK_FONTSIZE = 17    # xticks/yticks, +4 from previous
LEGEND_MARKER_SIZE = 8  # marker size in legend (points)
# Match downstream_evaluation/final/aion_vs_ours_all.py GROUP_LABEL_COLORS
LEGEND_TEXT_COLOR_LEGACY = "#2563a8"
LEGEND_TEXT_COLOR_HSC = "#996515"
# Optional: set to a path to save elsewhere; None = same dir as data, same stem as .npz
OUTPUT_PATH = None
# ----------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LATENT_SPACE_DIR = SCRIPT_DIR / 'neighbors_visualization' / 'latent_space'


class _PairsLegendHandle:
    """Dummy handle for the 'pairs' legend row; handler reads .pair_colors and .pair_markers."""
    pass


class _HandlerPairs(HandlerBase):
    """Draw 4 white shapes centered in the handle box."""
    def __init__(self, pair_markers, **kwargs):
        super().__init__(**kwargs)
        self.pair_markers = pair_markers

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = []
        y_center = ydescent + height / 2
        n = len(self.pair_markers)

        for i, mk in enumerate(self.pair_markers):
            # Distribute the N markers evenly across the entire handle width;
            # (i + 0.5) / n places the i-th marker at the center of the i-th slice
            x = xdescent + width * (i + 0.5) / n

            line = Line2D([x], [y_center], marker=mk, color='white', markerfacecolor='white',
                          markeredgecolor='black', markeredgewidth=1.2, markersize=fontsize * 0.8,
                          linestyle='None', transform=trans)
            artists.append(line)
        return artists


def load_umap_data(data_path: Path):
    """Load *_data.npz or *_data_<suffix>.npz and matching *_metadata[_.<suffix>].json."""
    data_path = Path(data_path).resolve()
    name = data_path.name
    if not data_path.suffix == '.npz' or '_data' not in name:
        raise ValueError("data_path must be a path to a *_data.npz or *_data_<suffix>.npz file")
    if name.endswith('_data.npz'):
        stem = name.replace('_data.npz', '')
        meta_path = data_path.parent / f'{stem}_metadata.json'
    elif '_data_' in name and name.endswith('.npz'):
        # e.g. umap_..._data_1.npz -> stem = umap_..., suffix = 1
        stem = name.rsplit('_data_', 1)[0]
        suffix = name.rsplit('_data_', 1)[1].replace('.npz', '')
        meta_path = data_path.parent / f'{stem}_metadata_{suffix}.json'
    else:
        raise ValueError("data_path must be a path to a *_data.npz or *_data_<suffix>.npz file")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with np.load(data_path, allow_pickle=False) as z:
        hsc_umap_1 = z['hsc_umap_1']
        legacy_umap_1 = z['legacy_umap_1']
        hsc_umap_2 = z['hsc_umap_2']
        legacy_umap_2 = z['legacy_umap_2']
        selected_indices = z['selected_indices']

    with open(meta_path) as f:
        meta = json.load(f)

    # selected_indices may be empty array
    if selected_indices.size == 0:
        selected_indices = None
    return {
        'hsc_umap_1': hsc_umap_1,
        'legacy_umap_1': legacy_umap_1,
        'hsc_umap_2': hsc_umap_2,
        'legacy_umap_2': legacy_umap_2,
        'selected_indices': selected_indices,
        'meta': meta,
        'stem': stem,
        'figures_dir': data_path.parent,
    }


def plot(dat, figsize=FIG_SIZE, point_size=POINT_SIZE,
         color_hsc=COLOR_HSC, color_legacy=COLOR_LEGACY, show_pairs=SHOW_PAIRS,
         pair_colors=None, pair_marker_size=PAIR_MARKER_SIZE, pair_linewidths=PAIR_LINEWIDTHS,
         output_path=None, dpi=DPI):
    meta = dat['meta']
    epoch = meta['epoch']
    selected_indices = dat['selected_indices']
    figures_dir = dat['figures_dir']
    stem = dat['stem']

    if pair_colors is None:
        pair_colors = PAIR_COLORS
    n_pair_colors = len(pair_colors)
    pair_markers = ['x', 's', 'o', '^']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Encoder 1
    ax1.scatter(dat['hsc_umap_1'][:, 0], dat['hsc_umap_1'][:, 1],
                s=point_size, label='HSC', alpha=alpha, c=color_hsc)
    ax1.scatter(dat['legacy_umap_1'][:, 0], dat['legacy_umap_1'][:, 1],
                s=point_size, label='Legacy', alpha=alpha, c=color_legacy)
    if show_pairs and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % n_pair_colors]
            marker = pair_markers[i % 4]
            lw = pair_linewidths
            lw_outline = lw + 2 if marker == 'x' else lw
            # For 'x', draw black outline first (thicker), then colored on top
            if marker == 'x':
                ax1.scatter(dat['hsc_umap_1'][idx, 0], dat['hsc_umap_1'][idx, 1],
                            marker=marker, s=pair_marker_size, c=['black'],
                            linewidths=lw_outline, zorder=4, alpha=1.0)
                ax1.scatter(dat['legacy_umap_1'][idx, 0], dat['legacy_umap_1'][idx, 1],
                            marker=marker, s=pair_marker_size, c=['black'],
                            linewidths=lw_outline, zorder=4, alpha=1.0)
            ax1.scatter(dat['hsc_umap_1'][idx, 0], dat['hsc_umap_1'][idx, 1],
                        marker=marker, s=pair_marker_size, c=[color],
                        linewidths=lw, zorder=5, edgecolors='black', alpha=1.0)
            ax1.scatter(dat['legacy_umap_1'][idx, 0], dat['legacy_umap_1'][idx, 1],
                        marker=marker, s=pair_marker_size, c=[color],
                        linewidths=lw, zorder=5, edgecolors='black', alpha=1.0)
    ax1.set_title(f'Physics Latent Space', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.set_xlabel('UMAP Component 1', fontsize=AXIS_FONTSIZE)
    ax1.set_ylabel('UMAP Component 2', fontsize=AXIS_FONTSIZE)
    ax1.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_hsc, markeredgecolor='black', markersize=LEGEND_MARKER_SIZE, label='HSC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_legacy, markeredgecolor='black', markersize=LEGEND_MARKER_SIZE, label='Legacy'),
    ]
    legend_labels = ['HSC', 'Legacy']
    if show_pairs and selected_indices is not None:
        legend_handles.append(_PairsLegendHandle())
        legend_labels.append('Pairs')
        ax1.legend(handles=legend_handles, labels=legend_labels, fontsize=LEGEND_FONTSIZE,
                   handlelength=4,
                   handler_map={_PairsLegendHandle: _HandlerPairs(pair_markers)})
    else:
        ax1.legend(handles=legend_handles, fontsize=LEGEND_FONTSIZE)
    # ax1.grid(True)

    # Encoder 2
    ax2.scatter(dat['hsc_umap_2'][:, 0], dat['hsc_umap_2'][:, 1],
                s=point_size, label='HSC', alpha=alpha*0.5, c=color_hsc)
    ax2.scatter(dat['legacy_umap_2'][:, 0], dat['legacy_umap_2'][:, 1],
                s=point_size, label='Legacy', alpha=alpha*0.5, c=color_legacy)
    if show_pairs and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % n_pair_colors]
            marker = pair_markers[i % 4]
            lw = pair_linewidths
            lw_outline = lw + 2 if marker == 'x' else lw
            if marker == 'x':
                ax2.scatter(dat['hsc_umap_2'][idx, 0], dat['hsc_umap_2'][idx, 1],
                            marker=marker, s=pair_marker_size, c=['black'],
                            linewidths=lw_outline, zorder=4, alpha=1.0)
                ax2.scatter(dat['legacy_umap_2'][idx, 0], dat['legacy_umap_2'][idx, 1],
                            marker=marker, s=pair_marker_size, c=['black'],
                            linewidths=lw_outline, zorder=4, alpha=1.0)
            ax2.scatter(dat['hsc_umap_2'][idx, 0], dat['hsc_umap_2'][idx, 1],
                        marker=marker, s=pair_marker_size, c=[color],
                        linewidths=lw, zorder=5, edgecolors='black', alpha=1.0)
            ax2.scatter(dat['legacy_umap_2'][idx, 0], dat['legacy_umap_2'][idx, 1],
                        marker=marker, s=pair_marker_size, c=[color],
                        linewidths=lw, zorder=5, edgecolors='black', alpha=1.0)
    ax2.set_title(f'Instrument Latent Space', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax2.set_xlabel('UMAP Component 1', fontsize=AXIS_FONTSIZE)
    ax2.set_ylabel('UMAP Component 2', fontsize=AXIS_FONTSIZE)
    ax2.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_hsc, markeredgecolor='black', markersize=LEGEND_MARKER_SIZE, label='HSC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_legacy, markeredgecolor='black', markersize=LEGEND_MARKER_SIZE, label='Legacy'),
    ]
    legend_labels = ['HSC', 'Legacy']
    if show_pairs and selected_indices is not None:
        legend_handles.append(_PairsLegendHandle())
        legend_labels.append('Pairs')
        ax2.legend(handles=legend_handles, labels=legend_labels, fontsize=LEGEND_FONTSIZE,
                   handlelength=4,
                   handler_map={_PairsLegendHandle: _HandlerPairs(pair_markers)})
    else:
        ax2.legend(handles=legend_handles, fontsize=LEGEND_FONTSIZE)
    # ax2.grid(True)

    plt.tight_layout()

    out = output_path if output_path is not None else (figures_dir / f'{stem}_tuned.png')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi)
    plt.close()
    print(f"Saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Plot UMAP from saved data')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data', type=Path,
                       help='Path to *_data.npz file')
    group.add_argument('--stem', type=str,
                       help='Stem of file (e.g. umap_both_encoders_zdim16_zoom_flat); looks in LATENT_SPACE_DIR')
    parser.add_argument('--no-pairs', action='store_true', help='Do not highlight pairs')
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--out', type=Path, default=None, help='Output path for figure')
    args = parser.parse_args()

    if args.data is not None:
        data_path = Path(args.data)
    elif args.stem is not None:
        data_path = LATENT_SPACE_DIR / f'{args.stem}_data.npz'
    else:
        # Default: use latest/most common stem (e.g. from current default run)
        data_path = LATENT_SPACE_DIR / 'umap_both_encoders_zdim16_zoom_flat_data.npz'
        if not data_path.exists():
            # Fallback: first *_data.npz in dir
            candidates = list(LATENT_SPACE_DIR.glob('*_data.npz'))
            if not candidates:
                raise FileNotFoundError(
                    f"No *_data.npz found in {LATENT_SPACE_DIR}. "
                    "Run load_pretrained_model_neighbors.py with GENERATE_UMAP=True first."
                )
            data_path = candidates[0]
            print(f"Using first found: {data_path}")

    dat = load_umap_data(data_path)
    plot(dat,
         show_pairs=not args.no_pairs,
         dpi=args.dpi,
         output_path=args.out or OUTPUT_PATH)


if __name__ == '__main__':
    main()
