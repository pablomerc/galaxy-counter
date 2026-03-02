"""
Compact neighbor plot for a single query index.

This script reuses the neighbor selections from neighbors_summary.csv (the same
ones used to create query_*.png via search_neighbors.py) and rearranges them
into a focused layout:

- HSC block (top 4 rows):
    - Column 0: Query HSC (same image repeated on each row)
    - Column 1: HSC inst kNN 1,2,3,4
    - Column 2: HSC phys kNN 1,2,3,10

- Legacy block (bottom 4 rows):
    - Column 0: Query Legacy
    - Column 1: Legacy inst kNN 1,2,3,4
    - Column 2: Legacy phys kNN 1,2,3,4

Visualization uses the same per-image percentile clipping + min/max scaling
as tests/triplet_images.py (via tensor_to_rgb), but with images loaded directly
from neighbours_v2.h5 (full 160x160, center-cropped to 64x64).

Run from galaxy_model/:

  python neighbor_search/neighbors_plot.py --query-idx 80

You can override the CSV / HDF5 paths and output path via flags.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

_here = Path(__file__).resolve().parent
from galaxy_counter.preprocessing.image_preprocessing import CenterCrop


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_CSV = _here / "query_results" / "neighbors_summary.csv"
DEFAULT_NEIGHBORS_HDF5 = Path(
    os.getenv("GALAXY_COUNTER_NEIGHBORS_H5", "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5")
)
_CROP_SIZE = 64
_CROPPER = CenterCrop(crop_size=_CROP_SIZE)


# ---------------------------------------------------------------------------
# Visualization helper (triplet_images-style)
# ---------------------------------------------------------------------------

def tensor_to_rgb(tensor: torch.Tensor, channels: Sequence[int] = (0, 1, 2), percentile_clip: float = 99.5) -> np.ndarray:
    """
    Convert a (C,H,W) tensor into an RGB image (H,W,3) using percentile clipping
    and per-image, per-channel min/max normalization.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C,H,W), got {tuple(tensor.shape)}")

    c_indices = list(channels)
    if max(c_indices) >= tensor.shape[0]:
        raise ValueError(
            f"Requested channels {c_indices} but tensor has only {tensor.shape[0]} channels"
        )

    rgb = tensor[c_indices].cpu().numpy()  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))     # (H, W, 3)

    # Percentile clipping
    for i in range(3):
        p_low = np.percentile(rgb[:, :, i], 100 - percentile_clip)
        p_high = np.percentile(rgb[:, :, i], percentile_clip)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_low, p_high)

    # Per-channel min/max normalization
    for i in range(3):
        ch = rgb[:, :, i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            rgb[:, :, i] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb[:, :, i] = 0.0

    return rgb


# ---------------------------------------------------------------------------
# CSV helpers (reuse logic from neighbors_nicer/search_neighbors)
# ---------------------------------------------------------------------------

def _parse_semicolon_ints(s: str) -> List[int]:
    if not s:
        return []
    return [int(x) for x in s.split(";") if x.strip() != ""]


def _parse_semicolon_strs(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip() != ""]


def load_summary_row(summary_csv: Path, query_idx: int) -> dict:
    """Load the neighbors_summary.csv row for a given query_idx."""
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["query_idx"]) == query_idx:
                return row
    raise ValueError(f"query_idx={query_idx} not found in {summary_csv}")


def build_neighbor_lists(row: dict) -> Tuple[
    List[Tuple[int, str]],
    List[Tuple[int, str]],
    List[Tuple[int, str]],
    List[Tuple[int, str]],
]:
    """
    From a CSV row, build 4 neighbor lists:
      - hsc_inst_neighbors, hsc_phys_neighbors, leg_inst_neighbors, leg_phys_neighbors
    Each element is (dataset_idx, source_str) where source_str is 'hsc' or 'legacy'.
    """
    def _pairs(prefix: str) -> List[Tuple[int, str]]:
        idxs = _parse_semicolon_ints(row[f"{prefix}_indices"])
        srcs = _parse_semicolon_strs(row[f"{prefix}_sources"])
        if len(idxs) != len(srcs):
            raise ValueError(
                f"Length mismatch for {prefix}: {len(idxs)} indices vs {len(srcs)} sources"
            )
        return list(zip(idxs, srcs))

    hsc_inst = _pairs("hsc_inst")
    hsc_phys = _pairs("hsc_phys")
    leg_inst = _pairs("leg_inst")
    leg_phys = _pairs("leg_phys")
    return hsc_inst, hsc_phys, leg_inst, leg_phys


# ---------------------------------------------------------------------------
# HDF5 helpers (raw 160x160 -> cropped 64x64)
# ---------------------------------------------------------------------------

def _center_crop_tensor(image: torch.Tensor) -> torch.Tensor:
    """Apply CenterCrop to a single (C,H,W) tensor."""
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(image.shape)}")
    img_batched = image.unsqueeze(0)
    cropped = _CROPPER(img_batched)
    return cropped.squeeze(0)


def get_indexes_mmu(h5_file: h5py.File) -> np.ndarray:
    """
    Reproduce NeighborsSimpleDataset's filtering:
      indexes_mmu = np.where(source_type == 0)[0]
    so dataset indices (0..N-1) map to raw HDF5 rows.
    """
    sources = h5_file["source_type"][:]
    return np.where(sources == 0)[0]


def load_query_images_h5(
    h5_file: h5py.File,
    indexes_mmu: np.ndarray,
    query_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (query_hsc, query_legacy) cropped tensors for a dataset index."""
    if query_idx < 0 or query_idx >= len(indexes_mmu):
        raise IndexError(f"query_idx={query_idx} out of range for indexes_mmu (len={len(indexes_mmu)})")
    raw_idx = int(indexes_mmu[query_idx])
    img_hsc = torch.from_numpy(h5_file["images_hsc"][raw_idx]).float()
    img_legacy = torch.from_numpy(h5_file["images_legacy"][raw_idx]).float()
    img_hsc = _center_crop_tensor(img_hsc)
    img_legacy = _center_crop_tensor(img_legacy)
    return img_hsc, img_legacy


def load_images_for_neighbors_h5(
    h5_file: h5py.File,
    indexes_mmu: np.ndarray,
    neighbor_list: List[Tuple[int, str]],
):
    """
    neighbor_list: list of (dataset_idx, source) where source is 'hsc' or 'legacy'.
    Returns (images, sources, indices) lists of same length.
    """
    if not neighbor_list:
        return [], [], []

    unique_indices = sorted({idx for idx, _ in neighbor_list})
    idx_to_rank = {idx: i for i, idx in enumerate(unique_indices)}

    hsc_list: List[torch.Tensor] = []
    leg_list: List[torch.Tensor] = []
    for dataset_idx in unique_indices:
        if dataset_idx < 0 or dataset_idx >= len(indexes_mmu):
            raise IndexError(
                f"Neighbor dataset_idx={dataset_idx} out of range for indexes_mmu (len={len(indexes_mmu)})"
            )
        raw_idx = int(indexes_mmu[dataset_idx])
        img_hsc = torch.from_numpy(h5_file["images_hsc"][raw_idx]).float()
        img_legacy = torch.from_numpy(h5_file["images_legacy"][raw_idx]).float()
        img_hsc = _center_crop_tensor(img_hsc)
        img_legacy = _center_crop_tensor(img_legacy)
        hsc_list.append(img_hsc)
        leg_list.append(img_legacy)

    images: List[torch.Tensor] = []
    sources: List[str] = []
    indices: List[int] = []
    for dataset_idx, source in neighbor_list:
        r = idx_to_rank[dataset_idx]
        img = hsc_list[r] if source == "hsc" else leg_list[r]
        images.append(img)
        sources.append(source)
        indices.append(dataset_idx)

    return images, sources, indices


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _select_neighbors_with_ranks(
    neighbors: List[Tuple[int, str]],
    positions: Sequence[int],
) -> List[Tuple[int, str, int]]:
    """
    Select neighbors at given 0-based positions and attach 1-based rank.
    Returns list of (dataset_idx, source, rank).
    """
    out: List[Tuple[int, str, int]] = []
    for pos in positions:
        if 0 <= pos < len(neighbors):
            idx, src = neighbors[pos]
            out.append((idx, src, pos + 1))
    return out


def _neighbors_to_images(
    h5_file: h5py.File,
    indexes_mmu: np.ndarray,
    neighbors_with_ranks: List[Tuple[int, str, int]],
):
    """
    Given neighbors_with_ranks = [(idx, source, rank), ...],
    load cropped images and return parallel lists:
      images, sources, indices, ranks
    """
    base_list = [(idx, src) for idx, src, _ in neighbors_with_ranks]
    images, sources, indices = load_images_for_neighbors_h5(h5_file, indexes_mmu, base_list)
    ranks = [r for _, _, r in neighbors_with_ranks]
    return images, sources, indices, ranks


def plot_neighbors_compact(
    query_hsc: torch.Tensor,
    query_legacy: torch.Tensor,
    hsc_inst_imgs: List[torch.Tensor],
    hsc_inst_src: List[str],
    hsc_inst_idx: List[int],
    hsc_inst_rank: List[int],
    hsc_phys_imgs: List[torch.Tensor],
    hsc_phys_src: List[str],
    hsc_phys_idx: List[int],
    hsc_phys_rank: List[int],
    leg_inst_imgs: List[torch.Tensor],
    leg_inst_src: List[str],
    leg_inst_idx: List[int],
    leg_inst_rank: List[int],
    leg_phys_imgs: List[torch.Tensor],
    leg_phys_src: List[str],
    leg_phys_idx: List[int],
    leg_phys_rank: List[int],
    query_idx: int,
    out_path: Path,
):
    """
    Create an 8x3 grid:
      Rows 0-3: HSC block (query HSC, inst, phys)
      Rows 4-7: Legacy block (query Legacy, inst, phys)
    """
    n_rows = 8
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.2))

    def show_img(ax, t: torch.Tensor, title: str | None = None, title_color: str = "black"):
        try:
            rgb = tensor_to_rgb(t)
            ax.imshow(rgb)
        except Exception:
            x = t.detach().cpu().float().numpy()
            if x.ndim == 3:
                x = x[0]
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            ax.imshow(x, cmap="gray")
        if title:
            ax.set_title(title, fontsize=11, color=title_color, fontweight="bold")
        ax.set_axis_off()

    def src_label(s: str) -> str:
        return "(HSC)" if s == "hsc" else "(Leg)"

    # Prepare row-wise access for neighbors, padding with None if needed
    def get_neighbor_for_row(imgs, srcs, idxs, ranks, row, default_rank_list_len):
        if row < len(imgs):
            return imgs[row], srcs[row], idxs[row], ranks[row]
        return None, None, None, None

    # --- HSC block (rows 0-3) ---
    for r in range(4):
        # Column 0: HSC query (same each row)
        ax_q = axes[r, 0]
        show_img(ax_q, query_hsc, "Query HSC" if r == 0 else None)

        # Column 1: HSC phys neighbor
        img_p, src_p, idx_p, rank_p = get_neighbor_for_row(
            hsc_phys_imgs, hsc_phys_src, hsc_phys_idx, hsc_phys_rank, r, 4
        )
        if img_p is not None:
            is_counterpart = (idx_p == query_idx and src_p == "legacy")
            is_cross = (src_p == "legacy")
            color = "red" if is_counterpart else ("gold" if is_cross else "black")
            title = f"HSC phys kNN {rank_p} {src_label(src_p)}"
            show_img(axes[r, 1], img_p, title, title_color=color)
            axes[r, 1].set_xlabel(f"idx {idx_p}", fontsize=9)
        else:
            axes[r, 1].set_axis_off()

        # Column 2: HSC inst neighbor
        img_i, src_i, idx_i, rank_i = get_neighbor_for_row(
            hsc_inst_imgs, hsc_inst_src, hsc_inst_idx, hsc_inst_rank, r, 4
        )
        if img_i is not None:
            is_counterpart = (idx_i == query_idx and src_i == "legacy")
            is_cross = (src_i == "legacy")
            color = "red" if is_counterpart else ("gold" if is_cross else "black")
            title = f"HSC inst kNN {rank_i} {src_label(src_i)}"
            show_img(axes[r, 2], img_i, title, title_color=color)
            axes[r, 2].set_xlabel(f"idx {idx_i}", fontsize=9)
        else:
            axes[r, 2].set_axis_off()

    # --- Legacy block (rows 4-7) ---
    for r in range(4):
        rr = r + 4
        # Column 0: Legacy query
        ax_q = axes[rr, 0]
        show_img(ax_q, query_legacy, "Query Legacy" if r == 0 else None)

        # Column 1: Legacy phys neighbor
        img_p, src_p, idx_p, rank_p = get_neighbor_for_row(
            leg_phys_imgs, leg_phys_src, leg_phys_idx, leg_phys_rank, r, 4
        )
        if img_p is not None:
            is_counterpart = (idx_p == query_idx and src_p == "hsc")
            is_cross = (src_p == "hsc")
            color = "red" if is_counterpart else ("gold" if is_cross else "black")
            title = f"Leg phys kNN {rank_p} {src_label(src_p)}"
            show_img(axes[rr, 1], img_p, title, title_color=color)
            axes[rr, 1].set_xlabel(f"idx {idx_p}", fontsize=9)
        else:
            axes[rr, 1].set_axis_off()

        # Column 2: Legacy inst neighbor
        img_i, src_i, idx_i, rank_i = get_neighbor_for_row(
            leg_inst_imgs, leg_inst_src, leg_inst_idx, leg_inst_rank, r, 4
        )
        if img_i is not None:
            is_counterpart = (idx_i == query_idx and src_i == "hsc")
            is_cross = (src_i == "hsc")
            color = "red" if is_counterpart else ("gold" if is_cross else "black")
            title = f"Leg inst kNN {rank_i} {src_label(src_i)}"
            show_img(axes[rr, 2], img_i, title, title_color=color)
            axes[rr, 2].set_xlabel(f"idx {idx_i}", fontsize=9)
        else:
            axes[rr, 2].set_axis_off()

    fig.suptitle(
        f"Compact neighbors for query_idx={query_idx}\n"
        "HSC block (top 4 rows), Legacy block (bottom 4 rows). "
        "Gold = cross-survey, red = direct counterpart.",
        fontsize=13,
    )
    # Make room at the top for column headers
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Add column supertitles
    fig.canvas.draw()
    col_titles = ["Query", "Physics NNs", "Instrument NNs"]
    for c, col_title in enumerate(col_titles):
        bbox = axes[0, c].get_position()
        x = (bbox.x0 + bbox.x1) / 2.0
        y = bbox.y1 + 0.01
        fig.text(x, y, col_title, ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Thicker vertical separating lines between columns (in figure coordinates)
    bottom = min(ax.get_position().y0 for ax in axes[:, 0])
    top = max(ax.get_position().y1 for ax in axes[:, 0])
    for c in range(1, n_cols):
        x = axes[0, c].get_position().x0
        fig.add_artist(
            Line2D(
                [x, x],
                [bottom, top],
                transform=fig.transFigure,
                linewidth=2.5,
                color="black",
            )
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot a compact neighbor grid for a single query index, using "
            "neighbors_summary.csv and neighbours_v2.h5."
        )
    )
    p.add_argument(
        "--query-idx",
        type=int,
        required=True,
        help="Dataset index used when building neighbors_summary.csv (e.g., 80).",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Path to neighbors_summary.csv (default: {DEFAULT_SUMMARY_CSV})",
    )
    p.add_argument(
        "--neighbors-h5",
        type=Path,
        default=DEFAULT_NEIGHBORS_HDF5,
        help=f"Path to neighbours_v2.h5 (default: {DEFAULT_NEIGHBORS_HDF5})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output figure path "
            "(default: neighbor_search/query_results/query_<idx>_compact.png)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    summary_csv = args.summary_csv
    neighbors_h5 = args.neighbors_h5
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    if not neighbors_h5.is_file():
        raise FileNotFoundError(f"Neighbors HDF5 not found: {neighbors_h5}")

    print(f"Loading summary row for query_idx={args.query_idx} from {summary_csv}")
    row = load_summary_row(summary_csv, args.query_idx)
    hsc_inst_neighbors, hsc_phys_neighbors, leg_inst_neighbors, leg_phys_neighbors = build_neighbor_lists(row)

    # Desired positions (0-based) for each category
    # Drop the last (4th) instrument neighbor so we only show 3.
    hsc_inst_positions = [0, 1, 2]          # kNN 1,2,3
    hsc_phys_positions = [0, 1, 2, 9]       # kNN 1,2,3,10
    leg_inst_positions = [0, 1, 2]          # kNN 1,2,3
    leg_phys_positions = [0, 1, 2, 3]       # kNN 1,2,3,4

    hsc_inst_sel = _select_neighbors_with_ranks(hsc_inst_neighbors, hsc_inst_positions)
    hsc_phys_sel = _select_neighbors_with_ranks(hsc_phys_neighbors, hsc_phys_positions)
    leg_inst_sel = _select_neighbors_with_ranks(leg_inst_neighbors, leg_inst_positions)
    leg_phys_sel = _select_neighbors_with_ranks(leg_phys_neighbors, leg_phys_positions)

    print(f"Opening neighbors HDF5 from {neighbors_h5}")
    with h5py.File(neighbors_h5, "r") as f:
        indexes_mmu = get_indexes_mmu(f)

        print("Loading query images (cropped to 64x64)...")
        query_hsc, query_legacy = load_query_images_h5(f, indexes_mmu, args.query_idx)

        print("Loading selected neighbor images...")
        hsc_inst_imgs, hsc_inst_src, hsc_inst_idx, hsc_inst_rank = _neighbors_to_images(
            f, indexes_mmu, hsc_inst_sel
        )
        hsc_phys_imgs, hsc_phys_src, hsc_phys_idx, hsc_phys_rank = _neighbors_to_images(
            f, indexes_mmu, hsc_phys_sel
        )
        leg_inst_imgs, leg_inst_src, leg_inst_idx, leg_inst_rank = _neighbors_to_images(
            f, indexes_mmu, leg_inst_sel
        )
        leg_phys_imgs, leg_phys_src, leg_phys_idx, leg_phys_rank = _neighbors_to_images(
            f, indexes_mmu, leg_phys_sel
        )

        if args.out is None:
            out_path = _here / "query_results" / f"query_{args.query_idx}_compact.png"
        else:
            out_path = args.out

        print(f"Plotting compact neighbor figure to {out_path} ...")
        plot_neighbors_compact(
            query_hsc,
            query_legacy,
            hsc_inst_imgs,
            hsc_inst_src,
            hsc_inst_idx,
            hsc_inst_rank,
            hsc_phys_imgs,
            hsc_phys_src,
            hsc_phys_idx,
            hsc_phys_rank,
            leg_inst_imgs,
            leg_inst_src,
            leg_inst_idx,
            leg_inst_rank,
            leg_phys_imgs,
            leg_phys_src,
            leg_phys_idx,
            leg_phys_rank,
            args.query_idx,
            out_path,
        )


if __name__ == "__main__":
    main()
