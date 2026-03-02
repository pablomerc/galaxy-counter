"""
Given an index into the neighbor latent file, show the HSC and Legacy query images (row 0, centered),
then for each query separately:
  - HSC query: top 3 kNN in instrument space and top 3 kNN in physics space (search over BOTH surveys).
  - Legacy query: top 3 kNN in instrument space and top 3 kNN in physics space (search over BOTH surveys).

Physics (and instrument) kNN are computed in a combined HSC+Legacy embedding space, so neighbors
can come from either survey. The plot labels each neighbor with where it was found, e.g. "(HSC)" or "(Legacy)".

Requires latents file with legacy_physics_embedding and legacy_instrument_embedding (run make_latents_all.py).
Run from galaxy_model/:
  python neighbor_search/search_neighbors.py --latents neighbor_search/neighbor_latents_<suffix>.h5
  python neighbor_search/search_neighbors.py --latents ... --index 5 --out query_5.png
"""
from pathlib import Path
from typing import Optional
import csv
import os

_here = Path(__file__).resolve().parent

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
K_NEIGHBORS = 30
N_COLS = 32  # first row: Query HSC and Query Legacy (cols 2, 3); other rows: K_NEIGHBORS images at cols 1..K_NEIGHBORS


def _list_to_str(xs):
    """Convert list of values to a semicolon-separated string for CSV storage."""
    return ";".join(str(x) for x in xs)


def load_images_for_indices(indices, hdf5_path=NEIGHBORS_HDF5):
    """Load (HSC, Legacy) preprocessed images for dataset indices via NeighborsSimpleDataset."""
    from galaxy_counter.neighbors import NeighborsSimpleDataset
    dataset = NeighborsSimpleDataset(hdf5_path=hdf5_path)
    hsc_list, leg_list = [], []
    for idx in indices:
        hsc, leg, _ = dataset[idx]
        hsc_list.append(hsc)
        leg_list.append(leg)
    return hsc_list, leg_list


def load_images_for_neighbors(neighbor_list, hdf5_path=NEIGHBORS_HDF5):
    """
    neighbor_list: list of (dataset_idx, source) where source is 'hsc' or 'legacy'.
    Returns (images_list, sources_list, indices_list) of same length.
    """
    unique_indices = sorted(set(idx for idx, _ in neighbor_list))
    idx_to_rank = {idx: i for i, idx in enumerate(unique_indices)}
    hsc_list, leg_list = load_images_for_indices(unique_indices, hdf5_path)
    images = []
    sources = []
    indices = []
    for dataset_idx, source in neighbor_list:
        r = idx_to_rank[dataset_idx]
        img = hsc_list[r] if source == "hsc" else leg_list[r]
        images.append(img)
        sources.append(source)
        indices.append(dataset_idx)
    return images, sources, indices


def tensor_to_display(t, channel=0):
    """Convert normalized tensor (C,H,W) to (H,W) for imshow; use first channel and simple scaling."""
    x = t.numpy()
    if x.ndim == 3:
        x = x[channel]
    x = np.clip(x, -3, 3)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def tensor_to_rgb_row_scaled(t, vmin, vmax):
    """
    Convert tensor (C,H,W) to RGB (H,W,3) using per-channel vmin/vmax,
    following the row-scaled visualization logic from double_train_fm.
    All images in a figure share the same vmin/vmax (taken from the query HSC).
    """
    x = t.numpy()
    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C,H,W), got {x.shape}")

    # Use first 3 channels as RGB; if fewer than 3 channels, fall back to grayscale.
    if x.shape[0] < 3:
        gray = tensor_to_display(t)
        return np.stack([gray, gray, gray], axis=-1)

    x = x[:3]  # (3,H,W)
    vmin = np.asarray(vmin).reshape(3, 1, 1)
    vmax = np.asarray(vmax).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    # (3,H,W) -> (H,W,3)
    return np.moveaxis(y, 0, 2)


def plot_query_and_neighbors(
    query_hsc,
    query_legacy,
    hsc_inst_images,
    hsc_inst_sources,
    hsc_inst_indices,
    hsc_phys_images,
    hsc_phys_sources,
    hsc_phys_indices,
    leg_inst_images,
    leg_inst_sources,
    leg_inst_indices,
    leg_phys_images,
    leg_phys_sources,
    leg_phys_indices,
    query_idx,
    out_path,
):
    """
    One figure: row 0 centered = Query HSC, Query Legacy.
    Rows 1–4: NNs from combined HSC+Legacy space; each title shows where found, e.g. "HSC phys kNN 1 (Legacy)".
    """
    n_cols = N_COLS
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.2))
    for ax in axes.flat:
        ax.set_axis_off()

    # --- Per-channel vmin/vmax from query HSC ---
    q = query_hsc.numpy()
    if q.ndim != 3:
        raise ValueError(f"Expected query_hsc with shape (C,H,W), got {q.shape}")
    if q.shape[0] < 3:
        use_rgb = False
        vmin = vmax = None
    else:
        use_rgb = True
        q_ch = q[:3]
        q_flat = q_ch.reshape(3, -1)
        vmin = q_flat.min(axis=1)
        vmax = q_flat.max(axis=1)

    def show_img(ax, t, title=None, title_color="black"):
        if use_rgb:
            ax.imshow(tensor_to_rgb_row_scaled(t, vmin, vmax))
        else:
            ax.imshow(tensor_to_display(t), cmap="gray")
        if title:
            ax.set_title(title, fontsize=8, color=title_color)
        ax.set_axis_off()

    # Row 0 (centered): Query HSC at col 2, Query Legacy at col 3
    show_img(axes[0, 2], query_hsc, "Query HSC")
    show_img(axes[0, 3], query_legacy, "Query Legacy")

    def src_label(s):
        return "(HSC)" if s == "hsc" else "(Leg)"

    # Row 1: HSC query → instrument kNN
    for j, img in enumerate(hsc_inst_images):
        is_counterpart = hsc_inst_indices[j] == query_idx and hsc_inst_sources[j] == "legacy"
        is_cross_survey = hsc_inst_sources[j] == "legacy"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[1, 1 + j]
        show_img(
            ax,
            img,
            f"HSC inst kNN {j+1} {src_label(hsc_inst_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {hsc_inst_indices[j]}", fontsize=7)

    # Row 2: HSC query → physics kNN
    for j, img in enumerate(hsc_phys_images):
        is_counterpart = hsc_phys_indices[j] == query_idx and hsc_phys_sources[j] == "legacy"
        is_cross_survey = hsc_phys_sources[j] == "legacy"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[2, 1 + j]
        show_img(
            ax,
            img,
            f"HSC phys kNN {j+1} {src_label(hsc_phys_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {hsc_phys_indices[j]}", fontsize=7)

    # Row 3: Legacy query → instrument kNN
    for j, img in enumerate(leg_inst_images):
        is_counterpart = leg_inst_indices[j] == query_idx and leg_inst_sources[j] == "hsc"
        is_cross_survey = leg_inst_sources[j] == "hsc"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[3, 1 + j]
        show_img(
            ax,
            img,
            f"Leg inst kNN {j+1} {src_label(leg_inst_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {leg_inst_indices[j]}", fontsize=7)

    # Row 4: Legacy query → physics kNN
    for j, img in enumerate(leg_phys_images):
        is_counterpart = leg_phys_indices[j] == query_idx and leg_phys_sources[j] == "hsc"
        is_cross_survey = leg_phys_sources[j] == "hsc"
        if is_counterpart:
            color = "red"
        elif is_cross_survey:
            color = "gold"
        else:
            color = "black"
        ax = axes[4, 1 + j]
        show_img(
            ax,
            img,
            f"Leg phys kNN {j+1} {src_label(leg_phys_sources[j])}",
            title_color=color,
        )
        ax.set_xlabel(f"idx {leg_phys_indices[j]}", fontsize=7)

    plt.suptitle("Physics/inst NNs searched over BOTH surveys; label shows where found (HSC/Leg)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Show query image and top-k neighbors in instrument and physics space."
    )
    p.add_argument("--latents", type=Path, default=None, help="Path to neighbor_latents_*.h5 (default: single neighbor_latents_*.h5 in script dir)")
    p.add_argument("--index", type=int, default=10, help="Dataset index (0 .. N-1); default 0")
    p.add_argument(
        "--batch-start",
        type=int,
        default=None,
        help="If set, run for all indices from batch-start to batch-end (inclusive).",
    )
    p.add_argument(
        "--batch-end",
        type=int,
        default=None,
        help="Used with --batch-start; if omitted, runs from batch-start up to N-1.",
    )
    p.add_argument("--out", type=Path, default=None, help="Output figure path (single run) or output directory (batch mode).")
    p.add_argument("--neighbors-h5", type=str, default=NEIGHBORS_HDF5, help="Neighbors HDF5 for loading images")
    args = p.parse_args()

    latents_path = Path(args.latents) if args.latents is not None else None
    if latents_path is None:
        candidates = list(_here.glob("neighbor_latents_*.h5"))
        if len(candidates) == 1:
            latents_path = candidates[0]
            print(f"Using latents file: {latents_path}")
        elif not candidates:
            raise FileNotFoundError("No neighbor_latents_*.h5 found in neighbor_search/; run make_latents_all.py first or pass --latents")
        else:
            raise FileNotFoundError(f"Multiple neighbor_latents_*.h5 found: {candidates}; pass --latents explicitly")
    if not latents_path.is_file():
        raise FileNotFoundError(f"Latents file not found: {latents_path}")

    with h5py.File(latents_path, "r") as f:
        idx = f["idx"][:]
        index_mmu = f["index_mmu"][:]
        hsc_physics_emb = f["physics_embedding"][:]
        hsc_instrument_emb = f["instrument_embedding"][:]
        if "legacy_physics_embedding" in f and "legacy_instrument_embedding" in f:
            leg_physics_emb = f["legacy_physics_embedding"][:]
            leg_instrument_emb = f["legacy_instrument_embedding"][:]
        else:
            print("Warning: legacy_* embeddings not in latents file; using HSC for Legacy query. Re-run make_latents_all.py.")
            leg_physics_emb = hsc_physics_emb
            leg_instrument_emb = hsc_instrument_emb

    n = len(idx)

    # Combined space: first N = HSC, next N = Legacy (so position i < N -> HSC at idx i; position N+i -> Legacy at idx i)
    combined_physics = np.concatenate([hsc_physics_emb, leg_physics_emb], axis=0)   # (2*n, D)
    combined_instrument = np.concatenate([hsc_instrument_emb, leg_instrument_emb], axis=0)

    def combined_pos_to_dataset_and_source(pos):
        """pos in 0..2*N-1 -> (dataset_idx, 'hsc'|'legacy')."""
        if pos < n:
            return (int(pos), "hsc")
        return (int(pos - n), "legacy")

    k_use = K_NEIGHBORS + 2  # need extra in case self is in top k+1
    nn_phys = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean", algorithm="auto")
    nn_phys.fit(combined_physics)
    nn_inst = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean", algorithm="auto")
    nn_inst.fit(combined_instrument)

    def get_neighbors_exclude_self(positions, self_pos):
        out = [p for p in positions if p != self_pos][:K_NEIGHBORS]
        return [(combined_pos_to_dataset_and_source(p)) for p in out]

    def process_index(query_idx: int, out_path: Optional[Path] = None):
        if query_idx < 0 or query_idx >= n:
            raise ValueError(f"Index must be in [0, {n-1}], got {query_idx}")

        # HSC query: self in combined space at position query_idx (physics and instrument)
        # Legacy query: self in combined space at position n + query_idx

        # HSC query on combined instrument space (query point at index query_idx)
        _, ind_hsc_inst = nn_inst.kneighbors(combined_instrument[query_idx : query_idx + 1], n_neighbors=k_use)
        hsc_inst_neighbors = get_neighbors_exclude_self(ind_hsc_inst[0].tolist(), query_idx)
        # HSC query on combined physics space
        _, ind_hsc_phys = nn_phys.kneighbors(combined_physics[query_idx : query_idx + 1], n_neighbors=k_use)
        hsc_phys_neighbors = get_neighbors_exclude_self(ind_hsc_phys[0].tolist(), query_idx)
        # Legacy query on combined instrument space (query point at index n + query_idx)
        _, ind_leg_inst = nn_inst.kneighbors(combined_instrument[n + query_idx : n + query_idx + 1], n_neighbors=k_use)
        leg_inst_neighbors = get_neighbors_exclude_self(ind_leg_inst[0].tolist(), n + query_idx)
        # Legacy query on combined physics space
        _, ind_leg_phys = nn_phys.kneighbors(combined_physics[n + query_idx : n + query_idx + 1], n_neighbors=k_use)
        leg_phys_neighbors = get_neighbors_exclude_self(ind_leg_phys[0].tolist(), n + query_idx)

        # Load query images
        query_hsc, query_legacy = load_images_for_indices([query_idx], args.neighbors_h5)
        query_hsc, query_legacy = query_hsc[0], query_legacy[0]

        # Load neighbor images (and get source labels from the neighbor lists)
        hsc_inst_images, hsc_inst_sources, hsc_inst_indices = load_images_for_neighbors(hsc_inst_neighbors, args.neighbors_h5)
        hsc_phys_images, hsc_phys_sources, hsc_phys_indices = load_images_for_neighbors(hsc_phys_neighbors, args.neighbors_h5)
        leg_inst_images, leg_inst_sources, leg_inst_indices = load_images_for_neighbors(leg_inst_neighbors, args.neighbors_h5)
        leg_phys_images, leg_phys_sources, leg_phys_indices = load_images_for_neighbors(leg_phys_neighbors, args.neighbors_h5)

        if out_path is None:
            out_path = _here / f"query_{query_idx}.png"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Count "interesting" neighbors (yellow + red) = cross-survey neighbors.
        num_hsc_inst_cross = sum(1 for s in hsc_inst_sources if s == "legacy")
        num_hsc_phys_cross = sum(1 for s in hsc_phys_sources if s == "legacy")
        num_leg_inst_cross = sum(1 for s in leg_inst_sources if s == "hsc")
        num_leg_phys_cross = sum(1 for s in leg_phys_sources if s == "hsc")
        total_highlighted = (
            num_hsc_inst_cross + num_hsc_phys_cross + num_leg_inst_cross + num_leg_phys_cross
        )

        plot_query_and_neighbors(
            query_hsc,
            query_legacy,
            hsc_inst_images,
            hsc_inst_sources,
            hsc_inst_indices,
            hsc_phys_images,
            hsc_phys_sources,
            hsc_phys_indices,
            leg_inst_images,
            leg_inst_sources,
            leg_inst_indices,
            leg_phys_images,
            leg_phys_sources,
            leg_phys_indices,
            query_idx,
            out_path,
        )
        # Build a summary record for logging / CSV.
        summary = {
            "query_idx": query_idx,
            "hsc_inst_indices": list(hsc_inst_indices),
            "hsc_inst_sources": list(hsc_inst_sources),
            "hsc_phys_indices": list(hsc_phys_indices),
            "hsc_phys_sources": list(hsc_phys_sources),
            "leg_inst_indices": list(leg_inst_indices),
            "leg_inst_sources": list(leg_inst_sources),
            "leg_phys_indices": list(leg_phys_indices),
            "leg_phys_sources": list(leg_phys_sources),
            "num_hsc_inst_cross": num_hsc_inst_cross,
            "num_hsc_phys_cross": num_hsc_phys_cross,
            "num_leg_inst_cross": num_leg_inst_cross,
            "num_leg_phys_cross": num_leg_phys_cross,
            "total_cross": total_highlighted,
        }
        return summary

    def append_summary_to_csv(csv_path: Path, summary: dict):
        """Append a single query's neighbor info to a CSV file."""
        fieldnames = [
            "query_idx",
            "hsc_inst_indices",
            "hsc_inst_sources",
            "hsc_phys_indices",
            "hsc_phys_sources",
            "leg_inst_indices",
            "leg_inst_sources",
            "leg_phys_indices",
            "leg_phys_sources",
            "num_hsc_inst_cross",
            "num_hsc_phys_cross",
            "num_leg_inst_cross",
            "num_leg_phys_cross",
            "total_cross",
        ]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {
                "query_idx": summary["query_idx"],
                "hsc_inst_indices": _list_to_str(summary["hsc_inst_indices"]),
                "hsc_inst_sources": _list_to_str(summary["hsc_inst_sources"]),
                "hsc_phys_indices": _list_to_str(summary["hsc_phys_indices"]),
                "hsc_phys_sources": _list_to_str(summary["hsc_phys_sources"]),
                "leg_inst_indices": _list_to_str(summary["leg_inst_indices"]),
                "leg_inst_sources": _list_to_str(summary["leg_inst_sources"]),
                "leg_phys_indices": _list_to_str(summary["leg_phys_indices"]),
                "leg_phys_sources": _list_to_str(summary["leg_phys_sources"]),
                "num_hsc_inst_cross": summary["num_hsc_inst_cross"],
                "num_hsc_phys_cross": summary["num_hsc_phys_cross"],
                "num_leg_inst_cross": summary["num_leg_inst_cross"],
                "num_leg_phys_cross": summary["num_leg_phys_cross"],
                "total_cross": summary["total_cross"],
            }
            writer.writerow(row)

    # Decide between single-index and batch modes.
    if args.batch_start is not None or args.batch_end is not None:
        start = args.batch_start if args.batch_start is not None else 0
        end = args.batch_end if args.batch_end is not None else n - 1
        if start < 0 or end >= n or start > end:
            raise ValueError(f"Batch range must satisfy 0 <= start <= end < {n}; got start={start}, end={end}")

        # In batch mode, --out is treated as an output directory (default: script dir).
        if args.out is not None:
            base_out_dir = Path(args.out)
        else:
            base_out_dir = _here
        base_out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = base_out_dir / "neighbors_summary.csv"

        best_so_far = {}
        processed = 0
        for query_idx in range(start, end + 1):
            out_path = base_out_dir / f"query_{query_idx}.png"
            summary = process_index(query_idx, out_path)
            highlighted = summary["total_cross"]
            best_so_far[query_idx] = highlighted
            append_summary_to_csv(csv_path, summary)
            processed += 1
            if processed % 20 == 0:
                # Find the top-10 indices with the maximum number of highlighted (yellow+red) neighbors so far.
                top_items = sorted(best_so_far.items(), key=lambda kv: kv[1], reverse=True)[:10]
                print(f"After {processed} queries (up to index {query_idx}), top {len(top_items)} most interesting so far:")
                for rank, (idx_best, count_best) in enumerate(top_items, start=1):
                    print(
                        f"  {rank:2d}. index {idx_best} with {count_best} cross-survey (yellow+red) neighbors"
                    )
    else:
        # Single index mode (original behavior)
        process_index(args.index, args.out)


if __name__ == "__main__":
    main()
