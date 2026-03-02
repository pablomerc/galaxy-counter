"""
kNN query visualization for contrastive neighbor latents.

Same behavior as galaxy_model/neighbor_search/search_neighbors.py, but reads
latents produced by contrastive_neighbors/make_latents_all.py.
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

_here = Path(__file__).resolve().parent

from galaxy_counter.neighbors import NeighborsSimpleDataset

NEIGHBORS_HDF5 = os.getenv(
    "GALAXY_COUNTER_NEIGHBORS_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
)
K_NEIGHBORS = 10
N_COLS = 12


def _list_to_str(xs):
    return ";".join(str(x) for x in xs)


def load_images_for_indices(indices, hdf5_path=NEIGHBORS_HDF5):
    dataset = NeighborsSimpleDataset(hdf5_path=hdf5_path)
    hsc_list, leg_list = [], []
    for idx in indices:
        hsc, leg, _ = dataset[idx]
        hsc_list.append(hsc)
        leg_list.append(leg)
    return hsc_list, leg_list


def load_images_for_neighbors(neighbor_list, hdf5_path=NEIGHBORS_HDF5):
    unique_indices = sorted(set(idx for idx, _ in neighbor_list))
    idx_to_rank = {idx: i for i, idx in enumerate(unique_indices)}
    hsc_list, leg_list = load_images_for_indices(unique_indices, hdf5_path)
    images, sources, indices = [], [], []
    for dataset_idx, source in neighbor_list:
        r = idx_to_rank[dataset_idx]
        images.append(hsc_list[r] if source == "hsc" else leg_list[r])
        sources.append(source)
        indices.append(dataset_idx)
    return images, sources, indices


def tensor_to_display(t, channel=0):
    x = t.numpy()
    if x.ndim == 3:
        x = x[channel]
    x = np.clip(x, -3, 3)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


def tensor_to_rgb_row_scaled(t, vmin, vmax):
    x = t.numpy()
    if x.shape[0] < 3:
        gray = tensor_to_display(t)
        return np.stack([gray, gray, gray], axis=-1)
    x = x[:3]
    vmin = np.asarray(vmin).reshape(3, 1, 1)
    vmax = np.asarray(vmax).reshape(3, 1, 1)
    y = (x - vmin) / (vmax - vmin + 1e-8)
    y = np.clip(y, 0.0, 1.0)
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
    n_cols = N_COLS
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.2))
    for ax in axes.flat:
        ax.set_axis_off()

    q = query_hsc.numpy()
    if q.shape[0] < 3:
        use_rgb = False
        vmin = vmax = None
    else:
        use_rgb = True
        q_flat = q[:3].reshape(3, -1)
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

    show_img(axes[0, 2], query_hsc, "Query HSC")
    show_img(axes[0, 3], query_legacy, "Query Legacy")

    def src_label(s):
        return "(HSC)" if s == "hsc" else "(Leg)"

    for j, img in enumerate(hsc_inst_images):
        is_counterpart = hsc_inst_indices[j] == query_idx and hsc_inst_sources[j] == "legacy"
        is_cross = hsc_inst_sources[j] == "legacy"
        color = "red" if is_counterpart else ("gold" if is_cross else "black")
        ax = axes[1, 1 + j]
        show_img(ax, img, f"HSC inst kNN {j+1} {src_label(hsc_inst_sources[j])}", title_color=color)
        ax.set_xlabel(f"idx {hsc_inst_indices[j]}", fontsize=7)

    for j, img in enumerate(hsc_phys_images):
        is_counterpart = hsc_phys_indices[j] == query_idx and hsc_phys_sources[j] == "legacy"
        is_cross = hsc_phys_sources[j] == "legacy"
        color = "red" if is_counterpart else ("gold" if is_cross else "black")
        ax = axes[2, 1 + j]
        show_img(ax, img, f"HSC phys kNN {j+1} {src_label(hsc_phys_sources[j])}", title_color=color)
        ax.set_xlabel(f"idx {hsc_phys_indices[j]}", fontsize=7)

    for j, img in enumerate(leg_inst_images):
        is_counterpart = leg_inst_indices[j] == query_idx and leg_inst_sources[j] == "hsc"
        is_cross = leg_inst_sources[j] == "hsc"
        color = "red" if is_counterpart else ("gold" if is_cross else "black")
        ax = axes[3, 1 + j]
        show_img(ax, img, f"Leg inst kNN {j+1} {src_label(leg_inst_sources[j])}", title_color=color)
        ax.set_xlabel(f"idx {leg_inst_indices[j]}", fontsize=7)

    for j, img in enumerate(leg_phys_images):
        is_counterpart = leg_phys_indices[j] == query_idx and leg_phys_sources[j] == "hsc"
        is_cross = leg_phys_sources[j] == "hsc"
        color = "red" if is_counterpart else ("gold" if is_cross else "black")
        ax = axes[4, 1 + j]
        show_img(ax, img, f"Leg phys kNN {j+1} {src_label(leg_phys_sources[j])}", title_color=color)
        ax.set_xlabel(f"idx {leg_phys_indices[j]}", fontsize=7)

    plt.suptitle("Contrastive neighbors searched over BOTH surveys", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Show query image and top-k neighbors in contrastive latent spaces.")
    p.add_argument("--latents", type=Path, default=None)
    p.add_argument("--index", type=int, default=10)
    p.add_argument("--batch-start", type=int, default=None)
    p.add_argument("--batch-end", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--neighbors-h5", type=str, default=NEIGHBORS_HDF5)
    args = p.parse_args()

    latents_path = Path(args.latents) if args.latents is not None else None
    if latents_path is None:
        candidates = list(_here.glob("contrastive_neighbor_latents_*.h5"))
        if len(candidates) == 1:
            latents_path = candidates[0]
            print(f"Using latents file: {latents_path}")
        elif not candidates:
            raise FileNotFoundError("No contrastive_neighbor_latents_*.h5 found; run make_latents_all.py first or pass --latents")
        else:
            raise FileNotFoundError(f"Multiple latents found: {candidates}; pass --latents explicitly")
    if not latents_path.is_file():
        raise FileNotFoundError(f"Latents file not found: {latents_path}")

    with h5py.File(latents_path, "r") as f:
        idx = f["idx"][:]
        hsc_physics_emb = f["physics_embedding"][:]
        hsc_instrument_emb = f["instrument_embedding"][:]
        leg_physics_emb = f["legacy_physics_embedding"][:]
        leg_instrument_emb = f["legacy_instrument_embedding"][:]

    n = len(idx)
    combined_physics = np.concatenate([hsc_physics_emb, leg_physics_emb], axis=0)
    combined_instrument = np.concatenate([hsc_instrument_emb, leg_instrument_emb], axis=0)

    def combined_pos_to_dataset_and_source(pos):
        return (int(pos), "hsc") if pos < n else (int(pos - n), "legacy")

    k_use = K_NEIGHBORS + 2
    nn_phys = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean", algorithm="auto")
    nn_phys.fit(combined_physics)
    nn_inst = NearestNeighbors(n_neighbors=min(k_use, 2 * n), metric="euclidean", algorithm="auto")
    nn_inst.fit(combined_instrument)

    def get_neighbors_exclude_self(positions, self_pos):
        out = [p for p in positions if p != self_pos][:K_NEIGHBORS]
        return [combined_pos_to_dataset_and_source(p) for p in out]

    def process_index(query_idx: int, out_path: Optional[Path] = None):
        if query_idx < 0 or query_idx >= n:
            raise ValueError(f"Index must be in [0, {n-1}], got {query_idx}")

        _, ind_hsc_inst = nn_inst.kneighbors(combined_instrument[query_idx:query_idx + 1], n_neighbors=k_use)
        hsc_inst_neighbors = get_neighbors_exclude_self(ind_hsc_inst[0].tolist(), query_idx)
        _, ind_hsc_phys = nn_phys.kneighbors(combined_physics[query_idx:query_idx + 1], n_neighbors=k_use)
        hsc_phys_neighbors = get_neighbors_exclude_self(ind_hsc_phys[0].tolist(), query_idx)
        _, ind_leg_inst = nn_inst.kneighbors(combined_instrument[n + query_idx:n + query_idx + 1], n_neighbors=k_use)
        leg_inst_neighbors = get_neighbors_exclude_self(ind_leg_inst[0].tolist(), n + query_idx)
        _, ind_leg_phys = nn_phys.kneighbors(combined_physics[n + query_idx:n + query_idx + 1], n_neighbors=k_use)
        leg_phys_neighbors = get_neighbors_exclude_self(ind_leg_phys[0].tolist(), n + query_idx)

        query_hsc, query_legacy = load_images_for_indices([query_idx], args.neighbors_h5)
        query_hsc, query_legacy = query_hsc[0], query_legacy[0]

        hsc_inst_images, hsc_inst_sources, hsc_inst_indices = load_images_for_neighbors(hsc_inst_neighbors, args.neighbors_h5)
        hsc_phys_images, hsc_phys_sources, hsc_phys_indices = load_images_for_neighbors(hsc_phys_neighbors, args.neighbors_h5)
        leg_inst_images, leg_inst_sources, leg_inst_indices = load_images_for_neighbors(leg_inst_neighbors, args.neighbors_h5)
        leg_phys_images, leg_phys_sources, leg_phys_indices = load_images_for_neighbors(leg_phys_neighbors, args.neighbors_h5)

        if out_path is None:
            out_path = _here / f"query_{query_idx}.png"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        num_hsc_inst_cross = sum(1 for s in hsc_inst_sources if s == "legacy")
        num_hsc_phys_cross = sum(1 for s in hsc_phys_sources if s == "legacy")
        num_leg_inst_cross = sum(1 for s in leg_inst_sources if s == "hsc")
        num_leg_phys_cross = sum(1 for s in leg_phys_sources if s == "hsc")

        plot_query_and_neighbors(
            query_hsc, query_legacy,
            hsc_inst_images, hsc_inst_sources, hsc_inst_indices,
            hsc_phys_images, hsc_phys_sources, hsc_phys_indices,
            leg_inst_images, leg_inst_sources, leg_inst_indices,
            leg_phys_images, leg_phys_sources, leg_phys_indices,
            query_idx, out_path,
        )

        return {
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
            "total_cross": (num_hsc_inst_cross + num_hsc_phys_cross + num_leg_inst_cross + num_leg_phys_cross),
        }

    def append_summary_to_csv(csv_path: Path, summary: dict):
        fieldnames = [
            "query_idx",
            "hsc_inst_indices", "hsc_inst_sources",
            "hsc_phys_indices", "hsc_phys_sources",
            "leg_inst_indices", "leg_inst_sources",
            "leg_phys_indices", "leg_phys_sources",
            "num_hsc_inst_cross", "num_hsc_phys_cross",
            "num_leg_inst_cross", "num_leg_phys_cross",
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

    if args.batch_start is not None or args.batch_end is not None:
        start = args.batch_start if args.batch_start is not None else 0
        end = args.batch_end if args.batch_end is not None else n - 1
        if start < 0 or end >= n or start > end:
            raise ValueError(f"Batch range must satisfy 0 <= start <= end < {n}; got {start}..{end}")

        base_out_dir = Path(args.out) if args.out is not None else (_here / "query_results")
        base_out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = base_out_dir / "neighbors_summary.csv"
        best_so_far = {}

        for processed, query_idx in enumerate(range(start, end + 1), start=1):
            out_path = base_out_dir / f"query_{query_idx}.png"
            summary = process_index(query_idx, out_path)
            best_so_far[query_idx] = summary["total_cross"]
            append_summary_to_csv(csv_path, summary)
            if processed % 20 == 0:
                top_items = sorted(best_so_far.items(), key=lambda kv: kv[1], reverse=True)[:10]
                print(f"After {processed} queries (up to {query_idx}), top interesting:")
                for rank, (idx_best, count_best) in enumerate(top_items, start=1):
                    print(f"  {rank:2d}. index {idx_best} with {count_best} cross-survey neighbors")
    else:
        process_index(args.index, args.out)


if __name__ == "__main__":
    main()
