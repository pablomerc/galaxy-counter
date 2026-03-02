"""
Load a pretrained neighbors model and run latent-space visualizations (UMAP, PCA, t-SNE)
using the same dataloader as training: NeighborsPrecomputedDataset with (target, samegal) pairs.

V2: First 2048 examples only; pairs are target (anchor) vs samegal (same galaxy, other instrument).
Saves to neighbors_visualization/latent_space/ with _v2 suffix in filenames.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from double_train_fm_neighbors import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader, Subset
from neighbors import NeighborsPrecomputedDataset, simple_collate
import time

import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Same as neighbours_train.py
PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5'
NUM_EXAMPLES = 2048  # First N examples for visualization
BATCH_SIZE = 256     # Load in batches to avoid one huge batch

# Neighbors model checkpoints (same as load_pretrained_model_neighbors.py)
CHECKPOINT_OPTIONS = [
    ("64 no geo", "tess-ml/galaxy-flow-matching-neighbours/n8szckjq",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/n8szckjq/checkpoints/latest-step=step=56000.ckpt",
     64, False, 56000),
    ("64 geo", "tess-ml/galaxy-flow-matching-neighbours/lr451mnx",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=52000.ckpt",
     64, True, 50000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/g2g9kvr4",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt",
     16, False, 75000),
]

idx = 0
_label, _wandb_id, checkpoint_path, dim, _geom, step = CHECKPOINT_OPTIONS[idx]
mode_tag = "geom" if _geom else ""
epoch = step

print(f"Checkpoint: {_label} (wandb {_wandb_id}), path={checkpoint_path}, dim={dim}, geom={_geom}, step={step}")
print(f"Data: NeighborsPrecomputedDataset, first {NUM_EXAMPLES} examples, pairs = (target, samegal)")

NEIGHBORS_VIZ_DIR = Path(__file__).parent / 'neighbors_visualization'
LATENT_SPACE_DIR = NEIGHBORS_VIZ_DIR / 'latent_space'
VIZ_SUFFIX = '_v2'  # All saved plots get this suffix

GENERATE_UMAP = True
GENERATE_PCA = True
GENERATE_TSNE = True
SHOW_PAIRS = True

device = torch.device('cpu')
if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        try:
            test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
            break
        except RuntimeError:
            continue

model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

# --- Same dataloader as training: NeighborsPrecomputedDataset + simple_collate, first NUM_EXAMPLES ---
print(f"\nLoading dataset (first {NUM_EXAMPLES} examples)...")
t0 = time.perf_counter()
full_dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
subset = Subset(full_dataset, range(min(NUM_EXAMPLES, len(full_dataset))))
actual_num = len(subset)
loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=simple_collate,
)
# Collect targets and samegals from batches
targets_list, samegals_list = [], []
for batch in loader:
    t, s, *_ = batch
    targets_list.append(t)
    samegals_list.append(s)
    if sum(x.shape[0] for x in targets_list) >= NUM_EXAMPLES:
        break
targets = torch.cat(targets_list, dim=0)[:NUM_EXAMPLES].to(device)
samegals = torch.cat(samegals_list, dim=0)[:NUM_EXAMPLES].to(device)
print(f"  Loaded {targets.shape[0]} pairs in {time.perf_counter() - t0:.2f} s")
print(f"  targets shape: {targets.shape}, samegals shape: {samegals.shape}")

# Encode with both encoders (target and samegal are both single images per sample)
with torch.no_grad():
    target_embeddings_1 = model.encoder_1(targets)
    samegal_embeddings_1 = model.encoder_1(samegals)
    target_embeddings_2 = model.encoder_2(targets)
    samegal_embeddings_2 = model.encoder_2(samegals)

all_embeddings_1 = torch.concat([target_embeddings_1, samegal_embeddings_1], dim=0).flatten(start_dim=1)
all_embeddings_2 = torch.concat([target_embeddings_2, samegal_embeddings_2], dim=0).flatten(start_dim=1)
num_pairs = target_embeddings_1.shape[0]
print(f"Encoder 1 flattened: {all_embeddings_1.shape}, Encoder 2 flattened: {all_embeddings_2.shape}, num_pairs={num_pairs}")

# Distance analysis (paired = target vs samegal for same index)
target_flat_1 = all_embeddings_1[:num_pairs].cpu().numpy()
samegal_flat_1 = all_embeddings_1[num_pairs:].cpu().numpy()
target_flat_2 = all_embeddings_2[:num_pairs].cpu().numpy()
samegal_flat_2 = all_embeddings_2[num_pairs:].cpu().numpy()
paired_d1 = np.linalg.norm(target_flat_1 - samegal_flat_1, axis=1)
paired_d2 = np.linalg.norm(target_flat_2 - samegal_flat_2, axis=1)
np.random.seed(42)
shuf = np.random.permutation(num_pairs)
unpaired_d1 = np.linalg.norm(target_flat_1 - samegal_flat_1[shuf], axis=1)
unpaired_d2 = np.linalg.norm(target_flat_2 - samegal_flat_2[shuf], axis=1)
print("\nDistance analysis (Target vs Same galaxy):")
print(f"  Encoder 1: avg paired {np.mean(paired_d1):.4f}, avg unpaired {np.mean(unpaired_d1):.4f}, ratio {np.mean(paired_d1)/np.mean(unpaired_d1):.4f}")
print(f"  Encoder 2: avg paired {np.mean(paired_d2):.4f}, avg unpaired {np.mean(unpaired_d2):.4f}, ratio {np.mean(paired_d2)/np.mean(unpaired_d2):.4f}")

LATENT_SPACE_DIR.mkdir(parents=True, exist_ok=True)
umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2, 'metric': 'euclidean', 'random_state': 42}
pca_params = {'n_components': 2, 'random_state': 42}
tsne_params = {'n_components': 2, 'perplexity': 30, 'random_state': 42, 'max_iter': 1000}

def _plot_pair_highlights(ax, emb_a, emb_b, selected_indices, pair_colors, pair_markers):
    if selected_indices is None:
        return
    for i, idx in enumerate(selected_indices):
        c, m = pair_colors[i % 5], pair_markers[i % 4]
        ax.scatter(emb_a[idx, 0], emb_a[idx, 1], marker=m, s=200, c=[c], linewidths=3, zorder=5, edgecolors='black')
        ax.scatter(emb_b[idx, 0], emb_b[idx, 1], marker=m, s=200, c=[c], linewidths=3, zorder=5, edgecolors='black')

selected_indices = None
pair_colors = plt.cm.tab10(np.linspace(0, 1, 5))
pair_markers = ['x', 's', 'o', '^']
if SHOW_PAIRS:
    np.random.seed(42)
    selected_indices = np.random.choice(num_pairs, size=min(20, num_pairs), replace=False)

# ----- UMAP -----
if GENERATE_UMAP:
    print("\nUMAP (Encoder 1)...")
    reducer_1 = umap.UMAP(**umap_params)
    emb1 = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())
    target_umap_1, samegal_umap_1 = emb1[:num_pairs], emb1[num_pairs:]
    print("UMAP (Encoder 2)...")
    reducer_2 = umap.UMAP(**umap_params)
    emb2 = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())
    target_umap_2, samegal_umap_2 = emb2[:num_pairs], emb2[num_pairs:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(target_umap_1[:, 0], target_umap_1[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax1.scatter(samegal_umap_1[:, 0], samegal_umap_1[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax1, target_umap_1, samegal_umap_1, selected_indices, pair_colors, pair_markers)
    ax1.set_title(f'Encoder 1 (Same Galaxy) – Physics (UMAP) step {epoch}\nData: training loader, target vs samegal')
    ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2'); ax1.legend(); ax1.grid(True)

    ax2.scatter(target_umap_2[:, 0], target_umap_2[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax2.scatter(samegal_umap_2[:, 0], samegal_umap_2[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax2, target_umap_2, samegal_umap_2, selected_indices, pair_colors, pair_markers)
    ax2.set_title(f'Encoder 2 (Same Instrument) – Instrument (UMAP) step {epoch}\nData: training loader, target vs samegal')
    ax2.set_xlabel('UMAP 1'); ax2.set_ylabel('UMAP 2'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    path = LATENT_SPACE_DIR / f'umap_both_encoders_zdim{dim}{mode_tag}{VIZ_SUFFIX}.png'
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")

# ----- PCA -----
if GENERATE_PCA:
    print("\nPCA (Encoder 1 & 2)...")
    pca_1 = PCA(**pca_params)
    emb1 = pca_1.fit_transform(all_embeddings_1.cpu().numpy())
    target_pca_1, samegal_pca_1 = emb1[:num_pairs], emb1[num_pairs:]
    pca_2 = PCA(**pca_params)
    emb2 = pca_2.fit_transform(all_embeddings_2.cpu().numpy())
    target_pca_2, samegal_pca_2 = emb2[:num_pairs], emb2[num_pairs:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(target_pca_1[:, 0], target_pca_1[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax1.scatter(samegal_pca_1[:, 0], samegal_pca_1[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax1, target_pca_1, samegal_pca_1, selected_indices, pair_colors, pair_markers)
    ax1.set_title(f'Encoder 1 – Physics (PCA) step {epoch}\nExpl. var: {pca_1.explained_variance_ratio_.sum():.2%}')
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2'); ax1.legend(); ax1.grid(True)
    ax2.scatter(target_pca_2[:, 0], target_pca_2[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax2.scatter(samegal_pca_2[:, 0], samegal_pca_2[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax2, target_pca_2, samegal_pca_2, selected_indices, pair_colors, pair_markers)
    ax2.set_title(f'Encoder 2 – Instrument (PCA) step {epoch}\nExpl. var: {pca_2.explained_variance_ratio_.sum():.2%}')
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    path = LATENT_SPACE_DIR / f'pca_both_encoders_zdim{dim}{mode_tag}{VIZ_SUFFIX}.png'
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")

# ----- t-SNE -----
if GENERATE_TSNE:
    print("\nt-SNE (Encoder 1 & 2)...")
    tsne_1 = TSNE(**tsne_params)
    emb1 = tsne_1.fit_transform(all_embeddings_1.cpu().numpy())
    target_tsne_1, samegal_tsne_1 = emb1[:num_pairs], emb1[num_pairs:]
    tsne_2 = TSNE(**tsne_params)
    emb2 = tsne_2.fit_transform(all_embeddings_2.cpu().numpy())
    target_tsne_2, samegal_tsne_2 = emb2[:num_pairs], emb2[num_pairs:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.scatter(target_tsne_1[:, 0], target_tsne_1[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax1.scatter(samegal_tsne_1[:, 0], samegal_tsne_1[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax1, target_tsne_1, samegal_tsne_1, selected_indices, pair_colors, pair_markers)
    ax1.set_title(f'Encoder 1 – Physics (t-SNE) step {epoch}'); ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2'); ax1.legend(); ax1.grid(True)
    ax2.scatter(target_tsne_2[:, 0], target_tsne_2[:, 1], s=5, label='Target', alpha=0.6, c='blue')
    ax2.scatter(samegal_tsne_2[:, 0], samegal_tsne_2[:, 1], s=5, label='Same galaxy', alpha=0.6, c='orange')
    _plot_pair_highlights(ax2, target_tsne_2, samegal_tsne_2, selected_indices, pair_colors, pair_markers)
    ax2.set_title(f'Encoder 2 – Instrument (t-SNE) step {epoch}'); ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    path = LATENT_SPACE_DIR / f'tsne_both_encoders_zdim{dim}{mode_tag}{VIZ_SUFFIX}.png'
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")

print("\nDone. All figures saved under neighbors_visualization/latent_space/ with _v2 suffix.")
