"""
Re-encoding script for models trained with neighbors (double_train_fm_neighbors.py).

Loads a batch of BATCH_SIZE_LOAD (e.g. 16), uses first N_VIS (e.g. 8) as the "main" examples:
- Sample 1: galaxies [:8] with their own sameins [:8] (original pairing)
- Sample 2: same galaxies [:8] with sameins [8:16] (no overlap, no repeated neighbor examples)
Then re-encodes targets, generated samples, and condition images with encoder_1 and encoder_2,
and builds a UMAP visualization (NUM_UMAP_BACKGROUND background + special points) with
NUM_EXAMPLES_PLOT examples highlighted. Offsets are dynamic.
Skips plotting neighbor points where mask==0 to avoid garbage from zero-padding.

Uses NeighborsPrecomputedDataset / simple_collate; batch format is
(targets, samegals, sameins, masks, metadata) with sameins (B, k, C, H, W) and masks (B, k).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from neighbors import NeighborsPrecomputedDataset, simple_collate
from double_train_fm_neighbors import ConditionalFlowMatchingModule

import numpy as np
import umap
import matplotlib.pyplot as plt


# --- Config (override as needed) ---
BATCH_SIZE_LOAD = 16  # Load this many; use first N_VIS and sameins [8:16] for sample 2 (no overlap)
N_VIS = 8  # Number of examples to visualize (first half of batch)
NUM_EXAMPLES_PLOT = 8  # Must be <= N_VIS
NUM_UMAP_BACKGROUND = 8192  # Background points for UMAP shape
N_NEIGHBORS_TO_USE = 1

checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=61000.ckpt'
# PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5'
PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5'
mode_suffix = 'neighbors'

# Determine device
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
    if device.type == 'cpu':
        print("No working GPU found, using CPU")

# Load model from checkpoint (neighbors-trained)
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

# Neighbors dataset: (targets, samegals, sameins, masks, metadata)
val_dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
data_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_LOAD,
    shuffle=False,
    num_workers=0,
    collate_fn=simple_collate,
)
batch = next(iter(data_loader))

x_1, cond_image_samegal, cond_image_sameins, masks, metadata = batch
# Shapes: (BATCH_SIZE_LOAD, ...). We use first N_VIS and sameins [8:16] for sample 2 (no overlap).

x_1 = x_1.to(device)
cond_image_samegal = cond_image_samegal.to(device)
cond_image_sameins = cond_image_sameins.to(device)
masks = masks.to(device)

B, k_max, _, _, _ = cond_image_sameins.shape
print('About to generate samples')
print('cond_image_sameins shape', cond_image_sameins.shape, 'masks shape', masks.shape)

# Slice to the N_VIS examples we visualize (first half of batch)
x_1_vis = x_1[:N_VIS]
cond_image_samegal_vis = cond_image_samegal[:N_VIS]

# Sample 1: first N_VIS galaxies with their own sameins [:N_VIS]
print(f'Generating {N_VIS} samples (Normal, sameins [:N_VIS])...')
samples_1 = model.sample(
    cond_image_samegal_vis,
    cond_image_sameins[:N_VIS, :N_NEIGHBORS_TO_USE, :, :, :],
    masks=masks[:N_VIS, :N_NEIGHBORS_TO_USE],
)
# Sample 2: same N_VIS galaxies with sameins [N_VIS:BATCH_SIZE_LOAD] (no repeated neighbor examples)
print(f'Generating {N_VIS} samples (Swapped, sameins [{N_VIS}:{BATCH_SIZE_LOAD}])...')
samples_2 = model.sample(
    cond_image_samegal_vis,
    cond_image_sameins[N_VIS:BATCH_SIZE_LOAD, :N_NEIGHBORS_TO_USE, :, :, :],
    masks=masks[N_VIS:BATCH_SIZE_LOAD, :N_NEIGHBORS_TO_USE],
)

print('samples_1 shape', samples_1.shape, 'samples_2 shape', samples_2.shape)

# Encode special points (N_VIS examples only)
embedding_target_enc1 = model.encoder_1(x_1_vis).cpu()
embedding_generated_samples_1_enc1 = model.encoder_1(samples_1).cpu()
embedding_generated_samples_2_enc1 = model.encoder_1(samples_2).cpu()
embedding_samegal = model.encoder_1(cond_image_samegal_vis).cpu()

embedding_target_enc2 = model.encoder_2(x_1_vis).cpu()
embedding_generated_samples_1_enc2 = model.encoder_2(samples_1).cpu()
embedding_generated_samples_2_enc2 = model.encoder_2(samples_2).cpu()

# Encode only the neighbors used for sampling: sameins_1 = [:N_VIS], sameins_2 = [N_VIS:BATCH_SIZE_LOAD]
cond_image_sameins_1_flat = cond_image_sameins[:N_VIS, :N_NEIGHBORS_TO_USE, :, :, :].flatten(0, 1)
cond_image_sameins_2_flat = cond_image_sameins[N_VIS:BATCH_SIZE_LOAD, :N_NEIGHBORS_TO_USE, :, :, :].flatten(0, 1)
embedding_sameins_1_all = model.encoder_2(cond_image_sameins_1_flat).cpu()
embedding_sameins_2_all = model.encoder_2(cond_image_sameins_2_flat).cpu()

# --- Collect background embeddings (targets only) for UMAP ---
bg_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    collate_fn=simple_collate,
)
all_embeddings_enc1_bg_list = []
all_embeddings_enc2_bg_list = []
count = 0
print('Collecting background embeddings...')
with torch.no_grad():
    for bg_batch in bg_loader:
        if count >= NUM_UMAP_BACKGROUND:
            break
        bx, _, _, _, _ = bg_batch
        bx = bx.to(device)
        all_embeddings_enc1_bg_list.append(model.encoder_1(bx).cpu())
        all_embeddings_enc2_bg_list.append(model.encoder_2(bx).cpu())
        count += bx.shape[0]

bg_enc1 = torch.cat(all_embeddings_enc1_bg_list, dim=0)[:NUM_UMAP_BACKGROUND]
bg_enc2 = torch.cat(all_embeddings_enc2_bg_list, dim=0)[:NUM_UMAP_BACKGROUND]

# Order: [Target, Gen1, Gen2, SameGal/SameIns, Background]
special_emb_enc1 = torch.cat([
    embedding_target_enc1,
    embedding_generated_samples_1_enc1,
    embedding_generated_samples_2_enc1,
    embedding_samegal,
], dim=0)

special_emb_enc2 = torch.cat([
    embedding_target_enc2,
    embedding_generated_samples_1_enc2,
    embedding_generated_samples_2_enc2,
    embedding_sameins_1_all,
    embedding_sameins_2_all,
], dim=0)

all_embeddings_enc1 = torch.cat([special_emb_enc1, bg_enc1], dim=0)
all_embeddings_enc2 = torch.cat([special_emb_enc2, bg_enc2], dim=0)
all_embeddings_enc1_flat = all_embeddings_enc1.flatten(start_dim=1).numpy()
all_embeddings_enc2_flat = all_embeddings_enc2.flatten(start_dim=1).numpy()

print('\nComputing UMAP for Encoder 1 (Physics)...')
umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 2, 'metric': 'euclidean', 'random_state': 42}
reducer_1 = umap.UMAP(**umap_params)
umap_enc1 = reducer_1.fit_transform(all_embeddings_enc1_flat)

print('Computing UMAP for Encoder 2 (Instrument)...')
reducer_2 = umap.UMAP(**umap_params)
umap_enc2 = reducer_2.fit_transform(all_embeddings_enc2_flat)

# Dynamic offsets (no hardcoded 32/64/96)
# enc1: 0:N_VIS target, N_VIS:2*N_VIS gen1, 2*N_VIS:3*N_VIS gen2, 3*N_VIS:4*N_VIS samegal
# enc2: 0:N_VIS target, N_VIS:2*N_VIS gen1, 2*N_VIS:3*N_VIS gen2,
#       3*N_VIS : 3*N_VIS+N_VIS*N_NEIGHBORS_TO_USE sameins_1, then sameins_2 (only the neighbor(s) used)
idx_gen1 = N_VIS
idx_gen2 = N_VIS * 2
idx_same = N_VIS * 3
idx_sameins1_start = N_VIS * 3
idx_sameins2_start = N_VIS * 3 + (N_VIS * N_NEIGHBORS_TO_USE)

masks_np = masks[:N_VIS].cpu().numpy()  # For skipping zero-padded neighbors in enc2 plot (first N_VIS only)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
marker_sizes = [150] * 8

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Encoder 1 (Physics)
ax1 = axes[0]
ax1.scatter(umap_enc1[:, 0], umap_enc1[:, 1], s=5, c='lightgray', alpha=0.3, label='All examples')
for i in range(NUM_EXAMPLES_PLOT):
    c = colors[i]
    ax1.scatter(umap_enc1[i, 0], umap_enc1[i, 1], s=marker_sizes[i], marker='s', c=c, edgecolors='black', linewidths=2,
                label=f'Target {i+1}' if i == 0 else None, alpha=0.8)
    ax1.scatter(umap_enc1[idx_gen1 + i, 0], umap_enc1[idx_gen1 + i, 1], s=marker_sizes[i], marker='^', c=c, edgecolors='black', linewidths=2,
                label=f'Sample 1 {i+1}' if i == 0 else None, alpha=0.8)
    ax1.scatter(umap_enc1[idx_gen2 + i, 0], umap_enc1[idx_gen2 + i, 1], s=marker_sizes[i], marker='v', c=c, edgecolors='black', linewidths=2,
                label=f'Sample 2 {i+1}' if i == 0 else None, alpha=0.8)
    ax1.scatter(umap_enc1[idx_same + i, 0], umap_enc1[idx_same + i, 1], s=marker_sizes[i], marker='x', c=c, edgecolors='black', linewidths=2,
                label=f'SameGal {i+1}' if i == 0 else None, alpha=0.8)
ax1.set_title('Encoder 1 (Physics) - UMAP Visualization', fontsize=14)
ax1.set_xlabel('UMAP Component 1', fontsize=12)
ax1.set_ylabel('UMAP Component 2', fontsize=12)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# Encoder 2 (Instrument): sameins = only the N_NEIGHBORS_TO_USE neighbor(s) used for sampling; skip where mask==0
ax2 = axes[1]
ax2.scatter(umap_enc2[:, 0], umap_enc2[:, 1], s=5, c='lightgray', alpha=0.3, label='All examples')
for i in range(NUM_EXAMPLES_PLOT):
    c = colors[i]
    ax2.scatter(umap_enc2[i, 0], umap_enc2[i, 1], s=marker_sizes[i], marker='s', c=c, edgecolors='black', linewidths=2,
                label=f'Target {i+1}' if i == 0 else None, alpha=0.8)
    ax2.scatter(umap_enc2[idx_gen1 + i, 0], umap_enc2[idx_gen1 + i, 1], s=marker_sizes[i], marker='^', c=c, edgecolors='black', linewidths=2,
                label=f'Sample 1 {i+1}' if i == 0 else None, alpha=0.8)
    ax2.scatter(umap_enc2[idx_gen2 + i, 0], umap_enc2[idx_gen2 + i, 1], s=marker_sizes[i], marker='v', c=c, edgecolors='black', linewidths=2,
                label=f'Sample 2 {i+1}' if i == 0 else None, alpha=0.8)
    for j in range(N_NEIGHBORS_TO_USE):
        if masks_np[i, j] == 0:
            continue
        idx1 = idx_sameins1_start + (i * N_NEIGHBORS_TO_USE) + j
        idx2 = idx_sameins2_start + (i * N_NEIGHBORS_TO_USE) + j
        ax2.scatter(umap_enc2[idx1, 0], umap_enc2[idx1, 1], s=marker_sizes[i] // 2, marker='x', c=c,
                    edgecolors='black', linewidths=1.5, label='SameIns 1' if i == 0 and j == 0 else None, alpha=0.7)
        ax2.scatter(umap_enc2[idx2, 0], umap_enc2[idx2, 1], s=marker_sizes[i] // 2, marker='o', facecolors='none',
                    edgecolors=c, linewidths=1.5, label='SameIns 2' if i == 0 and j == 0 else None, alpha=0.7)
ax2.set_title('Encoder 2 (Instrument) - UMAP Visualization', fontsize=14)
ax2.set_xlabel('UMAP Component 1', fontsize=12)
ax2.set_ylabel('UMAP Component 2', fontsize=12)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f'Re-encoding UMAP (Neighbors model): {NUM_EXAMPLES_PLOT} examples with targets, two generated samples (different same_ins), and condition images',
    fontsize=16, y=1.02,
)
plt.tight_layout()

figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / f're_encoding_umap_{NUM_EXAMPLES_PLOT}examples_{mode_suffix}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'\nUMAP visualization saved to: {output_path}')
print('Done.')
