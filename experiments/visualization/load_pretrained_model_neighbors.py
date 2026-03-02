"""
Load a pretrained neighbors model and run inference on it.
Uses ConditionalFlowMatchingModule from double_train_fm_neighbors (trained with
neighbours_train.py / double_train_fm_neighbors.py). Data from HSCLegacyDatasetZoom.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from double_train_fm_neighbors import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader
from data import HSCLegacyDatasetZoom
import time

import json
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Neighbors model checkpoints (trained with neighbours_train.py / double_train_fm_neighbors.py)
# Format: (label, wandb_run_id, local_checkpoint_path, z_dim, geom, step_for_plots)
CHECKPOINT_OPTIONS = [
    ("64 no geo", "tess-ml/galaxy-flow-matching-neighbours/n8szckjq",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/n8szckjq/checkpoints/latest-step=step=56000.ckpt",
     64, False, 56000),
    ("64 geo", "tess-ml/galaxy-flow-matching-neighbours/lr451mnx",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=53000.ckpt",
     64, True, 53000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/g2g9kvr4",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt",
    # '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/best-epoch=49-step=19000.ckpt',
     16, False, 75000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/x1xf4cym",
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/x1xf4cym/checkpoints/latest-step=step=53000.ckpt',
     16, False, 53000),
]

idx = 2  # 0=64 no geo, 1=64 geo, 2=16 no geo
_label, _wandb_id, checkpoint_path, dim, _geom, step = CHECKPOINT_OPTIONS[idx]
mode_tag = "geom" if _geom else ""
epoch = step  # used in plot titles
zoom_val = True  # always use HSCLegacyDatasetZoom
avg_latent_space = False

print(f"Checkpoint: {_label} (wandb {_wandb_id}), path={checkpoint_path}, dim={dim}, geom={_geom}, step={step}")

# Output directories (relative to this script)
NEIGHBORS_VIZ_DIR = Path(__file__).parent / 'neighbors_visualization'
LATENT_SPACE_DIR = NEIGHBORS_VIZ_DIR / 'latent_space'
GEN_STUDY_DIR = NEIGHBORS_VIZ_DIR / 'gen_study'

# HSC/Legacy colors (match downstream_evaluation/final/aion_vs_ours_all.py)
COLOR_HSC = '#e8c4a0'
COLOR_LEGACY = '#8eb8e8'

# Control flags
GENERATE_UMAP = True  # Set to False to skip UMAP generation and plotting
GENERATE_PCA = True    # Set to False to skip PCA generation and plotting
GENERATE_TSNE = True   # Set to False to skip t-SNE generation and plotting
SHOW_PAIRS = True     # Set to False to skip marking pairs on the plots
GENERATE_SAMPLES = False  # Disable generation study

# Determine device: try to find a working GPU, fallback to CPU
device = torch.device('cpu')  # Default to CPU for safe loading
if torch.cuda.is_available():
    # Try to find a working GPU by testing each one
    for gpu_id in range(torch.cuda.device_count()):
        try:
            # Test if GPU is accessible
            test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
            break
        except RuntimeError:
            print(f"GPU {gpu_id} is not available, trying next...")
            continue
    if device.type == 'cpu':
        print("No working GPU found, using CPU")

# Load checkpoint to CPU first to avoid device conflicts, then move to target device
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')

# Set the model to evaluation mode and disable gradient calculation for inference
model.eval()
torch.set_grad_enabled(False)

# Move model to the determined device
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())




# Time dataset initialization (loading from HDF5 into memory)
# Always use HSCLegacyDatasetZoom (same normalization/zoom as neighbors training)
dataset_start = time.perf_counter()
dataset = HSCLegacyDatasetZoom(
    hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
    idx_list=list(range(95_000, 97_048)),
)
dataset_time = time.perf_counter() - dataset_start

# Time DataLoader creation (very fast, usually negligible)
loader_start = time.perf_counter()
train_loader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=4)
loader_time = time.perf_counter() - loader_start

# Time getting first batch (includes worker startup if num_workers > 0)
batch_start = time.perf_counter()
batch = next(iter(train_loader))
batch_time = time.perf_counter() - batch_start

# Get actual batch size (batch is a tuple: (hsc_images, legacy_images))
# Each element has shape (batch_size, channels, height, width)
actual_batch_size = batch[0].shape[0]  # or batch[1].shape[0]

total_time = dataset_time + loader_time + batch_time

print(f"Timing breakdown for {actual_batch_size} examples:")
print(f"  Dataset initialization (HDF5 → memory): {dataset_time:.4f} s")
print(f"  DataLoader creation:                    {loader_time:.4f} s")
print(f"  First batch retrieval:                  {batch_time:.4f} s")
print(f"  Total time:                             {total_time:.4f} s")


# Move images to device (they come from DataLoader on CPU by default)
hsc_images = batch[0].to(device)
legacy_images = batch[1].to(device)  # Fixed typo: was "legacy_iamges"

# Encode images with both encoders
with torch.no_grad():  # No gradients needed for inference
    hsc_embeddings_1 = model.encoder_1(hsc_images)
    legacy_embeddings_1 = model.encoder_1(legacy_images)
    hsc_embeddings_2 = model.encoder_2(hsc_images)
    legacy_embeddings_2 = model.encoder_2(legacy_images)

print(f"\nEncoding results:")
print(f"  HSC images shape:        {hsc_images.shape}")
print(f"  HSC embeddings 1 shape: {hsc_embeddings_1.shape}")
print(f"  HSC embeddings 2 shape: {hsc_embeddings_2.shape}")
print(f"  Legacy images shape:    {legacy_images.shape}")
print(f"  Legacy embeddings 1 shape: {legacy_embeddings_1.shape}")
print(f"  Legacy embeddings 2 shape: {legacy_embeddings_2.shape}")
print(f"\n  Embedding shape breakdown: (batch_size, seq_len, embed_dim)")
print(f"    - seq_len = spatial locations (H/32 * W/32 for ResNet18)")
print(f"    - embed_dim = cross_attention_dim = {model.hparams.cross_attention_dim}")

# Prepare embeddings for encoder 1
all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)
if avg_latent_space:
    all_embeddings_1 = torch.mean(all_embeddings_1, dim=1)
else:
    all_embeddings_1 = all_embeddings_1.flatten(start_dim=1)
print(f"\nEncoder 1 flattened embeddings shape: {all_embeddings_1.shape}")

# Prepare embeddings for encoder 2
all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)
if avg_latent_space:
    all_embeddings_2 = torch.mean(all_embeddings_2, dim=1)
else:
    all_embeddings_2 = all_embeddings_2.flatten(start_dim=1)
print(f"Encoder 2 flattened embeddings shape: {all_embeddings_2.shape}")


num_hsc = hsc_embeddings_1.shape[0]

if GENERATE_UMAP:
    figures_dir = LATENT_SPACE_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # UMAP parameters
    umap_params = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_components': 2,
        'metric': 'euclidean',
        'random_state': 42,
    }

    # ===== Encoder 1 UMAP =====
    print('\nStarting UMAP calculation for Encoder 1...')
    reducer_1 = umap.UMAP(**umap_params)
    embedding_1 = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())

    hsc_embedding_1 = embedding_1[:num_hsc]
    legacy_embedding_1 = embedding_1[num_hsc:]

    # ===== Encoder 2 UMAP =====
    print('\nStarting UMAP calculation for Encoder 2...')
    reducer_2 = umap.UMAP(**umap_params)
    embedding_2 = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())

    hsc_embedding_2 = embedding_2[:num_hsc]
    legacy_embedding_2 = embedding_2[num_hsc:]

# ===== Distance Calculations =====
print("\n" + "="*60)
print("Distance Analysis in Latent Spaces")
print("="*60)

# Calculate distances in the original embedding space (before UMAP)
# Use the flattened embeddings for distance calculations
hsc_flat_1 = all_embeddings_1[:num_hsc].cpu().numpy()
legacy_flat_1 = all_embeddings_1[num_hsc:].cpu().numpy()

hsc_flat_2 = all_embeddings_2[:num_hsc].cpu().numpy()
legacy_flat_2 = all_embeddings_2[num_hsc:].cpu().numpy()

# 1) Average distance between pairs in Space 1 (Physics Latent Space)
paired_distances_1 = np.linalg.norm(hsc_flat_1 - legacy_flat_1, axis=1)
avg_paired_distance_1 = np.mean(paired_distances_1)
print(f"\n1) Average distance between pairs in Space 1 (Physics): {avg_paired_distance_1:.4f}")

# 2) Average distance between random unpaired examples in Space 1
np.random.seed(42)
shuffled_indices = np.random.permutation(num_hsc)
unpaired_distances_1 = np.linalg.norm(hsc_flat_1 - legacy_flat_1[shuffled_indices], axis=1)
avg_unpaired_distance_1 = np.mean(unpaired_distances_1)
print(f"2) Average distance between random unpaired in Space 1: {avg_unpaired_distance_1:.4f}")

# 3) Average distance between pairs in Space 2 (Instrument Latent Space)
paired_distances_2 = np.linalg.norm(hsc_flat_2 - legacy_flat_2, axis=1)
avg_paired_distance_2 = np.mean(paired_distances_2)
print(f"\n3) Average distance between pairs in Space 2 (Instrument): {avg_paired_distance_2:.4f}")

# 4) Average distance between random unpaired examples in Space 2
unpaired_distances_2 = np.linalg.norm(hsc_flat_2 - legacy_flat_2[shuffled_indices], axis=1)
avg_unpaired_distance_2 = np.mean(unpaired_distances_2)
print(f"4) Average distance between random unpaired in Space 2: {avg_unpaired_distance_2:.4f}")

# 5) Normalized distances (paired / unpaired)
normalized_distance_1 = avg_paired_distance_1 / avg_unpaired_distance_1
normalized_distance_2 = avg_paired_distance_2 / avg_unpaired_distance_2

print(f"\n5) Normalized distances (paired / unpaired):")
print(f"   Space 1 (Physics): {normalized_distance_1:.4f}")
print(f"   Space 2 (Instrument): {normalized_distance_2:.4f}")
print("="*60)

if GENERATE_UMAP:
    # Randomly select 20 pairs to highlight (if enabled)
    selected_indices = None
    pair_colors = None
    pair_markers = None
    if SHOW_PAIRS:
        np.random.seed(42)  # For reproducibility
        num_pairs_to_highlight = 20
        selected_indices = np.random.choice(num_hsc, size=num_pairs_to_highlight, replace=False)
        print(f"\nSelected {num_pairs_to_highlight} random pairs to highlight: indices {selected_indices}")
        # 5 colors and 4 shapes for 20 unique combinations
        pair_colors = plt.cm.tab10(np.linspace(0, 1, 5))  # 5 colors
        pair_markers = ['x', 's', 'o', '^']  # 4 shapes: X, square, circle, triangle

    # Save UMAP data to same directory as plots for quick tuned plotting
    zoom_suffix = '_zoom' if zoom_val else ''
    latent_suffix = '_avg' if avg_latent_space else '_flat'
    umap_stem = f'umap_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}{latent_suffix}'
    umap_data_path = figures_dir / f'{umap_stem}_data.npz'
    umap_meta_path = figures_dir / f'{umap_stem}_metadata.json'
    np.savez_compressed(
        umap_data_path,
        hsc_umap_1=hsc_embedding_1,
        legacy_umap_1=legacy_embedding_1,
        hsc_umap_2=hsc_embedding_2,
        legacy_umap_2=legacy_embedding_2,
        selected_indices=selected_indices if selected_indices is not None else np.array([], dtype=np.int64),
    )
    with open(umap_meta_path, 'w') as f:
        json.dump({
            'epoch': int(epoch),
            'dim': int(dim),
            'mode_tag': mode_tag,
            'zoom_val': zoom_val,
            'avg_latent_space': avg_latent_space,
            'num_hsc': int(num_hsc),
        }, f, indent=2)
    print(f"UMAP data saved to '{umap_data_path}' and metadata to '{umap_meta_path}'")

    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Encoder 1 plot
    ax1.scatter(hsc_embedding_1[:, 0], hsc_embedding_1[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax1.scatter(legacy_embedding_1[:, 0], legacy_embedding_1[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 1 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax1.scatter(hsc_embedding_1[idx, 0], hsc_embedding_1[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax1.scatter(legacy_embedding_1[idx, 0], legacy_embedding_1[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax1.set_title(f'Encoder 1 (Same Galaxy) \n Physics Latent space (UMAP). Epoch {epoch}')
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.legend()
    ax1.grid(True)

    # Encoder 2 plot
    ax2.scatter(hsc_embedding_2[:, 0], hsc_embedding_2[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax2.scatter(legacy_embedding_2[:, 0], legacy_embedding_2[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 2 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax2.scatter(hsc_embedding_2[idx, 0], hsc_embedding_2[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax2.scatter(legacy_embedding_2[idx, 0], legacy_embedding_2[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax2.set_title(f'Encoder 2 (Same Instrument) \n Instrument Latent space (UMAP). Epoch {epoch}')
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    combined_path = figures_dir / f'{umap_stem}.png'
    plt.savefig(combined_path, dpi=150)
    plt.close()

    print(f"\nCombined UMAP plot saved to '{combined_path}'")
    print(f"  HSC points: {num_hsc}")
    print(f"  Legacy points: {len(legacy_embedding_1)}")


# ===== PCA Visualization =====
if GENERATE_PCA:
    figures_dir = LATENT_SPACE_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # PCA parameters
    pca_params = {
        'n_components': 2,
        'random_state': 42,
    }

    # ===== Encoder 1 PCA =====
    print('\nStarting PCA calculation for Encoder 1...')
    pca_1 = PCA(**pca_params)
    embedding_1_pca = pca_1.fit_transform(all_embeddings_1.cpu().numpy())
    explained_variance_1 = pca_1.explained_variance_ratio_
    print(f"  Explained variance ratio (Encoder 1): {explained_variance_1}")
    print(f"  Total explained variance (Encoder 1): {explained_variance_1.sum():.4f}")

    hsc_embedding_1_pca = embedding_1_pca[:num_hsc]
    legacy_embedding_1_pca = embedding_1_pca[num_hsc:]

    # ===== Encoder 2 PCA =====
    print('\nStarting PCA calculation for Encoder 2...')
    pca_2 = PCA(**pca_params)
    embedding_2_pca = pca_2.fit_transform(all_embeddings_2.cpu().numpy())
    explained_variance_2 = pca_2.explained_variance_ratio_
    print(f"  Explained variance ratio (Encoder 2): {explained_variance_2}")
    print(f"  Total explained variance (Encoder 2): {explained_variance_2.sum():.4f}")

    hsc_embedding_2_pca = embedding_2_pca[:num_hsc]
    legacy_embedding_2_pca = embedding_2_pca[num_hsc:]

    # Randomly select 20 pairs to highlight (if enabled)
    selected_indices = None
    pair_colors = None
    pair_markers = None
    if SHOW_PAIRS:
        np.random.seed(42)  # For reproducibility
        num_pairs_to_highlight = 20
        selected_indices = np.random.choice(num_hsc, size=num_pairs_to_highlight, replace=False)
        print(f"\nSelected {num_pairs_to_highlight} random pairs to highlight: indices {selected_indices}")
        # 5 colors and 4 shapes for 20 unique combinations
        pair_colors = plt.cm.tab10(np.linspace(0, 1, 5))  # 5 colors
        pair_markers = ['x', 's', 'o', '^']  # 4 shapes: X, square, circle, triangle

    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Encoder 1 plot
    ax1.scatter(hsc_embedding_1_pca[:, 0], hsc_embedding_1_pca[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax1.scatter(legacy_embedding_1_pca[:, 0], legacy_embedding_1_pca[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 1 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax1.scatter(hsc_embedding_1_pca[idx, 0], hsc_embedding_1_pca[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax1.scatter(legacy_embedding_1_pca[idx, 0], legacy_embedding_1_pca[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax1.set_title(f'Encoder 1 (Same Galaxy) \n Physics Latent space (PCA). Epoch {epoch}\n'
                  f'Explained variance: {explained_variance_1.sum():.2%}')
    ax1.set_xlabel(f'PC1 ({explained_variance_1[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({explained_variance_1[1]:.2%} variance)')
    ax1.legend()
    ax1.grid(True)

    # Encoder 2 plot
    ax2.scatter(hsc_embedding_2_pca[:, 0], hsc_embedding_2_pca[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax2.scatter(legacy_embedding_2_pca[:, 0], legacy_embedding_2_pca[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 2 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax2.scatter(hsc_embedding_2_pca[idx, 0], hsc_embedding_2_pca[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax2.scatter(legacy_embedding_2_pca[idx, 0], legacy_embedding_2_pca[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax2.set_title(f'Encoder 2 (Same Instrument) \n Instrument Latent space (PCA). Epoch {epoch}\n'
                  f'Explained variance: {explained_variance_2.sum():.2%}')
    ax2.set_xlabel(f'PC1 ({explained_variance_2[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({explained_variance_2[1]:.2%} variance)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    zoom_suffix = '_zoom' if zoom_val else ''
    combined_path = figures_dir / f'pca_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}.png'
    plt.savefig(combined_path, dpi=150)
    plt.close()

    print(f"\nCombined PCA plot saved to '{combined_path}'")
    print(f"  HSC points: {num_hsc}")
    print(f"  Legacy points: {len(legacy_embedding_1_pca)}")


# ===== t-SNE Visualization =====
if GENERATE_TSNE:
    figures_dir = LATENT_SPACE_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE parameters
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'random_state': 42,
        'max_iter': 1000,
    }

    # ===== Encoder 1 t-SNE =====
    print('\nStarting t-SNE calculation for Encoder 1...')
    print('  (This may take a while for large datasets)')
    tsne_1 = TSNE(**tsne_params)
    embedding_1_tsne = tsne_1.fit_transform(all_embeddings_1.cpu().numpy())

    hsc_embedding_1_tsne = embedding_1_tsne[:num_hsc]
    legacy_embedding_1_tsne = embedding_1_tsne[num_hsc:]

    # ===== Encoder 2 t-SNE =====
    print('\nStarting t-SNE calculation for Encoder 2...')
    print('  (This may take a while for large datasets)')
    tsne_2 = TSNE(**tsne_params)
    embedding_2_tsne = tsne_2.fit_transform(all_embeddings_2.cpu().numpy())

    hsc_embedding_2_tsne = embedding_2_tsne[:num_hsc]
    legacy_embedding_2_tsne = embedding_2_tsne[num_hsc:]

    # Randomly select 20 pairs to highlight (if enabled)
    selected_indices = None
    pair_colors = None
    pair_markers = None
    if SHOW_PAIRS:
        np.random.seed(42)  # For reproducibility
        num_pairs_to_highlight = 20
        selected_indices = np.random.choice(num_hsc, size=num_pairs_to_highlight, replace=False)
        print(f"\nSelected {num_pairs_to_highlight} random pairs to highlight: indices {selected_indices}")
        # 5 colors and 4 shapes for 20 unique combinations
        pair_colors = plt.cm.tab10(np.linspace(0, 1, 5))  # 5 colors
        pair_markers = ['x', 's', 'o', '^']  # 4 shapes: X, square, circle, triangle

    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Encoder 1 plot
    ax1.scatter(hsc_embedding_1_tsne[:, 0], hsc_embedding_1_tsne[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax1.scatter(legacy_embedding_1_tsne[:, 0], legacy_embedding_1_tsne[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 1 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax1.scatter(hsc_embedding_1_tsne[idx, 0], hsc_embedding_1_tsne[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax1.scatter(legacy_embedding_1_tsne[idx, 0], legacy_embedding_1_tsne[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax1.set_title(f'Encoder 1 (Same Galaxy) \n Physics Latent space (t-SNE). Epoch {epoch}')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend()
    ax1.grid(True)

    # Encoder 2 plot
    ax2.scatter(hsc_embedding_2_tsne[:, 0], hsc_embedding_2_tsne[:, 1], s=5, label='HSC', alpha=0.6, c=COLOR_HSC)
    ax2.scatter(legacy_embedding_2_tsne[:, 0], legacy_embedding_2_tsne[:, 1], s=5, label='Legacy', alpha=0.6, c=COLOR_LEGACY)

    # Mark selected pairs on Encoder 2 plot with matching colors and shapes (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i % 5]  # Cycle through 5 colors
            marker = pair_markers[i % 4]  # Cycle through 4 shapes
            ax2.scatter(hsc_embedding_2_tsne[idx, 0], hsc_embedding_2_tsne[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')
            ax2.scatter(legacy_embedding_2_tsne[idx, 0], legacy_embedding_2_tsne[idx, 1],
                        marker=marker, s=200, c=[color], linewidths=3, zorder=5, edgecolors='black')

    ax2.set_title(f'Encoder 2 (Same Instrument) \n Instrument Latent space (t-SNE). Epoch {epoch}')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    zoom_suffix = '_zoom' if zoom_val else ''
    latent_suffix = '_avg' if avg_latent_space else '_flat'
    combined_path = figures_dir / f'tsne_both_encoders_zdim{dim}{mode_tag}{zoom_suffix}{latent_suffix}.png'
    plt.savefig(combined_path, dpi=150)
    plt.close()

    print(f"\nCombined t-SNE plot saved to '{combined_path}'")
    print(f"  HSC points: {num_hsc}")
    print(f"  Legacy points: {len(legacy_embedding_1_tsne)}")





#############
# Generation test
#############

if GENERATE_SAMPLES:
    print('\n' + '='*60)
    print('Generation Examples')
    print('='*60)

    # Configuration
    N = 6  # Number of target examples
    target_indices = [0, 100, 200, 300, 400, 500]

    def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> torch.Tensor:
        """
        Scale a (3,H,W) tensor to (H,W,3) in [0,1] using fixed per-channel vmin/vmax.
        vmin/vmax: tensor-like shape (3,)
        """
        x = x_chw[:3]
        vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
        vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
        y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
        y = y.clamp(0, 1)
        return y.permute(1, 2, 0)

    # Store all results for plotting
    all_results = []

    for target_idx in target_indices:
        print(f'\n--- Processing Target {target_idx} ---')

        # Target: hsc[target_idx] (the image we want to reconstruct)
        target_image = hsc_images[target_idx:target_idx+1].to(device)  # (1, C, H, W)
        target_type = "HSC"

        # Same galaxy: legacy[target_idx] (same galaxy as target, different instrument)
        samegal_image = legacy_images[target_idx:target_idx+1].to(device)  # (1, C, H, W)
        samegal_type = "Legacy"

        print(f"Target: {target_type}[{target_idx}]")
        print(f"Same Galaxy: {samegal_type}[{target_idx}]")

        # Generate noise once for this target (will be reused for all samples)
        noise = torch.randn(
            1, model.hparams.in_channels, model.hparams.image_size, model.hparams.image_size,
            device=device,
        )
        print(f"Generated shared noise for all samples (shape: {noise.shape})")

        # Generate 6 samples with different sameins conditions
        samples = []
        sample_descriptions = []

        # Sample 1: 5 sameins images from hsc[target_idx+1:target_idx+6]
        print('Sample 1: sameins hsc[{}:{}] (5 images)'.format(target_idx+1, target_idx+6))
        sameins_1 = hsc_images[target_idx+1:target_idx+6].to(device).unsqueeze(0)  # (1, 5, C, H, W)
        sample_1 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_1, x_noise=noise)
        samples.append(sample_1[0])
        sample_descriptions.append("Sample 1\nHSC[{}-{}] (5)".format(target_idx+1, target_idx+5))

        # Sample 2: 5 sameins images from hsc[target_idx+6:target_idx+11]
        print('Sample 2: sameins hsc[{}:{}] (5 images)'.format(target_idx+6, target_idx+11))
        sameins_2 = hsc_images[target_idx+6:target_idx+11].to(device).unsqueeze(0)  # (1, 5, C, H, W)
        sample_2 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_2, x_noise=noise)
        samples.append(sample_2[0])
        sample_descriptions.append("Sample 2\nHSC[{}-{}] (5)".format(target_idx+6, target_idx+10))

        # Sample 3: 5 sameins images from hsc[target_idx+11:target_idx+16]
        print('Sample 3: sameins hsc[{}:{}] (5 images)'.format(target_idx+11, target_idx+16))
        sameins_3 = hsc_images[target_idx+11:target_idx+16].to(device).unsqueeze(0)  # (1, 5, C, H, W)
        sample_3 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_3, x_noise=noise)
        samples.append(sample_3[0])
        sample_descriptions.append("Sample 3\nHSC[{}-{}] (5)".format(target_idx+11, target_idx+15))

        # Sample 4: 5 sameins images from hsc[target_idx+16:target_idx+21]
        print('Sample 4: sameins hsc[{}:{}] (5 images)'.format(target_idx+16, target_idx+21))
        sameins_4 = hsc_images[target_idx+16:target_idx+21].to(device).unsqueeze(0)  # (1, 5, C, H, W)
        sample_4 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_4, x_noise=noise)
        samples.append(sample_4[0])
        sample_descriptions.append("Sample 4\nHSC[{}-{}] (5)".format(target_idx+16, target_idx+20))

        # Sample 5: 1 sameins image from hsc[target_idx+2]
        print('Sample 5: sameins hsc[{}] (1 image)'.format(target_idx+2))
        sameins_5 = hsc_images[target_idx+2:target_idx+3].to(device).unsqueeze(0)  # (1, 1, C, H, W)
        sample_5 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_5, x_noise=noise)
        samples.append(sample_5[0])
        sample_descriptions.append("Sample 5\nHSC[{}] (1)".format(target_idx+2))

        # Sample 6: 3 sameins images from hsc[target_idx+3:target_idx+6]
        print('Sample 6: sameins hsc[{}:{}] (3 images)'.format(target_idx+3, target_idx+6))
        sameins_6 = hsc_images[target_idx+3:target_idx+6].to(device).unsqueeze(0)  # (1, 3, C, H, W)
        sample_6 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_6, x_noise=noise)
        samples.append(sample_6[0])
        sample_descriptions.append("Sample 6\nHSC[{}-{}] (3)".format(target_idx+3, target_idx+5))

        # Sample 7: 5 sameins images from legacy[target_idx+1:target_idx+6]
        print('Sample 7: sameins legacy[{}:{}] (5 images)'.format(target_idx+1, target_idx+6))
        sameins_7 = legacy_images[target_idx+1:target_idx+6].to(device).unsqueeze(0)  # (1, 5, C, H, W)
        sample_7 = model.sample(cond_image_samegal=samegal_image, cond_image_sameins=sameins_7, x_noise=noise)
        samples.append(sample_7[0])
        sample_descriptions.append("Sample 7\nLegacy[{}-{}] (5)".format(target_idx+1, target_idx+5))

        # Store results
        all_results.append({
            'target_idx': target_idx,
            'target_image': target_image,
            'target_type': target_type,
            'samegal_image': samegal_image,
            'samegal_type': samegal_type,
            'samegal_idx': target_idx,
            'samples': samples,
            'sample_descriptions': sample_descriptions
        })

    print('\n' + '='*60)
    print('Generation complete!')
    print('='*60)

    # Create visualization with vertical separators
    print('\nCreating visualization...')

    # Layout: target | samegal || sample1-4 || sample5 (1img) | sample6 (3img) | sample7 (legacy 5img)
    # Total columns: 2 + 4 + 2 + 1 = 9 columns per row
    num_cols = 9
    num_rows = N

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(27, 3 * num_rows))

    # If only one row, make axes 2D
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, result in enumerate(all_results):
        target_image = result['target_image']
        samegal_image = result['samegal_image']
        samples = result['samples']
        sample_descriptions = result['sample_descriptions']
        target_idx = result['target_idx']

        # Compute per-channel vmin/vmax from the TARGET for this row (row-scaled visualization)
        target_chw = target_image[0, :3]  # (3,H,W)
        vmin = target_chw.amin(dim=(1, 2))  # (3,)
        vmax = target_chw.amax(dim=(1, 2))  # (3,)

        # Column 0: Target
        target_vis = _row_scale_rgb(target_image[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[row_idx, 0].imshow(target_vis)
        axes[row_idx, 0].set_title(f'Target\n{result["target_type"]}[{target_idx}]', fontsize=9)
        axes[row_idx, 0].axis('off')

        # Column 1: Same Galaxy
        samegal_vis = _row_scale_rgb(samegal_image[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[row_idx, 1].imshow(samegal_vis)
        axes[row_idx, 1].set_title(f'SameGal\n{result["samegal_type"]}[{target_idx}]', fontsize=9)
        axes[row_idx, 1].axis('off')

        # Columns 2-5: Samples 1-4 (5 images each)
        for i in range(4):
            samp_vis = _row_scale_rgb(samples[i][:3], vmin, vmax).detach().cpu().numpy()
            axes[row_idx, i + 2].imshow(samp_vis)
            axes[row_idx, i + 2].set_title(sample_descriptions[i], fontsize=9)
            axes[row_idx, i + 2].axis('off')

        # Column 6: Sample 5 (1 image)
        samp_vis = _row_scale_rgb(samples[4][:3], vmin, vmax).detach().cpu().numpy()
        axes[row_idx, 6].imshow(samp_vis)
        axes[row_idx, 6].set_title(sample_descriptions[4], fontsize=9)
        axes[row_idx, 6].axis('off')

        # Column 7: Sample 6 (3 images)
        samp_vis = _row_scale_rgb(samples[5][:3], vmin, vmax).detach().cpu().numpy()
        axes[row_idx, 7].imshow(samp_vis)
        axes[row_idx, 7].set_title(sample_descriptions[5], fontsize=9)
        axes[row_idx, 7].axis('off')

        # Column 8: Sample 7 (5 legacy images)
        samp_vis = _row_scale_rgb(samples[6][:3], vmin, vmax).detach().cpu().numpy()
        axes[row_idx, 8].imshow(samp_vis)
        axes[row_idx, 8].set_title(sample_descriptions[6], fontsize=9)
        axes[row_idx, 8].axis('off')

    # Add column labels at the top
    col_labels = ['Target', 'SameGal', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7']
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes,
                              ha='center', va='bottom', fontsize=10, weight='bold')

    plt.suptitle(f'Reconstructions: Multiple Targets with Different SameIns Conditions. Epoch {epoch}',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    # Add vertical separators using figure coordinates
    # Separator 1: between SameGal (col 1) and Sample 1 (col 2)
    # Separator 2: between Sample 4 (col 5) and Sample 5 (col 6)
    # Use the first row to determine separator positions
    bbox1 = axes[0, 1].get_position()
    bbox2 = axes[0, 2].get_position()
    separator_x1 = (bbox1.x1 + bbox2.x0) / 2

    bbox5 = axes[0, 5].get_position()
    bbox6 = axes[0, 6].get_position()
    separator_x2 = (bbox5.x1 + bbox6.x0) / 2

    # Get the bottom and top of the figure (accounting for title)
    bbox_bottom = axes[-1, 0].get_position().y0
    bbox_top = axes[0, 0].get_position().y1

    # Draw vertical lines across all rows
    fig.add_artist(plt.Line2D([separator_x1, separator_x1], [bbox_bottom, bbox_top],
                              transform=fig.transFigure, color='black', linewidth=2,
                              clip_on=False))
    fig.add_artist(plt.Line2D([separator_x2, separator_x2], [bbox_bottom, bbox_top],
                              transform=fig.transFigure, color='black', linewidth=2,
                              clip_on=False))

    figures_dir = GEN_STUDY_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = '_zoom' if zoom_val else ''
    reconstruction_path = figures_dir / f'reconstructions_comparison_zdim{dim}{mode_tag}{zoom_suffix}.png'
    plt.savefig(reconstruction_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Reconstruction comparison saved to '{reconstruction_path}'")


### Latent prediction study

# Given a set of pairs hsc_iamge, legacy_image
# Calculate the galaxy/physics latents (looking at encoder_1)
# So we have z_hsc_ph, z_legacy_ph
# And calculate the instrument latents (looking at encoder_2)
# So we have z_hsc_ins, z_legacy_ins
# Then, train a
