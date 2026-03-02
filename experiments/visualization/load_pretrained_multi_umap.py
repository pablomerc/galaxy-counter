"""
Load a pretrained model and run UMAP visualization for each token position.
For the double-encoder model with embeddings of shape (B, seq_len=4, z_dim)
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from double_train_fm import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader
from data import HSCLegacyDataset
import time

import umap
import matplotlib.pyplot as plt
import numpy as np

# Checkpoint configuration
checkpoints = [
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/rach5aeu/checkpoints/latest-step=step=66000.ckpt', # z_dim = 8
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wdbsh3rc/checkpoints/latest-step=step=62000.ckpt', # z_dim = 8 geom
    '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/muia8i77/checkpoints/latest-step=step=40000.ckpt', # z_dim = 64
]

mode = ['', 'geom', '']
z_dim_list = [8, 8, 64]
epochs = [190, 190, 190]

zoom = True

# Start with model idx = 0
idx = 1
checkpoint_path = checkpoints[idx]
dim = z_dim_list[idx]
epoch = epochs[idx]
mode_tag = mode[idx]

# Control flags
SHOW_PAIRS = True  # Set to False to skip marking pairs on the plots

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
print(f"\nLoading model from checkpoint: {checkpoint_path}")
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')

# Set the model to evaluation mode and disable gradient calculation for inference
model.eval()
torch.set_grad_enabled(False)

# Move model to the determined device
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# Time dataset initialization (loading from HDF5 into memory)
dataset_start = time.perf_counter()
dataset = HSCLegacyDataset(
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
actual_batch_size = batch[0].shape[0]

total_time = dataset_time + loader_time + batch_time

print(f"\nTiming breakdown for {actual_batch_size} examples:")
print(f"  Dataset initialization (HDF5 → memory): {dataset_time:.4f} s")
print(f"  DataLoader creation:                    {loader_time:.4f} s")
print(f"  First batch retrieval:                  {batch_time:.4f} s")
print(f"  Total time:                             {total_time:.4f} s")

# Move images to device (they come from DataLoader on CPU by default)
hsc_images = batch[0].to(device)
legacy_images = batch[1].to(device)

# Encode images with both encoders
print("\nEncoding images...")
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
print(f"\n  Embedding shape breakdown: (batch_size, seq_len, z_dim)")
print(f"    - seq_len = {hsc_embeddings_1.shape[1]}")
print(f"    - z_dim = {hsc_embeddings_1.shape[2]}")

# Verify seq_len is 4
seq_len = hsc_embeddings_1.shape[1]
assert seq_len == 4, f"Expected seq_len=4, got {seq_len}"

num_hsc = hsc_embeddings_1.shape[0]

# Prepare embeddings for each encoder (DO NOT flatten - keep token structure)
all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)  # (B_total, seq_len=4, z_dim)
all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)  # (B_total, seq_len=4, z_dim)

print(f"\nCombined embeddings shape:")
print(f"  Encoder 1: {all_embeddings_1.shape}")
print(f"  Encoder 2: {all_embeddings_2.shape}")

# Create figures directory
figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

# UMAP parameters
umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
}

# Randomly select pairs to highlight (if enabled)
selected_indices = None
pair_colors = None
if SHOW_PAIRS:
    np.random.seed(42)  # For reproducibility
    num_pairs_to_highlight = 5
    selected_indices = np.random.choice(num_hsc, size=num_pairs_to_highlight, replace=False)
    print(f"\nSelected {num_pairs_to_highlight} random pairs to highlight: indices {selected_indices}")
    # Colors for each pair
    pair_colors = plt.cm.tab10(np.linspace(0, 1, num_pairs_to_highlight))

# Create figure with 4 rows (one per token) and 2 columns (encoder_1, encoder_2)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))

# Process each token position (0, 1, 2, 3)
for token_idx in range(4):
    print(f"\n{'='*60}")
    print(f"Processing Token Position {token_idx}")
    print(f"{'='*60}")

    # Extract embeddings for this token position
    # Shape: (B_total, z_dim)
    token_embeddings_1 = all_embeddings_1[:, token_idx, :].cpu().numpy()
    token_embeddings_2 = all_embeddings_2[:, token_idx, :].cpu().numpy()

    print(f"Token {token_idx} embeddings shape - Encoder 1: {token_embeddings_1.shape}")
    print(f"Token {token_idx} embeddings shape - Encoder 2: {token_embeddings_2.shape}")

    # Split into HSC and Legacy
    hsc_token_1 = token_embeddings_1[:num_hsc]
    legacy_token_1 = token_embeddings_1[num_hsc:]
    hsc_token_2 = token_embeddings_2[:num_hsc]
    legacy_token_2 = token_embeddings_2[num_hsc:]

    # ===== Encoder 1 UMAP for this token =====
    print(f'  Computing UMAP for Encoder 1, Token {token_idx}...')
    reducer_1 = umap.UMAP(**umap_params)
    embedding_1_umap = reducer_1.fit_transform(token_embeddings_1)

    hsc_embedding_1_umap = embedding_1_umap[:num_hsc]
    legacy_embedding_1_umap = embedding_1_umap[num_hsc:]

    # ===== Encoder 2 UMAP for this token =====
    print(f'  Computing UMAP for Encoder 2, Token {token_idx}...')
    reducer_2 = umap.UMAP(**umap_params)
    embedding_2_umap = reducer_2.fit_transform(token_embeddings_2)

    hsc_embedding_2_umap = embedding_2_umap[:num_hsc]
    legacy_embedding_2_umap = embedding_2_umap[num_hsc:]

    # Plot Encoder 1 (left column)
    ax1 = axes[token_idx, 0]
    ax1.scatter(hsc_embedding_1_umap[:, 0], hsc_embedding_1_umap[:, 1],
                s=5, label='HSC', alpha=0.6, c='blue')
    ax1.scatter(legacy_embedding_1_umap[:, 0], legacy_embedding_1_umap[:, 1],
                s=5, label='Legacy', alpha=0.6, c='orange')

    # Mark selected pairs on Encoder 1 plot with matching colors (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i]
            ax1.scatter(hsc_embedding_1_umap[idx, 0], hsc_embedding_1_umap[idx, 1],
                        marker='x', s=200, c=[color], linewidths=3, zorder=5)
            ax1.scatter(legacy_embedding_1_umap[idx, 0], legacy_embedding_1_umap[idx, 1],
                        marker='x', s=200, c=[color], linewidths=3, zorder=5)

    ax1.set_title(f'Encoder 1 (Same Galaxy) - Token {token_idx}\nPhysics Latent Space (UMAP). Epoch {epoch}', fontsize=10)
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.legend()
    ax1.grid(True)

    # Plot Encoder 2 (right column)
    ax2 = axes[token_idx, 1]
    ax2.scatter(hsc_embedding_2_umap[:, 0], hsc_embedding_2_umap[:, 1],
                s=5, label='HSC', alpha=0.6, c='blue')
    ax2.scatter(legacy_embedding_2_umap[:, 0], legacy_embedding_2_umap[:, 1],
                s=5, label='Legacy', alpha=0.6, c='orange')

    # Mark selected pairs on Encoder 2 plot with matching colors (if enabled)
    if SHOW_PAIRS and selected_indices is not None:
        for i, idx in enumerate(selected_indices):
            color = pair_colors[i]
            ax2.scatter(hsc_embedding_2_umap[idx, 0], hsc_embedding_2_umap[idx, 1],
                        marker='x', s=200, c=[color], linewidths=3, zorder=5)
            ax2.scatter(legacy_embedding_2_umap[idx, 0], legacy_embedding_2_umap[idx, 1],
                        marker='x', s=200, c=[color], linewidths=3, zorder=5)

    ax2.set_title(f'Encoder 2 (Same Instrument) - Token {token_idx}\nInstrument Latent Space (UMAP). Epoch {epoch}', fontsize=10)
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.legend()
    ax2.grid(True)

# Add column labels at the top
col_labels = ['Encoder 1 (Physics)', 'Encoder 2 (Instrument)']
for col_idx, label in enumerate(col_labels):
    axes[0, col_idx].text(0.5, 1.15, label, transform=axes[0, col_idx].transAxes,
                          ha='center', va='bottom', fontsize=12, weight='bold')

plt.suptitle(f'UMAP Visualization by Token Position (z_dim={dim}, Epoch {epoch})',
             fontsize=14, y=0.995)
plt.tight_layout()

# Save figure
zoom_suffix = '_zoom' if zoom else ''
mode_suffix = f'_{mode_tag}' if mode_tag else ''
combined_path = figures_dir / f'umap_multi_token_zdim{dim}{mode_suffix}{zoom_suffix}.png'
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print(f"Multi-token UMAP plot saved to '{combined_path}'")
print(f"  HSC points: {num_hsc}")
print(f"  Legacy points: {len(legacy_embedding_1_umap)}")
print(f"  Token positions: 0, 1, 2, 3")
print(f"{'='*60}")
