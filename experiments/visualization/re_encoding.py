"""
Script that loads a trained model, runs a batch of 32 images twice:
- First with original same_ins images
- Second with shifted same_ins images (using next element in batch)
Then takes the outputs of the model and encodes them again.
Then takes a batch of 8192 examples + special re-encoded ones (included in the 8192)
and makes a UMAP visualization. It marks targets, two generated samples, and condition images
with different markers ('s', '^', 'v', 'x', 'o') and colors for 5 examples.
"""

# Imports


import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import importlib.util
# from double_train_fm import ConditionalFlowMatchingModule
# Import module with hyphen in name using importlib
spec = importlib.util.spec_from_file_location(
    "double_train_fm_no_attn",
    Path(__file__).parent.parent / "double_train_fm_no-attn.py"
)
double_train_fm_no_attn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(double_train_fm_no_attn)
ConditionalFlowMatchingModule = double_train_fm_no_attn.ConditionalFlowMatchingModule


from torch.utils.data import DataLoader
from data import HSCLegacyDatasetZoom, HSCLegacyTripletDatasetZoom
import numpy as np
import h5py
import umap
import matplotlib.pyplot as plt


# Model checkpoint
# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt'  # z_dim = 16, geom, attn

checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/srj4opub/checkpoints/latest-step=step=75000.ckpt' # z_dim = 64, class cond

mode_suffix = 'classcond'

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

val_dataset = HSCLegacyTripletDatasetZoom(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000, 100_000)),
        deterministic_anchor_survey=True,  # Make validation batches consistent
)

data_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
batch = next(iter(data_loader))

x_1, cond_image_samegal, cond_image_sameins, metadata = batch

# Move tensors to device
x_1 = x_1.to(device)
cond_image_samegal = cond_image_samegal.to(device)
cond_image_sameins = cond_image_sameins.to(device)

print('About to generate samples')

print('Shape cond_image_sameins',cond_image_sameins.shape)

# Keep the k dimension: use [:32] to get (32, 5, 4, 48, 48) or [:32, 0:1, :, :, :] to get (32, 1, 4, 48, 48)
# Using [:32] to keep all 5 instrument images, or [:32, 0:1, :, :, :] to use just the first one
# Generate first set of samples with original same_ins
samples_1 = model.sample(cond_image_samegal[:32], cond_image_sameins[:32])

# Generate second set of samples with shifted same_ins (using next element in batch)
# For element i, use same_ins from element i+1 (wrapping around if needed)
cond_image_sameins_2 = cond_image_sameins[32:64]
samples_2 = model.sample(cond_image_samegal[:32], cond_image_sameins_2)

print('Shape of generated samples 1')
print(samples_1.shape)
print('Shape of generated samples 2')
print(samples_2.shape)

# lets encode

embedding_target_enc1 = model.encoder_1(x_1)
embedding_generated_samples_1_enc1 = model.encoder_1(samples_1)
embedding_generated_samples_2_enc1 = model.encoder_1(samples_2)
embedding_samegal = model.encoder_1(cond_image_samegal)

embedding_target_enc2 = model.encoder_2(x_1)
embedding_generated_samples_1_enc2 = model.encoder_2(samples_1)
embedding_generated_samples_2_enc2 = model.encoder_2(samples_2)
# Encode all 5 instrument images for both sameins_1 and sameins_2
# Following the approach in double_train_fm.py: flatten, encode all at once
# cond_image_sameins has shape (32, 5, 4, 48, 48)
cond_image_sameins_1_flat = cond_image_sameins[:32].flatten(0, 1)  # (32*5, 4, 48, 48)
cond_image_sameins_2_flat = cond_image_sameins_2.flatten(0, 1)      # (32*5, 4, 48, 48)
embedding_sameins_1_all5 = model.encoder_2(cond_image_sameins_1_flat)  # (32*5, seq_len, embed_dim)
embedding_sameins_2_all5 = model.encoder_2(cond_image_sameins_2_flat)   # (32*5, seq_len, embed_dim)

# Take all of the encoder 1 and all of the encoder 2 embeddings together and make a UMAP.

print('\nLoading 8192 examples for UMAP visualization...')
num_examples_umap = 8192
num_batches_needed = (num_examples_umap + 1023) // 1024  # Ceiling division

# Collect embeddings from multiple batches
all_embeddings_enc1_list = []
all_embeddings_enc2_list = []

# Process batches to get 8192 examples
batch_count = 0
with torch.no_grad():
    for batch in data_loader:
        if batch_count >= num_batches_needed:
            break

        x_batch, cond_gal_batch, cond_ins_batch, _ = batch
        x_batch = x_batch.to(device)

        # Encode batch
        emb_enc1 = model.encoder_1(x_batch)
        emb_enc2 = model.encoder_2(x_batch)

        all_embeddings_enc1_list.append(emb_enc1.cpu())
        all_embeddings_enc2_list.append(emb_enc2.cpu())

        batch_count += 1
        if batch_count % 2 == 0:
            print(f'  Processed {batch_count} batches...')

# Concatenate all embeddings
all_embeddings_enc1 = torch.concat(all_embeddings_enc1_list, dim=0)
all_embeddings_enc2 = torch.concat(all_embeddings_enc2_list, dim=0)

# Prepare special embeddings to insert at the beginning
# For encoder 1: targets (32), generated_1 (32), generated_2 (32), samegal (32) = 128 total
# For encoder 2: targets (32), generated_1 (32), generated_2 (32), sameins_1_all5 (32*5=160), sameins_2_all5 (32*5=160) = 416 total
special_emb_enc1 = torch.concat([
    embedding_target_enc1[:32].cpu(),
    embedding_generated_samples_1_enc1[:32].cpu(),
    embedding_generated_samples_2_enc1[:32].cpu(),
    embedding_samegal[:32].cpu()
], dim=0)

special_emb_enc2 = torch.concat([
    embedding_target_enc2[:32].cpu(),
    embedding_generated_samples_1_enc2[:32].cpu(),
    embedding_generated_samples_2_enc2[:32].cpu(),
    embedding_sameins_1_all5.cpu(),
    embedding_sameins_2_all5.cpu()
], dim=0)

num_special_enc1 = special_emb_enc1.shape[0]  # Should be 128
num_special_enc2 = special_emb_enc2.shape[0]  # Should be 416
num_regular_needed_enc1 = num_examples_umap - num_special_enc1  # 8192 - 128 = 8064
num_regular_needed_enc2 = num_examples_umap - num_special_enc2  # 8192 - 416 = 7776

# Take enough regular examples to fill the rest
# For encoder 1: we need num_regular_needed_enc1 regular examples
# For encoder 2: we need num_regular_needed_enc2 regular examples
all_embeddings_enc1_regular = all_embeddings_enc1[:num_examples_umap]
all_embeddings_enc2_regular = all_embeddings_enc2[:num_examples_umap]

# Replace the first num_special elements with our special embeddings
# and keep the rest (num_regular_needed elements) from the regular embeddings
all_embeddings_enc1 = torch.concat([
    special_emb_enc1,
    all_embeddings_enc1_regular[num_special_enc1:num_special_enc1+num_regular_needed_enc1]
], dim=0)
all_embeddings_enc2 = torch.concat([
    special_emb_enc2,
    all_embeddings_enc2_regular[num_special_enc2:num_special_enc2+num_regular_needed_enc2]
], dim=0)

# Flatten embeddings for UMAP
all_embeddings_enc1_flat = all_embeddings_enc1.flatten(start_dim=1).numpy()
all_embeddings_enc2_flat = all_embeddings_enc2.flatten(start_dim=1).numpy()

print(f'\nComputing UMAP for Encoder 1 (Physics)...')
umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
}
reducer_1 = umap.UMAP(**umap_params)
umap_enc1 = reducer_1.fit_transform(all_embeddings_enc1_flat)

print('Computing UMAP for Encoder 2 (Instrument)...')
reducer_2 = umap.UMAP(**umap_params)
umap_enc2 = reducer_2.fit_transform(all_embeddings_enc2_flat)

# Now plot for 5 examples: target in both spaces, two generated samples in both spaces,
# same gal in physics space (enc1), and two same_ins in instrument space (enc2)
# Use one color for each example and the signs 's' '^' 'v' 'x' 'o' '*'

print('\nCreating visualization for 5 examples...')
num_examples_to_plot = 5

# Indices:
# 0-31: targets (x_1[:32])
# 32-63: generated samples 1
# 64-95: generated samples 2
# 96-127: samegal (for enc1 only)
# For enc2:
# 96-255: sameins_1_all5 (32 examples × 5 instruments, organized as: example i has instruments 0-4 at indices 96+i*5 to 96+i*5+4)
# 256-415: sameins_2_all5 (same structure: example i has instruments 0-4 at indices 256+i*5 to 256+i*5+4)

# For each of the 5 examples, we'll plot:
# - Target at index i (0-4) in both spaces
# - Generated sample 1 at index 32+i in both spaces
# - Generated sample 2 at index 64+i in both spaces
# - Samegal at index 96+i in enc1 space only
# - Sameins_1: 5 crosses at indices 96+i*5+j (j=0-4) in enc2 space
# - Sameins_2: 5 crosses at indices 256+i*5+j (j=0-4) in enc2 space

colors = ['red', 'blue', 'green', 'orange', 'purple']
marker_sizes = [150, 150, 150, 150, 150]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Encoder 1 plot (Physics)
ax1 = axes[0]
# Plot all points in light gray
ax1.scatter(umap_enc1[:, 0], umap_enc1[:, 1], s=5, c='lightgray', alpha=0.3, label='All examples')

# Plot 5 examples
for i in range(num_examples_to_plot):
    color = colors[i]
    # Target at index i
    ax1.scatter(umap_enc1[i, 0], umap_enc1[i, 1], s=marker_sizes[i],
                marker='s', c=color, edgecolors='black', linewidths=2,
                label=f'Target {i+1}' if i == 0 else None, alpha=0.8)
    # Generated sample 1 at index 32+i
    ax1.scatter(umap_enc1[32+i, 0], umap_enc1[32+i, 1], s=marker_sizes[i],
                marker='^', c=color, edgecolors='black', linewidths=2,
                label=f'Sample 1 {i+1}' if i == 0 else None, alpha=0.8)
    # Generated sample 2 at index 64+i
    ax1.scatter(umap_enc1[64+i, 0], umap_enc1[64+i, 1], s=marker_sizes[i],
                marker='v', c=color, edgecolors='black', linewidths=2,
                label=f'Sample 2 {i+1}' if i == 0 else None, alpha=0.8)
    # Samegal at index 96+i
    ax1.scatter(umap_enc1[96+i, 0], umap_enc1[96+i, 1], s=marker_sizes[i],
                marker='x', c=color, edgecolors='black', linewidths=2,
                label=f'SameGal {i+1}' if i == 0 else None, alpha=0.8)

ax1.set_title('Encoder 1 (Physics) - UMAP Visualization', fontsize=14)
ax1.set_xlabel('UMAP Component 1', fontsize=12)
ax1.set_ylabel('UMAP Component 2', fontsize=12)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# Encoder 2 plot (Instrument)
ax2 = axes[1]
# Plot all points in light gray
ax2.scatter(umap_enc2[:, 0], umap_enc2[:, 1], s=5, c='lightgray', alpha=0.3, label='All examples')

# Plot 5 examples
for i in range(num_examples_to_plot):
    color = colors[i]
    # Target at index i
    ax2.scatter(umap_enc2[i, 0], umap_enc2[i, 1], s=marker_sizes[i],
                marker='s', c=color, edgecolors='black', linewidths=2,
                label=f'Target {i+1}' if i == 0 else None, alpha=0.8)
    # Generated sample 1 at index 32+i
    ax2.scatter(umap_enc2[32+i, 0], umap_enc2[32+i, 1], s=marker_sizes[i],
                marker='^', c=color, edgecolors='black', linewidths=2,
                label=f'Sample 1 {i+1}' if i == 0 else None, alpha=0.8)
    # Generated sample 2 at index 64+i
    ax2.scatter(umap_enc2[64+i, 0], umap_enc2[64+i, 1], s=marker_sizes[i],
                marker='v', c=color, edgecolors='black', linewidths=2,
                label=f'Sample 2 {i+1}' if i == 0 else None, alpha=0.8)
    # Sameins_1: all 5 instruments as crosses
    for j in range(5):
        idx = 96 + i * 5 + j
        ax2.scatter(umap_enc2[idx, 0], umap_enc2[idx, 1], s=marker_sizes[i]//2,
                    marker='x', c=color, edgecolors='black', linewidths=1.5,
                    label=f'SameIns 1 (all 5)' if i == 0 and j == 0 else None, alpha=0.7)
    # Sameins_2: all 5 instruments as empty circles
    for j in range(5):
        idx = 256 + i * 5 + j
        ax2.scatter(umap_enc2[idx, 0], umap_enc2[idx, 1], s=marker_sizes[i]//2,
                    marker='o', facecolors='none', edgecolors=color, linewidths=1.5,
                    label=f'SameIns 2 (all 5)' if i == 0 and j == 0 else None, alpha=0.7)

ax2.set_title('Encoder 2 (Instrument) - UMAP Visualization', fontsize=14)
ax2.set_xlabel('UMAP Component 1', fontsize=12)
ax2.set_ylabel('UMAP Component 2', fontsize=12)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Re-encoding UMAP Visualization: 5 Examples with Targets, Two Generated Samples (different same_ins), and Condition Images',
             fontsize=16, y=1.02)
plt.tight_layout()

# Save figure
figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / f're_encoding_umap_5examples_{mode_suffix}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'\nUMAP visualization saved to: {output_path}')
print('Visualization complete!')
