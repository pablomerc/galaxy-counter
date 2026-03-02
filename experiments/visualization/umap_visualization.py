"""
Minimal script to generate UMAP visualization for double-encoder model.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from double_train_fm import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader
from data import HSCLegacyDataset
import umap
import matplotlib.pyplot as plt
import numpy as np

# Configuration
checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wdbsh3rc/checkpoints/latest-step=step=63000.ckpt'
hdf5_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5'
idx_list = list(range(95_000, 97_048))
batch_size = 2048

# UMAP parameters
umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
}

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

# Load model
print(f"\nLoading model from: {checkpoint_path}")
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

# Load data
print(f"\nLoading dataset...")
dataset = HSCLegacyDataset(hdf5_path=hdf5_path, idx_list=idx_list)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
batch = next(iter(train_loader))

# Move images to device
hsc_images = batch[0].to(device)
legacy_images = batch[1].to(device)

# Encode images
print("\nEncoding images...")
with torch.no_grad():
    hsc_embeddings_1 = model.encoder_1(hsc_images)
    legacy_embeddings_1 = model.encoder_1(legacy_images)
    hsc_embeddings_2 = model.encoder_2(hsc_images)
    legacy_embeddings_2 = model.encoder_2(legacy_images)

# Flatten embeddings
all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0).flatten(start_dim=1)
all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0).flatten(start_dim=1)

num_hsc = hsc_embeddings_1.shape[0]
print(f"Encoded {num_hsc} HSC and {num_hsc} Legacy images")

# Compute UMAP
print("\nComputing UMAP for Encoder 1 (Physics)...")
reducer_1 = umap.UMAP(**umap_params)
embedding_1 = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())
hsc_embedding_1 = embedding_1[:num_hsc]
legacy_embedding_1 = embedding_1[num_hsc:]

print("Computing UMAP for Encoder 2 (Instrument)...")
reducer_2 = umap.UMAP(**umap_params)
embedding_2 = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())
hsc_embedding_2 = embedding_2[:num_hsc]
legacy_embedding_2 = embedding_2[num_hsc:]

# Create visualization
print("\nCreating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Encoder 1 plot
ax1.scatter(hsc_embedding_1[:, 0], hsc_embedding_1[:, 1], s=5, label='HSC', alpha=0.6, c='blue')
ax1.scatter(legacy_embedding_1[:, 0], legacy_embedding_1[:, 1], s=5, label='Legacy', alpha=0.6, c='orange')
ax1.set_title('Encoder 1 (Same Galaxy)\nPhysics Latent Space (UMAP)')
ax1.set_xlabel('UMAP Component 1')
ax1.set_ylabel('UMAP Component 2')
ax1.legend()
ax1.grid(True)

# Encoder 2 plot
ax2.scatter(hsc_embedding_2[:, 0], hsc_embedding_2[:, 1], s=5, label='HSC', alpha=0.6, c='blue')
ax2.scatter(legacy_embedding_2[:, 0], legacy_embedding_2[:, 1], s=5, label='Legacy', alpha=0.6, c='orange')
ax2.set_title('Encoder 2 (Same Instrument)\nInstrument Latent Space (UMAP)')
ax2.set_xlabel('UMAP Component 1')
ax2.set_ylabel('UMAP Component 2')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Save figure
figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / 'umap_minimal.png'
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nUMAP visualization saved to: {output_path}")
