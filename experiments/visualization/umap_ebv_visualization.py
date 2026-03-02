"""
Script to generate UMAP visualization colored by various metadata parameters for double-encoder model.
Loads 8192 examples and creates two large combined figures: one with ordered indices and one with randomized indices.
Each figure contains all parameters arranged in a grid (one row per parameter, 4 columns per row).
"""

import sys
import importlib.util
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from data import HSCLegacyDataset
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import h5py
import random

# Configuration

hdf5_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5'
metadata_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_metadata_8192.h5'
num_examples = 8192
idx_list_ordered = list(range(0, num_examples))  # Ordered indices
idx_list_random = list(range(0, num_examples))
random.seed(42)
random.shuffle(idx_list_random)  # Randomized indices
batch_size = 512

# Parameters to visualize - all parameters from the full list
params_to_visualize = ['EBV', 'FLUX_G', 'FLUX_I', 'FLUX_R', 'FLUX_W1',
                       'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_Z', 'SHAPE_E1',
                       'SHAPE_E2', 'SHAPE_R', 'a_g', 'a_i', 'a_r', 'a_y', 'a_z', 'g_cmodel_mag', 'g_cmodel_magerr', 'g_extendedness_value', 'g_sdssshape_psf_shape11',
                       'g_sdssshape_psf_shape12', 'g_sdssshape_psf_shape22', 'g_sdssshape_shape11', 'g_sdssshape_shape12',
                       'g_sdssshape_shape22', 'i_cmodel_mag', 'i_cmodel_magerr', 'i_extendedness_value',
                       'i_sdssshape_psf_shape11', 'i_sdssshape_psf_shape12', 'i_sdssshape_psf_shape22', 'i_sdssshape_shape11',
                       'i_sdssshape_shape12', 'i_sdssshape_shape22', 'r_cmodel_mag',
                       'r_cmodel_magerr', 'r_extendedness_value', 'r_sdssshape_psf_shape11', 'r_sdssshape_psf_shape12',
                       'r_sdssshape_psf_shape22', 'r_sdssshape_shape11', 'r_sdssshape_shape12', 'r_sdssshape_shape22',
                       'y_cmodel_mag', 'y_cmodel_magerr', 'y_extendedness_value', 'y_sdssshape_psf_shape11',
                       'y_sdssshape_psf_shape12', 'y_sdssshape_psf_shape22', 'y_sdssshape_shape11', 'y_sdssshape_shape12',
                       'y_sdssshape_shape22', 'z_cmodel_mag', 'z_cmodel_magerr', 'z_extendedness_value', 'z_sdssshape_psf_shape11',
                       'z_sdssshape_psf_shape12', 'z_sdssshape_psf_shape22', 'z_sdssshape_shape11', 'z_sdssshape_shape12',
                       'z_sdssshape_shape22']

colorbar_scale = 'linear' # 'log' or 'linear'

model_type = 'attention_conditional' # 'attention_conditional' or 'class_conditional'

geom_suffix = '_geom'

try_random_order = False # True or False

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

# Import appropriate model class based on model_type
if model_type == 'attention_conditional':
    from double_train_fm import ConditionalFlowMatchingModule
    # checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/4o2v012z/checkpoints/latest-step=step=75000.ckpt'
    checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt' # z_dim = 16 + geom
elif model_type == 'class_conditional':
    # Handle hyphenated filename using importlib
    module_path = Path(__file__).parent.parent / 'double_train_fm_no-attn.py'
    spec = importlib.util.spec_from_file_location("double_train_fm_no_attn", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ConditionalFlowMatchingModule = module.ConditionalFlowMatchingModule
    # checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/p9tj82az/checkpoints/latest-step=step=75000.ckpt'
    checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/srj4opub/checkpoints/latest-step=step=75000.ckpt'
else:
    raise ValueError(f"Unknown model_type: {model_type}. Must be 'attention_conditional' or 'class_conditional'")

# Load model
print(f"\nLoading model from: {checkpoint_path}")
print(f"Model type: {model_type}")
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

# Load all metadata
print(f"\nLoading metadata from: {metadata_path}")
with h5py.File(metadata_path, 'r') as f:
    # Load all parameters
    metadata_dict = {}
    for param in params_to_visualize:
        if param in f:
            metadata_dict[param] = f[param][:].astype(np.float32)
            print(f"  Loaded {param}: shape={metadata_dict[param].shape}, range=[{metadata_dict[param].min():.4f}, {metadata_dict[param].max():.4f}]")
        else:
            print(f"  Warning: {param} not found in metadata file")

    # Check if there's an indices field to map metadata to dataset indices
    if 'indices' in f:
        metadata_indices = f['indices'][:]
        print(f"Found indices mapping: shape={metadata_indices.shape}")
        # Create a mapping from dataset index to metadata index
        idx_to_metadata_idx = {dataset_idx: meta_idx for meta_idx, dataset_idx in enumerate(metadata_indices)}
    else:
        print("No indices field found, assuming metadata is in order (0 to num_examples-1)")
        idx_to_metadata_idx = None


# Map indices to metadata values
def get_metadata_for_indices(idx_list, param_name):
    """Get metadata values for a list of dataset indices."""
    if param_name not in metadata_dict:
        raise ValueError(f"Parameter {param_name} not found in metadata")

    param_values_all = metadata_dict[param_name]

    if idx_to_metadata_idx is not None:
        # Map dataset indices to metadata indices
        metadata_idx_list = [idx_to_metadata_idx[idx] for idx in idx_list]
        return param_values_all[metadata_idx_list]
    else:
        # Assume metadata is in order
        return param_values_all[idx_list]

def compute_umap_embeddings():
    """Compute UMAP embeddings once using ordered indices."""
    print(f"\n{'='*60}")
    print("Computing UMAP embeddings (one-time computation)")
    print(f"{'='*60}")

    # Load data with ordered indices
    print(f"\nLoading dataset with {len(idx_list_ordered)} examples...")
    dataset = HSCLegacyDataset(hdf5_path=hdf5_path, idx_list=idx_list_ordered)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Encode images in batches
    print("\nEncoding images...")
    hsc_embeddings_1_list = []
    legacy_embeddings_1_list = []
    hsc_embeddings_2_list = []
    legacy_embeddings_2_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            hsc_images = batch[0].to(device)
            legacy_images = batch[1].to(device)

            hsc_emb_1 = model.encoder_1(hsc_images)
            legacy_emb_1 = model.encoder_1(legacy_images)
            hsc_emb_2 = model.encoder_2(hsc_images)
            legacy_emb_2 = model.encoder_2(legacy_images)

            hsc_embeddings_1_list.append(hsc_emb_1.cpu())
            legacy_embeddings_1_list.append(legacy_emb_1.cpu())
            hsc_embeddings_2_list.append(hsc_emb_2.cpu())
            legacy_embeddings_2_list.append(legacy_emb_2.cpu())

            if (batch_idx + 1) % 4 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Concatenate all embeddings
    hsc_embeddings_1 = torch.concat(hsc_embeddings_1_list, dim=0)
    legacy_embeddings_1 = torch.concat(legacy_embeddings_1_list, dim=0)
    hsc_embeddings_2 = torch.concat(hsc_embeddings_2_list, dim=0)
    legacy_embeddings_2 = torch.concat(legacy_embeddings_2_list, dim=0)

    # Flatten embeddings
    all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0).flatten(start_dim=1)
    all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0).flatten(start_dim=1)

    num_hsc = hsc_embeddings_1.shape[0]
    print(f"Encoded {num_hsc} HSC and {num_hsc} Legacy images")

    # Compute UMAP
    print("\nComputing UMAP for Encoder 1 (Physics)...")
    reducer_1 = umap.UMAP(**umap_params)
    embedding_1 = reducer_1.fit_transform(all_embeddings_1.numpy())
    hsc_embedding_1 = embedding_1[:num_hsc]
    legacy_embedding_1 = embedding_1[num_hsc:]

    print("Computing UMAP for Encoder 2 (Instrument)...")
    reducer_2 = umap.UMAP(**umap_params)
    embedding_2 = reducer_2.fit_transform(all_embeddings_2.numpy())
    hsc_embedding_2 = embedding_2[:num_hsc]
    legacy_embedding_2 = embedding_2[num_hsc:]

    return hsc_embedding_1, legacy_embedding_1, hsc_embedding_2, legacy_embedding_2

def create_large_combined_visualization(hsc_embedding_1, legacy_embedding_1, hsc_embedding_2, legacy_embedding_2,
                                        all_param_data, param_names, title_suffix, colorbar_scale='linear', model_type='attention_conditional', geom_suffix=''):
    """Create one large figure with all parameters arranged in a grid."""
    print(f"\nCreating large combined visualization - {title_suffix}...")
    print(f"  Number of parameters: {len(param_names)}")
    print(f"  Colorbar scale: {colorbar_scale}")
    print(f"  Model type: {model_type}")

    # Filter out parameters that don't have data
    valid_params = [p for p in param_names if p in all_param_data]
    num_params = len(valid_params)

    if num_params == 0:
        print("  No valid parameters to visualize!")
        return None

    print(f"  Valid parameters: {num_params}")

    # Create figure with num_params rows and 4 columns
    # Columns: Encoder 1 HSC, Encoder 1 Legacy, Encoder 2 HSC, Encoder 2 Legacy
    fig, axes = plt.subplots(num_params, 4, figsize=(24, 6 * num_params))

    # Handle case where there's only one parameter (axes will be 1D)
    if num_params == 1:
        axes = axes.reshape(1, -1)

    for row_idx, param_name in enumerate(valid_params):
        param_values = all_param_data[param_name]

        # Create norm based on colorbar_scale
        if colorbar_scale == 'log':
            # For log scale, take absolute value to handle negative values
            param_values_plot = np.abs(param_values)
            # Calculate percentiles from absolute values
            vmin_plot = np.percentile(param_values_plot, 5)
            vmax_plot = np.percentile(param_values_plot, 95)
            # Ensure minimum is positive and non-zero (LogNorm requires > 0)
            vmin_plot = max(vmin_plot, 1e-10)
            vmax_plot = max(vmax_plot, 1e-10)
            # Ensure vmax > vmin (LogNorm requires vmax > vmin)
            if vmax_plot <= vmin_plot:
                vmax_plot = vmin_plot * 1.1  # Make vmax 10% larger than vmin
            norm = mcolors.LogNorm(vmin=vmin_plot, vmax=vmax_plot)
        else:
            # Calculate percentiles for robust color scaling
            vmin = np.percentile(param_values, 5)
            vmax = np.percentile(param_values, 95)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            param_values_plot = param_values

        # Encoder 1 - HSC
        scatter1_hsc = axes[row_idx, 0].scatter(hsc_embedding_1[:, 0], hsc_embedding_1[:, 1],
                                                s=10, c=param_values_plot, cmap='viridis', alpha=0.5,
                                                edgecolors='none', norm=norm)
        axes[row_idx, 0].set_title(f'{param_name}\nEncoder 1 (Physics) - HSC', fontsize=10)
        axes[row_idx, 0].set_xlabel('UMAP 1', fontsize=8)
        axes[row_idx, 0].set_ylabel('UMAP 2', fontsize=8)
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].tick_params(labelsize=7)
        plt.colorbar(scatter1_hsc, ax=axes[row_idx, 0], label=param_name, shrink=0.8)

        # Encoder 1 - Legacy
        scatter1_legacy = axes[row_idx, 1].scatter(legacy_embedding_1[:, 0], legacy_embedding_1[:, 1],
                                                   s=10, c=param_values_plot, cmap='viridis', alpha=0.5,
                                                   edgecolors='none', norm=norm)
        axes[row_idx, 1].set_title(f'{param_name}\nEncoder 1 (Physics) - Legacy', fontsize=10)
        axes[row_idx, 1].set_xlabel('UMAP 1', fontsize=8)
        axes[row_idx, 1].set_ylabel('UMAP 2', fontsize=8)
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].tick_params(labelsize=7)
        plt.colorbar(scatter1_legacy, ax=axes[row_idx, 1], label=param_name, shrink=0.8)

        # Encoder 2 - HSC
        scatter2_hsc = axes[row_idx, 2].scatter(hsc_embedding_2[:, 0], hsc_embedding_2[:, 1],
                                                s=10, c=param_values_plot, cmap='viridis', alpha=0.5,
                                                edgecolors='none', norm=norm)
        axes[row_idx, 2].set_title(f'{param_name}\nEncoder 2 (Instrument) - HSC', fontsize=10)
        axes[row_idx, 2].set_xlabel('UMAP 1', fontsize=8)
        axes[row_idx, 2].set_ylabel('UMAP 2', fontsize=8)
        axes[row_idx, 2].grid(True, alpha=0.3)
        axes[row_idx, 2].tick_params(labelsize=7)
        plt.colorbar(scatter2_hsc, ax=axes[row_idx, 2], label=param_name, shrink=0.8)

        # Encoder 2 - Legacy
        scatter2_legacy = axes[row_idx, 3].scatter(legacy_embedding_2[:, 0], legacy_embedding_2[:, 1],
                                                   s=10, c=param_values_plot, cmap='viridis', alpha=0.5,
                                                   edgecolors='none', norm=norm)
        axes[row_idx, 3].set_title(f'{param_name}\nEncoder 2 (Instrument) - Legacy', fontsize=10)
        axes[row_idx, 3].set_xlabel('UMAP 1', fontsize=8)
        axes[row_idx, 3].set_ylabel('UMAP 2', fontsize=8)
        axes[row_idx, 3].grid(True, alpha=0.3)
        axes[row_idx, 3].tick_params(labelsize=7)
        plt.colorbar(scatter2_legacy, ax=axes[row_idx, 3], label=param_name, shrink=0.8)

        if (row_idx + 1) % 10 == 0:
            print(f"  Processed {row_idx + 1}/{num_params} parameters...")

    # Create model type label for display
    model_type_label = 'Attention' if model_type == 'attention_conditional' else 'Class'
    plt.suptitle(f'UMAP Visualizations - {title_suffix} ({model_type_label} Conditioning)', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures/umap_correlations')
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_type_short = 'attn' if model_type == 'attention_conditional' else 'class'
    output_path = figures_dir / f'umap_all_params_8k_{title_suffix.lower().replace(" ", "_")}_{colorbar_scale}_{model_type_short}{geom_suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Large combined visualization saved to: {output_path}")
    return output_path

# Compute UMAP embeddings once (using ordered indices)
hsc_embedding_1, legacy_embedding_1, hsc_embedding_2, legacy_embedding_2 = compute_umap_embeddings()

# Process all parameters
print(f"\n{'='*60}")
print("Starting visualization generation")
print(f"{'='*60}")

# Collect all parameter data for ordered indices
print("\nCollecting metadata for all parameters (ordered indices)...")
all_param_data_ordered = {}
valid_params = []

for param_name in params_to_visualize:
    if param_name not in metadata_dict:
        print(f"  Skipping {param_name} - not found in metadata")
        continue

    valid_params.append(param_name)
    param_ordered = get_metadata_for_indices(idx_list_ordered, param_name)
    all_param_data_ordered[param_name] = param_ordered
    print(f"  Loaded {param_name}: min={param_ordered.min():.4f}, max={param_ordered.max():.4f}, "
          f"mean={param_ordered.mean():.4f}, std={param_ordered.std():.4f}")

print(f"\nLoaded {len(valid_params)} valid parameters")

# Create randomized data (shuffle each parameter independently) - only if try_random_order is True
all_param_data_randomized = {}
if try_random_order:
    print("\nCreating randomized parameter data...")
    np.random.seed(42)
    for param_name in valid_params:
        param_random = all_param_data_ordered[param_name].copy()
        np.random.shuffle(param_random)
        all_param_data_randomized[param_name] = param_random

# Create large combined figure for ordered indices
print(f"\n{'='*60}")
print("Creating large combined figure - Ordered Indices")
print(f"{'='*60}")
output_path_ordered = create_large_combined_visualization(
    hsc_embedding_1, legacy_embedding_1, hsc_embedding_2, legacy_embedding_2,
    all_param_data_ordered, valid_params, "Ordered Indices",
    colorbar_scale=colorbar_scale, model_type=model_type, geom_suffix=geom_suffix
)

# Create large combined figure for randomized indices - only if try_random_order is True
output_path_randomized = None
if try_random_order:
    print(f"\n{'='*60}")
    print("Creating large combined figure - Randomized Indices")
    print(f"{'='*60}")
    output_path_randomized = create_large_combined_visualization(
        hsc_embedding_1, legacy_embedding_1, hsc_embedding_2, legacy_embedding_2,
        all_param_data_randomized, valid_params, "Randomized Indices",
        colorbar_scale=colorbar_scale, model_type=model_type, geom_suffix=geom_suffix
    )

print(f"\n{'='*60}")
print("Summary - All visualizations completed!")
print(f"{'='*60}")
print(f"Output directory: /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures/umap_correlations/")
print(f"  - Ordered indices figure: {output_path_ordered}")
if try_random_order:
    print(f"  - Randomized indices figure: {output_path_randomized}")
else:
    print(f"  - Randomized indices figure: Skipped (try_random_order=False)")
