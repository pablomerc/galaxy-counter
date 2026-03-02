"""
Generate reconstruction plots from pretrained neighbors model.
Loads pretrained model, processes validation batch, and saves results for quick replotting.
"""

import sys
from pathlib import Path

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from double_train_fm_neighbors import ConditionalFlowMatchingModule
from neighbors import NeighborsPrecomputedDataset, simple_collate

# ===== Configuration =====

# Checkpoint path (modify as needed)
CHECKPOINT_PATH = "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"

# Validation data path
VAL_DATA_PATH = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/val_neighbors.vds"

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'reconstruction_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of examples to process (tune this later)
NUM_EXAMPLES = 128

# Number of samples to generate per example
NUM_SAMPLES = 5

# Colors for HSC/Legacy (match downstream_evaluation/final/aion_vs_ours_all.py)
COLOR_HSC = '#e8c4a0'
COLOR_LEGACY = '#8eb8e8'

# ===== Helper Functions =====

def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> torch.Tensor:
    """
    Scale a (3,H,W) tensor to (H,W,3) in [0,1] using fixed per-channel vmin/vmax.

    Args:
        x_chw: Image tensor (3, H, W) or (C, H, W) where C >= 3
        vmin: Minimum values per channel (3,)
        vmax: Maximum values per channel (3,)

    Returns:
        Scaled RGB image (H, W, 3) in [0, 1]
    """
    x = x_chw[:3]  # Take first 3 channels (RGB)
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    return y.permute(1, 2, 0)


def determine_device():
    """Determine the best available device (GPU or CPU)."""
    device = torch.device('cpu')
    if torch.cuda.is_available():
        # Try to find a working GPU
        for gpu_id in range(torch.cuda.device_count()):
            try:
                test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
                del test_tensor
                torch.cuda.empty_cache()
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                break
            except RuntimeError:
                print(f"GPU {gpu_id} is not available, trying next...")
                continue
        if device.type == 'cpu':
            print("No working GPU found, using CPU")
    else:
        print("CUDA not available, using CPU")
    return device


def load_model(checkpoint_path, device):
    """Load pretrained model from checkpoint."""
    print(f"\nLoading model from: {checkpoint_path}")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu'
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully ({total_params:,} parameters)")
    return model


def load_validation_batch(val_data_path, num_examples):
    """Load validation data batch."""
    print(f"\nLoading validation data from: {val_data_path}")

    # Load dataset
    dataset = NeighborsPrecomputedDataset(val_data_path)
    print(f"Dataset size: {len(dataset)} examples")

    # Get first num_examples
    batch = []
    for i in range(min(num_examples, len(dataset))):
        batch.append(dataset[i])

    # Collate into batch
    targets, samegals, sameins, masks, metadata = simple_collate(batch)

    print(f"\nBatch shapes:")
    print(f"  Targets: {targets.shape}")
    print(f"  SameGals: {samegals.shape}")
    print(f"  SameIns: {sameins.shape}")
    print(f"  Masks: {masks.shape}")
    print(f"  Metadata: {len(metadata)} entries")

    return targets, samegals, sameins, masks, metadata


def generate_reconstructions(model, targets, samegals, sameins, masks, device, num_samples=5):
    """
    Generate multiple reconstruction samples for each example.

    Returns:
        samples: (batch_size, num_samples, C, H, W)
        mean_samples: (batch_size, C, H, W)
    """
    batch_size = targets.shape[0]
    C, H, W = targets.shape[1:]

    # Move to device
    targets = targets.to(device)
    samegals = samegals.to(device)
    sameins = sameins.to(device)
    masks = masks.to(device)

    print(f"\nGenerating {num_samples} reconstruction samples per example...")

    # Store all samples
    all_samples = []

    for i in range(batch_size):
        # Get single example
        samegal_i = samegals[i:i+1]  # (1, C, H, W)
        sameins_i = sameins[i:i+1]   # (1, k, C, H, W)
        masks_i = masks[i:i+1]       # (1, k)

        # Repeat for multiple samples
        samegal_repeated = samegal_i.repeat(num_samples, 1, 1, 1)  # (num_samples, C, H, W)
        sameins_repeated = sameins_i.repeat(num_samples, 1, 1, 1, 1)  # (num_samples, k, C, H, W)
        masks_repeated = masks_i.repeat(num_samples, 1)  # (num_samples, k)

        # Generate samples
        samples_i = model.sample(samegal_repeated, sameins_repeated, masks=masks_repeated)
        all_samples.append(samples_i.unsqueeze(0))  # (1, num_samples, C, H, W)

        print(f"  Generated samples for example {i+1}/{batch_size}")

    # Stack all samples: (batch_size, num_samples, C, H, W)
    samples = torch.cat(all_samples, dim=0)

    # Compute mean samples: (batch_size, C, H, W)
    mean_samples = samples.mean(dim=1)

    return samples, mean_samples


def create_reconstruction_plot(targets, samegals, sameins, samples, mean_samples, metadata, output_path):
    """
    Create reconstruction plot similar to on_validation_epoch_end.

    Layout: SameGal | SameIns (1st) | Target | Sample1 | Sample2 | ... | SampleN | Mean
    """
    batch_size, num_samples, C, H, W = samples.shape
    num_cols = 3 + num_samples + 1  # samegal + sameins_first + target + samples + mean

    fig, axes = plt.subplots(
        batch_size, num_cols,
        figsize=(2 * num_cols, 2 * batch_size),
        squeeze=False,
    )

    col_titles = ["SameGal", "SameIns (1st)", "Target"] + [f"Sample {j+1}" for j in range(num_samples)] + ["Mean"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for i in range(batch_size):
        # Extract data for this row
        target = targets[i]
        samegal = samegals[i]
        sameins_first = sameins[i, 0]  # First same-instrument neighbor

        # Compute per-channel vmin/vmax from the TARGET for this row
        target_chw = target[:3]  # (3, H, W)
        vmin = target_chw.amin(dim=(1, 2))  # (3,)
        vmax = target_chw.amax(dim=(1, 2))  # (3,)

        # Plot SameGal
        samegal_vis = _row_scale_rgb(samegal[:3], vmin, vmax).detach().cpu().numpy()
        axes[i, 0].imshow(samegal_vis)
        axes[i, 0].axis('off')

        # Add row label with survey information
        survey = metadata[i].get('anchor_survey', 'unknown')
        survey_color = COLOR_HSC if survey == 'hsc' else COLOR_LEGACY
        axes[i, 0].text(-0.1, 0.5, f'{survey.upper()}\n[{i}]',
                       transform=axes[i, 0].transAxes,
                       ha='right', va='center', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round', facecolor=survey_color, alpha=0.5))

        # Plot SameIns (first)
        sameins_first_vis = _row_scale_rgb(sameins_first[:3], vmin, vmax).detach().cpu().numpy()
        axes[i, 1].imshow(sameins_first_vis)
        axes[i, 1].axis('off')

        # Plot Target
        target_vis = _row_scale_rgb(target[:3], vmin, vmax).detach().cpu().numpy()
        axes[i, 2].imshow(target_vis)
        axes[i, 2].axis('off')

        # Plot Samples
        for j in range(num_samples):
            sample_vis = _row_scale_rgb(samples[i, j, :3], vmin, vmax).detach().cpu().numpy()
            axes[i, 3 + j].imshow(sample_vis)
            axes[i, 3 + j].axis('off')

        # Plot Mean
        mean_vis = _row_scale_rgb(mean_samples[i, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, -1].imshow(mean_vis)
        axes[i, -1].axis('off')

    plt.suptitle('Reconstruction Samples from Pretrained Model', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nReconstruction plot saved to: {output_path}")


def save_data_for_replotting(targets, samegals, sameins, masks, samples, mean_samples, metadata, output_path):
    """
    Save all data to HDF5 file for quick replotting.

    Stores:
    - targets: (batch_size, C, H, W)
    - samegals: (batch_size, C, H, W)
    - sameins: (batch_size, k, C, H, W) - ALL instrument pairs
    - masks: (batch_size, k)
    - samples: (batch_size, num_samples, C, H, W)
    - mean_samples: (batch_size, C, H, W)
    - metadata: anchor_survey for each example
    """
    print(f"\nSaving data for replotting to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Save image data
        f.create_dataset('targets', data=targets.cpu().numpy(), compression='gzip')
        f.create_dataset('samegals', data=samegals.cpu().numpy(), compression='gzip')
        f.create_dataset('sameins', data=sameins.cpu().numpy(), compression='gzip')
        f.create_dataset('masks', data=masks.cpu().numpy(), compression='gzip')
        f.create_dataset('samples', data=samples.cpu().numpy(), compression='gzip')
        f.create_dataset('mean_samples', data=mean_samples.cpu().numpy(), compression='gzip')

        # Save metadata
        surveys = [m.get('anchor_survey', 'unknown') for m in metadata]
        f.create_dataset('anchor_surveys', data=np.array(surveys, dtype='S10'))

        indices = [m.get('idx', -1) for m in metadata]
        f.create_dataset('indices', data=np.array(indices, dtype=np.int64))

        num_same = [m.get('num_same_instrument', -1) for m in metadata]
        f.create_dataset('num_same_instrument', data=np.array(num_same, dtype=np.int64))

        # Save shape information
        f.attrs['batch_size'] = targets.shape[0]
        f.attrs['num_samples'] = samples.shape[1]
        f.attrs['image_channels'] = targets.shape[1]
        f.attrs['image_height'] = targets.shape[2]
        f.attrs['image_width'] = targets.shape[3]
        f.attrs['num_neighbors'] = sameins.shape[1]

    print(f"Data saved successfully")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def print_summary(targets, samples, mean_samples, metadata):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("RECONSTRUCTION SUMMARY")
    print("="*60)

    # Move all tensors to CPU for statistics computation
    targets = targets.cpu()
    samples = samples.cpu()
    mean_samples = mean_samples.cpu()

    # Compute MSE for each example
    mse_samples = ((targets.unsqueeze(1) - samples) ** 2).mean(dim=(2, 3, 4))  # (batch_size, num_samples)
    mse_mean = ((targets - mean_samples) ** 2).mean(dim=(1, 2, 3))  # (batch_size,)

    for i in range(len(targets)):
        survey = metadata[i].get('anchor_survey', 'unknown').upper()
        idx = metadata[i].get('idx', -1)
        num_neighbors = metadata[i].get('num_same_instrument', -1)

        print(f"\nExample {i} ({survey} idx={idx}, neighbors={num_neighbors}):")
        print(f"  MSE per sample: {mse_samples[i].cpu().numpy()}")
        print(f"  MSE mean: {mse_mean[i].item():.6f}")
        print(f"  Avg MSE: {mse_samples[i].mean().item():.6f} ± {mse_samples[i].std().item():.6f}")

    # Overall statistics
    print("\n" + "-"*60)
    print("Overall Statistics:")
    print(f"  Avg MSE (all samples): {mse_samples.mean().item():.6f}")
    print(f"  Avg MSE (mean samples): {mse_mean.mean().item():.6f}")

    # By survey
    hsc_mask = torch.tensor([m.get('anchor_survey', '') == 'hsc' for m in metadata])
    legacy_mask = torch.tensor([m.get('anchor_survey', '') == 'legacy' for m in metadata])

    if hsc_mask.any():
        print(f"  Avg MSE (HSC): {mse_mean[hsc_mask].mean().item():.6f}")
    if legacy_mask.any():
        print(f"  Avg MSE (Legacy): {mse_mean[legacy_mask].mean().item():.6f}")

    print("="*60)


# ===== Main Script =====

def main():
    print("="*60)
    print("RECONSTRUCTION FROM PRETRAINED NEIGHBORS MODEL")
    print("="*60)

    # Setup
    device = determine_device()

    # Load model
    model = load_model(CHECKPOINT_PATH, device)

    # Load validation batch
    targets, samegals, sameins, masks, metadata = load_validation_batch(
        VAL_DATA_PATH, NUM_EXAMPLES
    )

    # Generate reconstructions
    samples, mean_samples = generate_reconstructions(
        model, targets, samegals, sameins, masks, device, num_samples=NUM_SAMPLES
    )

    # Print summary statistics
    print_summary(targets, samples, mean_samples, metadata)

    # Create reconstruction plot
    plot_path = OUTPUT_DIR / 'reconstruction_plot.png'
    create_reconstruction_plot(
        targets.cpu(), samegals.cpu(), sameins.cpu(),
        samples.cpu(), mean_samples.cpu(), metadata, plot_path
    )

    # Save data for replotting
    data_path = OUTPUT_DIR / 'reconstruction_data.h5'
    save_data_for_replotting(
        targets, samegals, sameins, masks, samples, mean_samples, metadata, data_path
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Plot: {plot_path.name}")
    print(f"  - Data: {data_path.name}")
    print("\nYou can now use a second script to quickly replot specific examples")
    print("by reading from the saved data file.")


if __name__ == "__main__":
    main()
