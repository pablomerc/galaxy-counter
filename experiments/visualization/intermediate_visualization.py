import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import timm
import math
from typing import Optional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from double_train_fm import ConditionalFlowMatchingModule
from data import HSCLegacyDatasetZoom

class ResNetEncoder(nn.Module):
    """
    ResNet18 encoder from timm that produces spatial feature maps for conditioning.
    Uses feature extraction to get intermediate spatial features for cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cross_attention_dim: int = 256,
        pretrained: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),  # Get features from layer2, layer3, layer4
        )

        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        self.proj = nn.Conv2d(512, cross_attention_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Conditioning image (B, C, H, W)
        Returns:
            Spatial embeddings (B, seq_len, cross_attention_dim) for cross-attention
        """
        features = self.backbone(x)
        feat = features[-1]  # (B, 512, H/32, W/32)
        feat = self.proj(feat)  # (B, cross_attention_dim, H', W')

        B, D, H, W = feat.shape
        feat = feat.view(B, D, H * W).permute(0, 2, 1)
        return feat

    def intermediate_states(self, x: torch.Tensor) -> list:
        features = self.backbone(x)
        for i, element in enumerate(features):
            print(f'Shape of features element {i} is {element.shape}')
        return features


def visualize_feat0(x: torch.Tensor, save_path: Optional[str] = None) -> None:
    """
    Visualize intermediate CNN feature maps.

    Parameters
    ----------
    x : torch.Tensor
        Feature tensor of shape (B, C, H, W).
    save_path : Optional[str]
        Path to save the figure. If None, saves to 'intermediate_features.png' in current directory.

    Returns
    -------
    None
    """
    _, C, H, W = x.shape
    image = x[0].detach().cpu()

    nrows = 4
    ncols = math.ceil(C / nrows)  # Ensure we have enough columns

    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

    for i, ax in enumerate(axes.flat):
        if i >= C:
            ax.axis("off")
            continue

        ax.imshow(image[i], cmap="viridis")
        ax.set_title(f"ch {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = "intermediate_features.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()

def visualize_feat0_double_encoder(x1: torch.Tensor, x2: torch.Tensor, save_path: Optional[str] = None) -> None:
    """
    Visualize intermediate CNN feature maps from two encoders in a single plot.

    Parameters
    ----------
    x1 : torch.Tensor
        Feature tensor from encoder_1 of shape (B, C1, H, W).
    x2 : torch.Tensor
        Feature tensor from encoder_2 of shape (B, C2, H, W).
    save_path : Optional[str]
        Path to save the figure. If None, saves to 'intermediate_features_double.png' in current directory.

    Returns
    -------
    None
    """
    _, C1, H1, W1 = x1.shape
    _, C2, H2, W2 = x2.shape

    image1 = x1[0].detach().cpu()
    image2 = x2[0].detach().cpu()

    # Compute global min and max across both feature tensors for consistent color scale
    vmin = min(image1.min().item(), image2.min().item())
    vmax = max(image1.max().item(), image2.max().item())

    nrows_per_encoder = 4
    ncols1 = math.ceil(C1 / nrows_per_encoder)
    ncols2 = math.ceil(C2 / nrows_per_encoder)

    # Use the maximum number of columns to align both sections
    ncols = max(ncols1, ncols2)

    # Total rows: encoder_1 rows + separator + encoder_2 rows
    total_rows = nrows_per_encoder + 1 + nrows_per_encoder

    fig, axes = plt.subplots(total_rows, ncols, figsize=(2*ncols, 2*total_rows))

    # Handle both 1D and 2D axes arrays
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    elif total_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot encoder_1 features (first section)
    for i in range(nrows_per_encoder * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        if i >= C1:
            ax.axis("off")
            continue

        ax.imshow(image1[i], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Encoder 1, ch {i}", fontsize=8)
        ax.axis("off")

    # Add horizontal separator (blank row for visual separation)
    separator_row = nrows_per_encoder
    for col in range(ncols):
        ax = axes[separator_row, col]
        ax.axis("off")
        # Set a light gray background to make the separator visible
        ax.set_facecolor('lightgray')

    # Plot encoder_2 features (second section)
    start_row = nrows_per_encoder + 1
    for i in range(nrows_per_encoder * ncols):
        row = start_row + (i // ncols)
        col = i % ncols
        ax = axes[row, col]

        if i >= C2:
            ax.axis("off")
            continue

        ax.imshow(image2[i], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Encoder 2, ch {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_path = "intermediate_features_double.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/twop8sfb/checkpoints/latest-step=step=47000.ckpt'

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

    dataset = HSCLegacyDatasetZoom(
        hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
        idx_list=list(range(95_000, 97_048)),
    )

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    batch = next(iter(train_loader))

    # Extract HSC and legacy images from batch tuple
    hsc_images = batch[0].to(device)  # (B, C, H, W)
    legacy_images = batch[1].to(device)  # (B, C, H, W)

    print(f'HSC images shape: {hsc_images.shape}')
    print(f'Legacy images shape: {legacy_images.shape}')


    features_1_hsc = model.encoder_1.intermediate_states(hsc_images)
    features_1_hsc = features_1_hsc[0]

    features_1_legacy = model.encoder_1.intermediate_states(legacy_images)
    features_1_legacy = features_1_legacy[0]

    features_2_hsc = model.encoder_2.intermediate_states(hsc_images)
    features_2_hsc = features_2_hsc [0]

    features_2_legacy = model.encoder_2.intermediate_states(legacy_images)
    features_2_legacy = features_2_legacy[0]

    diff_1 = features_1_hsc - features_1_legacy
    diff_2 = features_2_hsc - features_2_legacy

    visualize_feat0_double_encoder(diff_1, diff_2)
