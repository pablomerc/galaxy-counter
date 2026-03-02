import h5py
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# --- PATH SETUP ---
current_path = os.path.abspath(__file__)
# Go up 4 levels to reach root
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_path))))
sys.path.insert(0, root_dir)

from galaxy_images.image_preprocessing import preprocess_image_v2
from galaxy_images.galaxy_model.data import zoom_legacy_image
from galaxy_images.galaxy_model.double_train_fm import ConditionalFlowMatchingModule

# --- CONFIG ---
LENSES_PATH = '/data/vision/billf/scratch/pablomer/data/lens_matches.h5'
CHECKPOINT_PATH = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt'

NORM_DICT = {
    'hsc': [0.022, 0.05],
    'legacy_zoom': [0.045, 0.078],
}

# --- HELPER FUNCTIONS ---
def preprocess_raw_image(image, survey="hsc"):
    """ Preprocess raw images """
    # Convert to tensor if numpy
    if not torch.is_tensor(image):
        image = torch.from_numpy(image).float()

    # 1. Standard pipeline (Crop, Clamp, Rescale, RangeCompress)
    image = preprocess_image_v2(image, crop_size=48, survey=survey)

    # 2. Specific Normalization for this model
    if survey == "legacy":
        image = zoom_legacy_image(image)
        image = (image - NORM_DICT['legacy_zoom'][0]) / NORM_DICT['legacy_zoom'][1]
    else:
        image = (image - NORM_DICT['hsc'][0]) / NORM_DICT['hsc'][1]
    return image

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# --- DATASET CLASS ---
class LensMatchesDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        self.hsc_dset = None
        self.leg_dset = None

        # Open briefly just to get length
        with h5py.File(h5_path, 'r') as f:
            self.length = f['hsc_images'].shape[0]

    def __len__(self):
        return self.length

    def _open_file(self):
        """Helper to open file on first access (lazy)"""
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            self.hsc_dset = self.file['hsc_images']
            self.leg_dset = self.file['legacy_images']

    def __getitem__(self, idx):
        self._open_file()

        # 1. Load Main Pair (HSC + Legacy)
        hsc_img = self.hsc_dset[idx]
        leg_img = self.leg_dset[idx]

        hsc_tensor = preprocess_raw_image(hsc_img, survey="hsc")
        leg_tensor = preprocess_raw_image(leg_img, survey="legacy")

        # 2. Load 5 Random HSC Images ("same_ins_batch")
        random_indices = torch.randint(0, self.length, (5,))

        random_images = []
        for rand_idx in random_indices:
            # Load raw random image
            r_img = self.hsc_dset[rand_idx]
            # Preprocess
            r_tensor = preprocess_raw_image(r_img, survey="hsc")
            random_images.append(r_tensor)

        # Stack them: [5, C, H, W]
        same_ins_batch = torch.stack(random_images)

        return hsc_tensor[:4, :, :], leg_tensor[:4, :, :], same_ins_batch[:, :4, :, :]





def save_batch_visualization(legacy, hsc, samples, batch_idx, save_dir="batch_plots"):
    """
    Saves a grid of images: Legacy | HSC (Target) | Generated Sample.
    Uses row-wise scaling based on the HSC (Target) min/max per channel.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    batch_size = legacy.shape[0]
    num_cols = 3  # Legacy, HSC, Sample

    # Setup Figure
    fig, axes = plt.subplots(
        batch_size, num_cols,
        figsize=(3 * num_cols, 3 * batch_size),
        squeeze=False
    )

    # Set Column Titles
    col_titles = ["Legacy (Input)", "HSC (Target)", "Prediction"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12)

    def _row_scale_rgb(x_chw, vmin, vmax):
        """
        Scale a (C,H,W) tensor to (H,W,3) in [0,1] using fixed vmin/vmax.
        Only uses first 3 channels for RGB visualization.
        """
        # Take first 3 channels
        x = x_chw[:3]

        # Reshape vmin/vmax for broadcasting (3, 1, 1)
        vmin_t = vmin.view(3, 1, 1)
        vmax_t = vmax.view(3, 1, 1)

        # Scale
        y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
        y = y.clamp(0, 1)

        # Permute to HWC for matplotlib
        return y.permute(1, 2, 0).cpu().numpy()

    # Loop through items in the batch
    for i in range(batch_size):
        # 1. Get Tensors for this row
        leg_tens = legacy[i].detach()   # [C, H, W]
        hsc_tens = hsc[i].detach()      # [C, H, W]
        sam_tens = samples[i].detach()  # [C, H, W]

        # We use the first 3 channels to determine scaling for RGB plot
        # Target (HSC)
        vmin_target = hsc_tens[:3].amin(dim=(1, 2))
        vmax_target = hsc_tens[:3].amax(dim=(1, 2))

        # Legacy
        vmin_legacy = leg_tens[:3].amin(dim=(1, 2))
        vmax_legacy = leg_tens[:3].amax(dim=(1, 2))

        # Sample
        vmin_sample = sam_tens[:3].amin(dim=(1, 2))
        vmax_sample = sam_tens[:3].amax(dim=(1, 2))

        # 3. Apply Scaling to all images in this row using Target's limits
        leg_vis = _row_scale_rgb(leg_tens, vmin_legacy, vmax_legacy)
        hsc_vis = _row_scale_rgb(hsc_tens, vmin_target, vmax_target)
        sam_vis = _row_scale_rgb(sam_tens, vmin_sample, vmax_sample)

        # 4. Plot
        axes[i, 0].imshow(leg_vis)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hsc_vis)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(sam_vis)
        axes[i, 2].axis("off")

    # Save
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"batch_{batch_idx:04d}_results.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")





# ============================================================================
# Main Script
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # 1. Setup Dataset & Loader
    dataset = LensMatchesDataset(LENSES_PATH)

    # DataLoader handles batching automatically
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    print("Starting processing...")

    # # 2. Iterate
    # for batch_idx, (hsc, legacy, same_ins_batch) in enumerate(loader):
    #     hsc = hsc.to(device)
    #     legacy = legacy.to(device)
    #     same_ins_batch = same_ins_batch.to(device)

    #     print(f"\n--- Batch {batch_idx} ---")
    #     print(f"HSC Image:         {hsc.shape}")            # [32, 5, 48, 48]
    #     print(f"Legacy Image:      {legacy.shape}")         # [32, 4, 48, 48]
    #     print(f"Same Ins Batch:    {same_ins_batch.shape}") # [32, 5, 5, 48, 48]

    #     # Dimensions explained:
    #     # [Batch Size, Num_Random_Samples, Channels, Height, Width]

    #     if batch_idx == 0:
    #         break

    # Let's load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = ConditionalFlowMatchingModule.load_from_checkpoint(CHECKPOINT_PATH)
    model.to(device)
    model.eval()

    # ... previous setup code ...

    print("Model loaded successfully")

    # Create a folder for plots
    PLOT_DIR = "inference_plots"

    for batch_idx, (hsc, legacy, same_ins_batch) in enumerate(loader):
        hsc = hsc.to(device)
        legacy = legacy.to(device)
        same_ins_batch = same_ins_batch.to(device)

        print(f"\n--- Batch {batch_idx} ---")

        # Generate samples
        # Assuming model.sample returns [Batch, Channels, Height, Width]
        # If it returns [Batch, Num_Samples, C, H, W], we take the first sample: samples[:, 0]
        with torch.no_grad():
            samples = model.sample(legacy, same_ins_batch)

        print(f"Samples shape: {samples.shape}")

        # Visualization
        # We pass hsc as the target to determine scaling
        save_batch_visualization(legacy, hsc, samples, batch_idx, save_dir=PLOT_DIR)

        # Optional: Stop after a few batches to save time while testing
        if batch_idx >= 2:
            break
