import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage, HSCImage
import h5py

# --- AION LIBRARY BUG FIX ---
# The installed version of AION has a typo in 'rescale.py'.
# It calls '_reverse_zeropoint' but only 'reverse_zeropoint' exists.
# This patch redirects the missing method to the correct one.

try:
    from aion.codecs.preprocessing.image import RescaleToLegacySurvey

    # If the private method is missing, alias it to the public one
    if not hasattr(RescaleToLegacySurvey, '_reverse_zeropoint'):
        RescaleToLegacySurvey._reverse_zeropoint = RescaleToLegacySurvey.reverse_zeropoint
        print("Successfully patched RescaleToLegacySurvey._reverse_zeropoint bug.")
except ImportError:
    print("Could not patch AION: RescaleToLegacySurvey not found.")
# -----------------------------


# Another bug fix: I had to go to image.py and change the backward pass to use the correct method.
# in aion/codecs/preprocessing/image.py im not sure if this is okay or needs to be fixed?
# class RescaleToLegacySurvey:
    # """Formatter that rescales the images to have a fixed number of bands."""

    # def __init__(self):
    #     pass

    # def convert_zeropoint(self, zp: float) -> float:
    #     return 10.0 ** ((zp - 22.5) / 2.5)

    # def reverse_zeropoint(self, scale: float) -> float:
    #     return 22.5 - 2.5 * torch.log10(scale)

    # def forward(self, image, survey):
    #     zpscale = self.convert_zeropoint(27.0) if survey == "HSC" else 1.0
    #     image /= zpscale
    #     return image

    # def backward(self, image, survey):
    #     zpscale = self._reverse_zeropoint(27.0) if survey == "HSC" else 1.0
    #     image *= zpscale
    #     return image


# --- FIX IS HERE: Removed the square brackets ---
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

print('Device:', device) # Should now print "cuda", not "['cuda']"

model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
codec_manager = CodecManager(device=device)

h5_path = '/data/vision/billf/scratch/pablomer/data/test_neighbours_v2.h5'
N_EXAMPLES = 8

with h5py.File(h5_path, 'r') as f:
    legacy_tensor = f['images_legacy'][:N_EXAMPLES]
    hsc_tensor = f['images_hsc'][:N_EXAMPLES]


# This will now work correctly as well
legacy_tensor = torch.from_numpy(legacy_tensor).to(device)
hsc_tensor = torch.from_numpy(hsc_tensor).to(device)

print('Loaded legacy and hsc tensors')


legacy_tensor_copy = legacy_tensor.clone()
hsc_tensor_copy = hsc_tensor.clone()

image_leg = LegacySurveyImage(
    flux=legacy_tensor_copy,  # Shape: [batch, 4, height, width] for g,r,i,z bands
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

tokens = codec_manager.encode(image_leg)
embeddings = model.encode(tokens, num_encoder_tokens=600)

preds = model(
    codec_manager.encode(image_leg),
    target_modality=HSCImage,
)

logits = preds['tok_image_hsc']
token_indices = torch.argmax(logits, dim=-1)  # Shape: [Batch, Seq_Len]

token_key = HSCImage.token_key  # This is 'tok_image_hsc'
tokens_for_decoder = {token_key: token_indices}

decoded_hsc = codec_manager.decode(
    tokens=tokens_for_decoder,
    modality_type=HSCImage,
    bands=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
)


decoded_hsc_images = decoded_hsc.flux
print("Shape of decoded_hsc_images:", decoded_hsc_images.shape)

# 1. Use amin/amax which support tuple dimensions (dim=(1,2,3))
# 2. Add a small epsilon (1e-8) to avoid division by zero
min_vals = torch.amin(decoded_hsc_images, dim=(1, 2, 3), keepdim=True)
max_vals = torch.amax(decoded_hsc_images, dim=(1, 2, 3), keepdim=True)

decoded_hsc_images = (decoded_hsc_images - min_vals) / (max_vals - min_vals + 1e-8)
# Plot them with astropy
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

def create_rgb(flux_tensor, bands_indices):
    """
    Helper to create an RGB image from specific bands.
    Using Lupton et al. (2004) scaling.
    """
    # Move from [C, H, W] to [H, W, C] and convert to numpy
    data = flux_tensor.cpu().detach().numpy()

    # Select bands for RGB (e.g., i, r, g)
    r_idx, g_idx, b_idx = bands_indices

    # Optional: clip or normalize data if make_lupton_rgb is too sensitive
    rgb = make_lupton_rgb(data[r_idx], data[g_idx], data[b_idx], Q=10, stretch=0.5)
    return rgb

# Set up the figure
fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(15, 5 * N_EXAMPLES))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# Extract ground truth HSC for comparison
hsc_ground_truth = hsc_tensor.cpu().numpy()

for i in range(N_EXAMPLES):
    # 1. Input Legacy Survey
    legacy_rgb = create_rgb(legacy_tensor[i], [2, 1, 0])
    axes[i, 0].imshow(legacy_rgb, origin='lower')
    axes[i, 0].set_title(f"Input: Legacy Survey")
    axes[i, 0].axis('off')

    # 2. Ground Truth HSC (from your H5 file)
    gt_hsc_rgb = create_rgb(hsc_tensor[i], [2, 1, 0])
    axes[i, 1].imshow(gt_hsc_rgb, origin='lower')
    axes[i, 1].set_title(f"Ground Truth: HSC")
    axes[i, 1].axis('off')

    # 3. Model Prediction (Decoded HSC)
    pred_hsc_rgb = create_rgb(decoded_hsc_images[i], [2, 1, 0])
    axes[i, 2].imshow(pred_hsc_rgb, origin='lower')
    axes[i, 2].set_title(f"AION Prediction: HSC")
    axes[i, 2].axis('off')

# Save the plot
output_filename = 'aion_reconstruction_results.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved successfully to: {os.getcwd()}/{output_filename}")


print(decoded_hsc_images.shape)
print(decoded_hsc_images[0])
