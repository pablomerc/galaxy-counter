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




print('Legacy tensor shape:', legacy_tensor.shape)
print('HSC tensor shape:', hsc_tensor.shape)

# Prepare your astronomical data (example: Legacy Survey image)
image_leg = LegacySurveyImage(
    flux=legacy_tensor,  # Shape: [batch, 4, height, width] for g,r,i,z bands
    bands=['DES-G', 'DES-R', 'DES-I', 'DES-Z']
)

print('Tokenizing image')
# Encode data to tokens
tokens = codec_manager.encode(image_leg)

print('Encoding image')
# Option 1: Extract embeddings for downstream tasks
embeddings = model.encode(tokens, num_encoder_tokens=600)

print('Embeddings shape:', embeddings.shape)


print('Now trying to prepare HSC + Legacy embeddings')

image_hsc = HSCImage(
    flux=hsc_tensor,
    bands=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
)

print('Tokenizing HSC image')
tokens_hsc = codec_manager.encode(image_hsc)

print('Encoding HSC image')
embeddings_hsc = model.encode(tokens_hsc, num_encoder_tokens=600)

print('HSC embeddings shape:', embeddings_hsc.shape)


print('Now trying to prepare HSC + Legacy embeddings')
tokens_hsc_leg = codec_manager.encode(image_hsc, image_leg)

# 1) One fused embedding that uses both modalities
emb_hsc_leg = model.encode(tokens_hsc_leg, num_encoder_tokens=1200)

print('HSC + Legacy embeddings shape:', emb_hsc_leg.shape)

emb_hsc_leg = emb_hsc_leg.mean(dim=1)




print('Now trying to predict redshift from Legacy image')
# Option 2: Generate predictions (e.g., redshift)
from aion.modalities import Z
preds = model(
    codec_manager.encode(image_leg),
    target_modality=Z,
)

# print('Redshift predictions shape:', preds.shape)
print(preds.keys())



print('Now trying to predict HSC image from Legacy')
preds = model(
    codec_manager.encode(image_leg),
    target_modality=HSCImage,
)

# print('HSC predictions keys', preds.keys())
# print(preds['tok_image_hsc'].shape)

# images_hsc = codec_manager.decode(preds, HSCImage)



# ... [Your existing script ends here]

print('## Trying to decode HSC image from Legacy')
# 1. Convert Logits to Indices
# The model outputs shape [Batch, Seq_Len, Vocab_Size].
# We take the 'argmax' to find the index of the highest probability token.
logits = preds['tok_image_hsc']
token_indices = torch.argmax(logits, dim=-1)  # Shape: [Batch, Seq_Len]

# 2. Prepare the dictionary for the decoder
# CodecManager.decode expects a dict format: { 'token_key': tensor_of_indices }
token_key = HSCImage.token_key  # This is 'tok_image_hsc'
tokens_for_decoder = {token_key: token_indices}

# 3. Decode back to Image
# We must pass 'bands' because the CodecManager docstring mentions
# metadata is required (e.g., "bands for images").
decoded_hsc = codec_manager.decode(
    tokens=tokens_for_decoder,
    modality_type=HSCImage,
    bands=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
)

# 4. Check the results
print("Decoded Flux Shape:", decoded_hsc.flux.shape)
# Output should be [Batch, 5, Height, Width]

hsc_images = decoded_hsc.flux # [Batch, 5, Height, Width]

# Plot them with astropy
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
import numpy as np
import os

# --- CLUSTER FIX: Use a non-interactive backend ---
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
    pred_hsc_rgb = create_rgb(hsc_images[i], [2, 1, 0])
    axes[i, 2].imshow(pred_hsc_rgb, origin='lower')
    axes[i, 2].set_title(f"AION Prediction: HSC")
    axes[i, 2].axis('off')

# Save the plot
output_filename = 'aion_reconstruction_results.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved successfully to: {os.getcwd()}/{output_filename}")

# Clean up memory
plt.close(fig)


import matplotlib.pyplot as plt
import torch

# --- Ensure backend is Agg for cluster ---
import matplotlib
matplotlib.use('Agg')

def save_raw_comparison(tensor_in, tensor_out, num_examples=5, filename='raw_comparison.png'):
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 5 * num_examples))

    for i in range(num_examples):
        # 1. Process Input (Legacy)
        # Select first 3 channels, permute to [H, W, C], convert to numpy
        raw_in = tensor_in[i][:3].permute(1, 2, 0).cpu().detach().numpy()

        # 2. Process Output (Decoded HSC)
        raw_out = tensor_out[i][:3].permute(1, 2, 0).cpu().detach().numpy()

        # Simple Normalization for visualization (0 to 1 range)
        # Raw astronomical flux can be negative or very large, so we clip
        raw_in = (raw_in - raw_in.min()) / (raw_in.max() - raw_in.min() + 1e-8)
        raw_out = (raw_out - raw_out.min()) / (raw_out.max() - raw_out.min() + 1e-8)

        axes[i, 0].imshow(raw_in)
        axes[i, 0].set_title(f"Raw Input (Ch 0,1,2) - Ex {i}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(raw_out)
        axes[i, 1].set_title(f"Raw Predicted (Ch 0,1,2) - Ex {i}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Raw plot saved to {filename}")

# Call the function
save_raw_comparison(legacy_tensor, hsc_images, num_examples=N_EXAMPLES)
