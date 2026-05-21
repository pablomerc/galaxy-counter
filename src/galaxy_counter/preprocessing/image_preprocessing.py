'''
Script for preprocessing galaxy images (COSMOS and EUCLID).

Following the AION-1 Paper from Parker at al 2025.

'''
from __future__ import annotations
import torch


# Keeps track of the band indices for HSC and DES bands
BAND_TO_INDEX = {
    "VIS": 0,
    "F115W": 1
}
# Maximum band center values for HSC and DES bands
BAND_CENTER_MAX = {
    "VIS": 80,
    "F115W": 110
}

class CenterCrop:
    """Formatter that crops the images to have a fixed number of bands.
    i.e. It crops a square region of size crop_size × crop_size from the center of each image.
    """

    def __init__(self, crop_size: int = 96):
        self.crop_size = crop_size

    def __call__(self, image):
        _, _, height, width = image.shape
        start_x = (width - self.crop_size) // 2
        start_y = (height - self.crop_size) // 2
        return image[
            :, :, start_y : start_y + self.crop_size, start_x : start_x + self.crop_size
        ]


class Clamp:
    """Formatter that clamps the images to a given range."""

    def __init__(self):
        self.clamp_dict = BAND_CENTER_MAX

    def __call__(self, image, bands):
        for i, band in enumerate(bands):
            image[:, i, :, :] = torch.clip(
                image[:, i, :, :], -self.clamp_dict[band], self.clamp_dict[band]
            )
        return image
    

COSMOS_ZP = 23.9

# Euclid AB zeropoints per filter — verify these for your data release
EUCLID_ZP = {
    "VIS": 25.6578,   # TODO: verify for your DR
    "Y":   25.0,      # TODO: verify for your DR
    "J":   25.0,      # TODO: verify for your DR
    "H":   25.0,      # TODO: verify for your DR
}


class RescaleToCOSMOS:
    """Rescales Euclid flux to the COSMOS zeropoint (23.9 AB).

    COSMOS images are already at ZP=23.9, so they pass through unchanged.
    Euclid images are multiplied by 10^((COSMOS_ZP - EUCLID_ZP[band]) / 2.5).
    """

    def _scale(self, band: str) -> float:
        if band not in EUCLID_ZP:
            return 1.0  # COSMOS band — no rescaling needed
        return 10.0 ** ((COSMOS_ZP - EUCLID_ZP[band]) / 2.5)

    def forward(self, image: torch.Tensor, band: str) -> torch.Tensor:
        return image.clone() * self._scale(band)

    def backward(self, image: torch.Tensor, band: str) -> torch.Tensor:
        return image.clone() / self._scale(band)


class RangeCompress:
    """Formatter that applies arcsinh-based range compression."""

    def __init__(self, range_compression_factor: float = 0.01, mult_factor: float = 10.0):
        """
        Initialize range compression.

        Args:
            range_compression_factor: Factor for arcsinh compression (default: 0.01)
            mult_factor: Multiplicative factor after compression (default: 10.0)
        """
        self.range_compression_factor = range_compression_factor
        self.mult_factor = mult_factor

    def forward(self, image):
        """
        Apply range compression: arcsinh(x / factor) * factor * mult_factor.

        Args:
            image: Input tensor

        Returns:
            Range-compressed tensor
        """
        image = image.clone()  # Avoid in-place modification
        image = (
            torch.arcsinh(image / self.range_compression_factor)
            * self.range_compression_factor
        )
        image = image * self.mult_factor
        return image

    def backward(self, image):
        """
        Reverse range compression.

        Args:
            image: Range-compressed tensor

        Returns:
            Decompressed tensor
        """
        image = image.clone()  # Avoid in-place modification
        image = image / self.mult_factor
        image = (
            torch.sinh(image / self.range_compression_factor)
            * self.range_compression_factor
        )
        return image


def get_survey(bands: list[str]) -> str:
    """
    Extract survey name from band names.

    Args:
        bands: List of band names (e.g., ['EUC-VIS', 'EUC-Y', ...])

    Returns:
        Survey name (e.g., 'EUC' or 'COS')
    """
    if not bands:
        raise ValueError("bands list cannot be empty")
    survey = bands[0].split("-")[0]
    return survey


def preprocess_image(
    image: torch.Tensor,
    bands: list[str],
    crop_size: int = 96,
    range_compression_factor: float = 0.01,
    mult_factor: float = 10.0,
    apply_range_compression: bool = True,
) -> torch.Tensor:
    """
    Apply full preprocessing pipeline to an image.

    Pipeline steps:
    1. Center crop to specified size
    2. Clamp values to band-specific ranges
    3. Rescale based on survey zeropoint
    4. (Optional) Apply range compression

    Args:
        image: Input image tensor with shape [batch, channels, height, width]
        bands: List of band names corresponding to channels
        crop_size: Size to crop to (default: 96)
        range_compression_factor: Factor for range compression (default: 0.01)
        mult_factor: Multiplicative factor for range compression (default: 10.0)
        apply_range_compression: Whether to apply range compression (default: True)

    Returns:
        Preprocessed image tensor
    """
    # Step 1: Center crop
    cropper = CenterCrop(crop_size=crop_size)
    processed = cropper(image)

    # Step 2: Clamp
    clamper = Clamp()
    processed = clamper(processed.clone(), bands)

    # Step 3: Rescale
    survey = get_survey(bands)
    rescaler = RescaleToCOSMOS()
    processed = rescaler.forward(processed.clone(), survey)

    # Step 4: Range compression (optional)
    if apply_range_compression:
        range_compressor = RangeCompress(
            range_compression_factor=range_compression_factor,
            mult_factor=mult_factor,
        )
        processed = range_compressor.forward(processed.clone())

    return processed



# Define ordered band lists for v2 lookup
EUC_BANDS = ["EUC-VIS", "EUC-Y", "EUC-J", "EUC-K"]
COSMOS_BANDS = ["COS-F115W", "COS-F150W", "COS-F277W", "COS-F444W"]


def preprocess_image_v2(
    image: torch.Tensor,
    crop_size: int = 96,
    survey: str = "cosmos",
    bands: list[str] | None = None,
) -> torch.Tensor:
    """
    Simplified preprocessing pipeline (V2).

    Infers bands from survey name ('cosmos' or 'euclid') unless `bands` is
    passed explicitly, in which case survey is ignored and any channel count works.
    Expects input shape (C, H, W) or (B, C, H, W).
    """
    # Handle dimensions: Ensure [Batch, Channel, H, W] for the classes
    is_batched = image.ndim == 4
    if not is_batched:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        else:
            raise ValueError(f"Image must be 3D or 4D tensor, got shape {image.shape}")

    # Determine bands
    if bands is None:
        survey_key = survey.lower().strip()
        if survey_key == 'cos':
            bands = COSMOS_BANDS
        elif survey_key == 'euc':
            bands = EUC_BANDS
        else:
            raise ValueError(f"Unknown survey: '{survey}'. Supported: 'cosmos', 'euclid'")
        if image.shape[1] != len(bands):
            raise ValueError(
                f"Survey '{survey}' expects {len(bands)} channels, got {image.shape[1]}"
            )

    # 3. Pipeline Execution

    # Crop (Default 96)
    # cropper = CenterCrop(crop_size=crop_size)
    # processed = cropper(image)

    # Clamp
    # clamper = Clamp()
    # processed = clamper(processed.clone(), bands)

    # Rescale (Uses survey string to decide logic)
    rescaler = RescaleToCOSMOS()
    processed = rescaler.forward(processed.clone(), survey)

    # Range Compress (Defaults)
    range_compressor = RangeCompress()
    processed = range_compressor.forward(processed.clone())

    # 4. Output handling
    # If input was not batched (3D), return 3D. If batched, return 4D.
    if not is_batched:
        processed = processed.squeeze(0)

    return processed


def main():
    """Demonstrate the preprocessing pipeline on one Euclid VIS and one COSMOS F115W cutout."""
    import numpy as np
    from astropy.io import fits

    EUCLID_FILE = "/n03data/fontirro/euclid/40_cutouts/40_cutouts-vis/cutout_process_032_147271e7TILE_101542815_14969272158331464_149.692722_1.968534_1502704970019216576_cutout.fits"
    #COSMOS_FILE = "/n03data/fontirro/cosmos/120_cutouts/F115W_your_galaxy_id.fits"


    label,filepath,hdu_index, band =  "EUC-VIS", EUCLID_FILE, 1, "VIS"

    print("\n" + "=" * 60)
    print(f"PREPROCESSING PIPELINE — {label}")
    print("=" * 60)

    # Load image data.
    with fits.open(filepath) as hdul:
        data = hdul[hdu_index].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]
    im_full = torch.from_numpy(data).unsqueeze(0)  # (1, 1, H, W)
    print(f"\n1. Original image shape: {im_full.shape}")
    print(f"   Range: [{im_full.min():.4f}, {im_full.max():.4f}]")

    # Step 3: Rescale Euclid to COSMOS ZP (23.9); COSMOS passes through unchanged
    rescaler = RescaleToCOSMOS()
    im_rescaled = rescaler.forward(im_full.clone(), band)
    print(f"\n2. After rescale.forward (band={band}): {im_rescaled.shape}")
    print(f"   Range: [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")

    # Step 4: Range compression
    range_compression_factor = 0.01
    mult_factor = 10.0
    range_compressor = RangeCompress(
        range_compression_factor=range_compression_factor,
        mult_factor=mult_factor,
    )
    im_range_compressed = range_compressor.forward(im_rescaled.clone())
    print(f"\n3. After range_compress: {im_range_compressed.shape}")
    print(f"   Range: [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    print(f"   range_compression_factor: {range_compression_factor}")
    print(f"   mult_factor: {mult_factor}")
    print(f"   Formula: arcsinh(x / {range_compression_factor}) * {range_compression_factor} * {mult_factor}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF TRANSFORMATIONS")
    print("=" * 60)
    print(f"Original range:    [{im_full.min():.4f}, {im_full.max():.4f}]")
    print(f"Rescaled range:    [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")
    print(f"Range compressed:  [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    print("=" * 60)

    # Comparison: preprocess_image_v2 with explicit bands
    print("\n" + "=" * 60)
    print("USING preprocess_image_v2 FUNCTION: preprocess_image_v2()")
    print("=" * 60)
    im_preprocessed = preprocess_image_v2(im_full, bands=[band])
    print(f"Preprocessed image shape: {im_preprocessed.shape}")
    print(f"Preprocessed image range: [{im_preprocessed.min():.4f}, {im_preprocessed.max():.4f}]")
    print("=" * 60)



if __name__ == "__main__":
    main()
