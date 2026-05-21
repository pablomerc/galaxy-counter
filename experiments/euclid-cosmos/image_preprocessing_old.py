'''
Script for preprocessing galaxy images.

Following the AION-1 Paper from Parker at al 2025.

'''
from __future__ import annotations
import torch


# Keeps track of the band indices for HSC and DES bands
BAND_TO_INDEX = {
    "HSC-G": 0,
    "HSC-R": 1,
    "HSC-I": 2,
    "HSC-Z": 3,
    "HSC-Y": 4,
    "DES-G": 5,
    "DES-R": 6,
    "DES-I": 7,
    "DES-Z": 8,
}
# Maximum band center values for HSC and DES bands
BAND_CENTER_MAX = {
    "HSC-G": 80,
    "HSC-R": 110,
    "HSC-I": 200,
    "HSC-Z": 330,
    "HSC-Y": 500,
    "DES-G": 6,
    "DES-R": 15,
    "DES-I": 20,
    "DES-Z": 25,
}

class CenterCrop:
    """Formatter that crops the images to have a fixed number of bands."""

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


class RescaleToLegacySurvey:
    """Formatter that rescales the images based on survey zeropoint."""

    def __init__(self):
        pass

    def convert_zeropoint(self, zp: float) -> float:
        """Convert zeropoint to scale factor."""
        return 10.0 ** ((zp - 22.5) / 2.5)

    def reverse_zeropoint(self, scale: float) -> float:
        """Convert scale factor back to zeropoint."""
        return 22.5 - 2.5 * torch.log10(scale)

    def forward(self, image, survey):
        """Rescale image by dividing by zeropoint scale factor."""
        zpscale = self.convert_zeropoint(27.0) if (survey and survey.upper() == "HSC") else 1.0
        image = image.clone()  # Avoid in-place modification
        image /= zpscale
        return image

    def backward(self, image, survey):
        """Reverse rescale by multiplying by zeropoint scale factor."""
        zpscale = self.convert_zeropoint(27.0) if (survey and survey.upper() == "HSC") else 1.0
        image = image.clone()  # Avoid in-place modification
        image *= zpscale
        return image


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
        bands: List of band names (e.g., ['HSC-G', 'HSC-R', ...])

    Returns:
        Survey name (e.g., 'HSC' or 'DES')
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
    rescaler = RescaleToLegacySurvey()
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
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
LEGACY_BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]


def preprocess_image_v2(image: torch.Tensor, crop_size: int = 96, survey: str = "hsc") -> torch.Tensor:
    """
    Simplified preprocessing pipeline (V2).

    Automatically infers bands based on survey name ('hsc' or 'legacy').
    Expects input shape (C, 160, 160) or (B, C, 160, 160).
    """
    # 1. Standardize inputs
    survey_key = survey.lower().strip()

    # Handle dimensions: Ensure [Batch, Channel, H, W] for the classes
    is_batched = image.ndim == 4
    if not is_batched:
        if image.ndim == 3:
            image = image.unsqueeze(0) # [C, H, W] -> [1, C, H, W]
        else:
            raise ValueError(f"Image must be 3D or 4D tensor, got shape {image.shape}")

    # 2. Determine bands and validate channel count
    if survey_key == 'hsc':
        bands = HSC_BANDS
        expected_channels = 5
    elif survey_key == 'legacy':
        bands = LEGACY_BANDS
        expected_channels = 4
    else:
        raise ValueError(f"Unknown survey: '{survey}'. Supported: 'hsc', 'legacy'")

    if image.shape[1] != expected_channels:
        raise ValueError(f"Survey '{survey}' expects {expected_channels} channels, but got input with {image.shape[1]}")

    # 3. Pipeline Execution

    # Crop (Default 96)
    cropper = CenterCrop(crop_size=crop_size)
    processed = cropper(image)

    # Clamp
    clamper = Clamp()
    processed = clamper(processed.clone(), bands)

    # Rescale (Uses survey string to decide logic)
    rescaler = RescaleToLegacySurvey()
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
    """Demonstrate the full preprocessing pipeline."""
    import glob
    import os
    import numpy as np
    from datasets import load_dataset

    # Load dataset
    dataset_path = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'
    pattern = os.path.join(dataset_path, "*.parquet")
    all_files = sorted(glob.glob(pattern))
    dataset = load_dataset("parquet", data_files=all_files, split="train")

    idx = 10
    example_record = dataset[idx]

    # Get HSC image
    hsc_image = np.array(example_record['hsc_image']['flux'])
    hsc_image_tensor = torch.from_numpy(hsc_image).float()
    print(f"Original HSC image shape: {hsc_image_tensor.shape}")

    # Define bands
    bands_in = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']
    im_full = hsc_image_tensor.unsqueeze(0)  # Add batch dimension: [1, 5, 160, 160]

    print("\n" + "=" * 60)
    print("FULL PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"\n1. Original image shape: {im_full.shape}")
    print(f"   Range: [{im_full.min():.4f}, {im_full.max():.4f}]")

    # Step 1: Center crop
    cropper = CenterCrop(crop_size=96)
    im_cropped = cropper(im_full)
    print(f"\n2. After center_crop (96x96): {im_cropped.shape}")
    print(f"   Range: [{im_cropped.min():.4f}, {im_cropped.max():.4f}]")

    # Step 2: Clamp
    clamper = Clamp()
    im_clamped = clamper(im_cropped.clone(), bands_in)
    print(f"\n3. After clamp: {im_clamped.shape}")
    print(f"   Range: [{im_clamped.min():.4f}, {im_clamped.max():.4f}]")
    print(f"   (Clamped to band-specific ranges)")

    # Step 3: Rescale to Legacy Survey
    survey = get_survey(bands_in)
    rescaler = RescaleToLegacySurvey()
    im_rescaled = rescaler.forward(im_clamped.clone(), survey)
    print(f"\n4. After rescale.forward (survey={survey}): {im_rescaled.shape}")
    print(f"   Range: [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")
    zpscale = rescaler.convert_zeropoint(27.0) if survey == "HSC" else 1.0
    print(f"   Zeropoint scale factor: {zpscale:.6f}")

    # Step 4: Range compression
    range_compression_factor = 0.01
    mult_factor = 10.0
    range_compressor = RangeCompress(
        range_compression_factor=range_compression_factor,
        mult_factor=mult_factor,
    )
    im_range_compressed = range_compressor.forward(im_rescaled.clone())
    print(f"\n5. After range_compress: {im_range_compressed.shape}")
    print(f"   Range: [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    print(f"   range_compression_factor: {range_compression_factor}")
    print(f"   mult_factor: {mult_factor}")
    print(f"   Formula: arcsinh(x / {range_compression_factor}) * {range_compression_factor} * {mult_factor}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF TRANSFORMATIONS")
    print("=" * 60)
    print(f"Original range:      [{im_full.min():.4f}, {im_full.max():.4f}]")
    print(f"Cropped range:      [{im_cropped.min():.4f}, {im_cropped.max():.4f}]")
    print(f"Clamped range:      [{im_clamped.min():.4f}, {im_clamped.max():.4f}]")
    print(f"Rescaled range:     [{im_rescaled.min():.4f}, {im_rescaled.max():.4f}]")
    print(f"Range compressed:   [{im_range_compressed.min():.4f}, {im_range_compressed.max():.4f}]")
    print("=" * 60)

    # Demonstrate using the convenience function
    print("\n" + "=" * 60)
    print("USING CONVENIENCE FUNCTION: preprocess_image()")
    print("=" * 60)
    im_preprocessed = preprocess_image(
        im_full,
        bands_in,
        crop_size=96,
        range_compression_factor=range_compression_factor,
        mult_factor=mult_factor,
        apply_range_compression=True,
    )
    print(f"Preprocessed image shape: {im_preprocessed.shape}")
    print(f"Preprocessed image range: [{im_preprocessed.min():.4f}, {im_preprocessed.max():.4f}]")
    print("=" * 60)



if __name__ == "__main__":
    main()
