"""
Dataset class for HSC and Legacy Survey images. Returns them normalized.
"""

import torch
import torch.nn.functional as F
import h5py
from pathlib import Path
from torch.utils.data import Dataset
import random
import math
import time
# import torch
from torch.utils.data import Sampler
from torch.utils.data._utils.collate import default_collate


NORM_DICT = {
    'hsc': [0.022, 0.05],
    'legacy': [0.023, 0.063],
    'legacy_zoom': [0.045, 0.078],
    'hsc96': [0.00897, 0.0312],
    'legacy96': [0.0108, 0.050],
    'legacy96_zoom': [0.0173, 0.053],
    # 'hsc64': [0.022, 0.05], # TODO: actually measure these (for now using 48x48 stats)
    # 'legacy64': [0.023, 0.063],
    # 'legacy64_zoom': [0.045, 0.078],
}

# Note: for now the 48x48 stats are used for 64x64 images (TODO: actually measure these)



# === SUMMARY ===
# Before zoom - Mean: 0.024447, Std: 0.065834
# After zoom  - Mean: 0.044522, Std: 0.077894
# Mean change: 0.020075
# Std change:  0.012059

# For 96x96 images - based on 10k examples '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_laptop.h5'
# 'hsc96': [0.00897, 0.0312]
# 'legacy96': [0.0108, 0.050]
# 'legacy96_zoom': [0.0173, 0.053]

#Maybe should change to per channel normalization
# === BEFORE ZOOM ===
# HSC Images 96x96 Mean (per channel): [0.004292047116905451, 0.007768720388412476, 0.010728799737989902, 0.013123715296387672]
# HSC Images 96x96 Std (per channel): [0.01780957169830799, 0.027371028438210487, 0.03468557074666023, 0.03976300731301308]
# Legacy Images 96x96 Mean (per channel): [0.005523075349628925, 0.009824461303651333, 0.013055658899247646, 0.014965437352657318]
# Legacy Images 96x96 Std (per channel): [0.026880666613578796, 0.0409843772649765, 0.053803086280822754, 0.06831938773393631]

# === AFTER ZOOM ===
# Legacy Images 96x96 (zoomed) Mean (per channel): [0.008737790398299694, 0.01580115407705307, 0.020881297066807747, 0.02388158068060875]
# Legacy Images 96x96 (zoomed) Std (per channel): [0.030721960589289665, 0.04676024243235588, 0.05805562436580658, 0.06820148229598999]


# For lenses
# lense_indices = [3199, 3298, 4368, 4556, 8357, 9503, 19076, 20869, 26247, 40506, 51839, 53037, 60565, 60980, 64245, 72326, 74053, 77857, 99695]


class HSCLegacyDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
    ):
        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.num_images = len(idx_list) if idx_list is not None else None

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]
        mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        mean_legacy, std_legacy = self.norm_dict['legacy']
        legacy_image = (legacy_image - mean_legacy) / std_legacy
        return hsc_image, legacy_image

class HSCLegacyDatasetZoom(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
        is96: bool = False,
    ):
        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.num_images = len(idx_list) if idx_list is not None else None
        self.is96 = is96

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]

        if self.is96:
            mean_hsc, std_hsc = self.norm_dict['hsc96']
        else:
            mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc

        # Zoom legacy image and normalize with legacy_zoom stats
        legacy_image = zoom_legacy_image(legacy_image)
        if self.is96:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy96_zoom']
        else:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy_zoom']
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom
        return hsc_image, legacy_image

class HSCLegacyTripletDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
        deterministic_anchor_survey: bool = False,
    ):

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found:{hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.deterministic_anchor_survey = deterministic_anchor_survey
        self.num_images = len(idx_list) if idx_list is not None else None

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        """
        Returns an example with anchor image, same galaxy on the other instrument, and k examples of same instrument with different galaxies.

        Args:
            idx: Either an int (dataset index) or a tuple (idx, anchor_survey) when using BalancedAnchorBatchSampler.
                 If tuple, anchor_survey will be used instead of random choice.

        Returns:
            tuple: (anchor_image, same_galaxy, same_instrument, metadata)
                - anchor_image: torch.Tensor, shape (C, H, W) - normalized anchor image
                - same_galaxy: torch.Tensor, shape (C, H, W) - same galaxy from other instrument, normalized
                - same_instrument: torch.Tensor, shape (k, C, H, W) - k different galaxies from same instrument, normalized
                - metadata: dict with keys:
                    - 'anchor_survey': str, either 'hsc' or 'legacy'
                    - 'idx': int, the dataset index used
                    - 'num_same_instrument': int, actual number of same_instrument examples (may be < k for small datasets)
        """
        # Handle tuple from BalancedAnchorBatchSampler: (idx, anchor_survey)
        if isinstance(idx, tuple):
            idx, anchor_survey = idx
        else:
            anchor_survey = None  # Will be randomly chosen below

        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]
        mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc
        mean_legacy, std_legacy = self.norm_dict['legacy']
        legacy_image = (legacy_image - mean_legacy) / std_legacy

        # Use provided anchor_survey or choose deterministically/randomly
        if anchor_survey is None:
            if self.deterministic_anchor_survey:
                # Deterministic assignment: even idx -> 'hsc', odd idx -> 'legacy'
                anchor_survey = 'hsc' if idx % 2 == 0 else 'legacy'
            else:
                anchor_survey = random.choice(['hsc', 'legacy'])


        # TODO: Replace this by SNR-based matching
        k = 5
        # Generate enough candidates to ensure we get k unique indices (excluding idx)
        # Use a set to ensure uniqueness, and keep sampling until we have enough
        different_indexes_set = set()
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(different_indexes_set) < k and attempts < max_attempts:
            candidates = torch.randint(0, self.num_images, (k * 2,)).tolist()
            for cand_idx in candidates:
                if cand_idx != idx:
                    different_indexes_set.add(cand_idx)
                if len(different_indexes_set) >= k:
                    break
            attempts += 1

        if len(different_indexes_set) < k:
            # Fallback: if we can't get k unique indices, use what we have
            # This can happen with very small datasets
            different_indexes = torch.tensor(list(different_indexes_set), dtype=torch.long)
        else:
            different_indexes = torch.tensor(list(different_indexes_set)[:k], dtype=torch.long)

        anchor_image = None
        same_galaxy = None
        same_instrument = None

        if anchor_survey == 'hsc':
            anchor_image = hsc_image
            same_galaxy = legacy_image

            # Normalize same_instrument images
            same_instrument_raw = self.hsc_images[different_indexes]
            same_instrument = (same_instrument_raw - mean_hsc) / std_hsc

        elif anchor_survey == 'legacy':
            anchor_image = legacy_image
            same_galaxy = hsc_image

            # Normalize same_instrument images
            same_instrument_raw = self.legacy_images[different_indexes]
            same_instrument = (same_instrument_raw - mean_legacy) / std_legacy

        # Metadata dictionary for debugging, analysis, and logging
        metadata = {
            'anchor_survey': anchor_survey,
            'idx': idx,
            'num_same_instrument': len(different_indexes),
        }

        return anchor_image, same_galaxy, same_instrument, metadata

class HSCLegacyTripletDatasetZoom(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
        deterministic_anchor_survey: bool = False,
        is96: bool = False,
    ):

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found:{hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.deterministic_anchor_survey = deterministic_anchor_survey
        self.num_images = len(idx_list) if idx_list is not None else None
        self.is96 = is96

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        """
        Returns an example with anchor image, same galaxy on the other instrument, and k examples of same instrument with different galaxies.

        Args:
            idx: Either an int (dataset index) or a tuple (idx, anchor_survey) when using BalancedAnchorBatchSampler.
                 If tuple, anchor_survey will be used instead of random choice.

        Returns:
            tuple: (anchor_image, same_galaxy, same_instrument, metadata)
                - anchor_image: torch.Tensor, shape (C, H, W) - normalized anchor image
                - same_galaxy: torch.Tensor, shape (C, H, W) - same galaxy from other instrument, normalized
                - same_instrument: torch.Tensor, shape (k, C, H, W) - k different galaxies from same instrument, normalized
                - metadata: dict with keys:
                    - 'anchor_survey': str, either 'hsc' or 'legacy'
                    - 'idx': int, the dataset index used
                    - 'num_same_instrument': int, actual number of same_instrument examples (may be < k for small datasets)
        """
        # Handle tuple from BalancedAnchorBatchSampler: (idx, anchor_survey)
        if isinstance(idx, tuple):
            idx, anchor_survey = idx
        else:
            anchor_survey = None  # Will be randomly chosen below

        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]

        if self.is96:
            mean_hsc, std_hsc = self.norm_dict['hsc96']
        else:
            mean_hsc, std_hsc = self.norm_dict['hsc']

        hsc_image = (hsc_image - mean_hsc) / std_hsc
        # mean_legacy, std_legacy = self.norm_dict['legacy']
        if self.is96:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy96_zoom']
        else:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy_zoom']
        # legacy_image = (legacy_image - mean_legacy) / std_legacy
        legacy_image = zoom_legacy_image(legacy_image)
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom

        # Use provided anchor_survey or choose deterministically/randomly
        if anchor_survey is None:
            if self.deterministic_anchor_survey:
                # Deterministic assignment: even idx -> 'hsc', odd idx -> 'legacy'
                anchor_survey = 'hsc' if idx % 2 == 0 else 'legacy'
            else:
                anchor_survey = random.choice(['hsc', 'legacy'])


        # TODO: Replace this by SNR-based matching
        k = 5
        # Generate enough candidates to ensure we get k unique indices (excluding idx)
        # Use a set to ensure uniqueness, and keep sampling until we have enough
        different_indexes_set = set()
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        while len(different_indexes_set) < k and attempts < max_attempts:
            candidates = torch.randint(0, self.num_images, (k * 2,)).tolist()
            for cand_idx in candidates:
                if cand_idx != idx:
                    different_indexes_set.add(cand_idx)
                if len(different_indexes_set) >= k:
                    break
            attempts += 1

        if len(different_indexes_set) < k:
            # Fallback: if we can't get k unique indices, use what we have
            # This can happen with very small datasets
            different_indexes = torch.tensor(list(different_indexes_set), dtype=torch.long)
        else:
            different_indexes = torch.tensor(list(different_indexes_set)[:k], dtype=torch.long)

        anchor_image = None
        same_galaxy = None
        same_instrument = None

        if anchor_survey == 'hsc':
            anchor_image = hsc_image
            same_galaxy = legacy_image

            # Normalize same_instrument images
            same_instrument_raw = self.hsc_images[different_indexes]
            same_instrument = (same_instrument_raw - mean_hsc) / std_hsc

        elif anchor_survey == 'legacy':
            anchor_image = legacy_image
            same_galaxy = hsc_image

            # Zoom and normalize same_instrument images
            same_instrument_raw = self.legacy_images[different_indexes]
            same_instrument = zoom_legacy_image(same_instrument_raw)
            same_instrument = (same_instrument - mean_legacy_zoom) / std_legacy_zoom

        # Metadata dictionary for debugging, analysis, and logging
        metadata = {
            'anchor_survey': anchor_survey,
            'idx': idx,
            'num_same_instrument': len(different_indexes),
        }

        return anchor_image, same_galaxy, same_instrument, metadata


class HSCLegacyTripletDatasetZoomLenses(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
        is96: bool = False,
        lense_indices: list = None,
    ):

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found:{hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.is96 = is96
        self.lense_indices = lense_indices
        self.num_images = len(self.lense_indices)

        with h5py.File(hdf5_path, 'r') as f:

            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            self.hsc_images = torch.from_numpy(f['hsc_images'][self.lense_indices]).float()
            self.legacy_images = torch.from_numpy(f['legacy_images'][self.lense_indices]).float()
            self.hsc_pairs = torch.from_numpy(f['hsc_images'][:256]).float()
        print(f"Loaded {len(self.lense_indices)} lens images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        """
        Returns an example with anchor image, same galaxy on the other instrument, and k examples of same instrument with different galaxies.

        Args:
            idx: Either an int (dataset index) or a tuple (idx, anchor_survey) when using BalancedAnchorBatchSampler.
                 If tuple, anchor_survey will be used instead of random choice.

        Returns:
            tuple: (anchor_image, same_galaxy, same_instrument, metadata)
                - anchor_image: torch.Tensor, shape (C, H, W) - normalized anchor image
                - same_galaxy: torch.Tensor, shape (C, H, W) - same galaxy from other instrument, normalized
                - same_instrument: torch.Tensor, shape (k, C, H, W) - k different galaxies from same instrument, normalized
                - metadata: dict with keys:
                    - 'anchor_survey': str, either 'hsc' or 'legacy'
                    - 'idx': int, the dataset index used
                    - 'num_same_instrument': int, actual number of same_instrument examples (may be < k for small datasets)
        """

        if idx < 0 or idx >= len(self.lense_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.lense_indices)})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]

        if self.is96:
            mean_hsc, std_hsc = self.norm_dict['hsc96']
        else:
            mean_hsc, std_hsc = self.norm_dict['hsc']

        hsc_image = (hsc_image - mean_hsc) / std_hsc
        # mean_legacy, std_legacy = self.norm_dict['legacy']
        if self.is96:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy96_zoom']
        else:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy_zoom']
        # legacy_image = (legacy_image - mean_legacy) / std_legacy
        legacy_image = zoom_legacy_image(legacy_image)
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom

        # Always HSC as anchor for lens dataset
        anchor_survey = 'hsc'

        # TODO: Replace this by SNR-based matching
        k = 5
        different_indexes = torch.randint(0, 256, (k,))

        anchor_image = None
        same_galaxy = None
        same_instrument = None

        if anchor_survey == 'hsc':
            anchor_image = hsc_image
            same_galaxy = legacy_image

            # Normalize same_instrument images
            same_instrument_raw = self.hsc_pairs[different_indexes]
            same_instrument = (same_instrument_raw - mean_hsc) / std_hsc

        elif anchor_survey == 'legacy':
            raise NotImplementedError("Legacy anchor survey is not implemented")
        #     anchor_image = legacy_image
        #     same_galaxy = hsc_image

        #     # Zoom and normalize same_instrument images
        #     same_instrument_raw = self.hsc_pairs[different_indexes]
        #     same_instrument = zoom_legacy_image(same_instrument_raw)
        #     same_instrument = (same_instrument - mean_legacy_zoom) / std_legacy_zoom

        # Metadata dictionary for debugging, analysis, and logging
        metadata = {
            'anchor_survey': anchor_survey,
            'idx': idx,
            'num_same_instrument': len(different_indexes),
        }

        return anchor_image, same_galaxy, same_instrument, metadata


class HSCLegacyTripletDatasetMask(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        norm_dict: dict = NORM_DICT,
        idx_list: list = None,
        deterministic_anchor_survey: bool = False,
        is96: bool = True,
    ):

        hdf5_path = Path(hdf5_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found:{hdf5_path}")
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.idx_list = idx_list
        self.deterministic_anchor_survey = deterministic_anchor_survey
        self.num_images = len(idx_list) if idx_list is not None else None
        self.is96 = is96

        with h5py.File(hdf5_path, 'r') as f:
            total_images = f.attrs['num_images']
            self.crop_size = f.attrs['crop_size']
            self.num_channels = f.attrs['num_channels']
            if self.idx_list is not None:
                self.hsc_images = torch.from_numpy(f['hsc_images'][self.idx_list]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][self.idx_list]).float()
            else:
                self.hsc_images = torch.from_numpy(f['hsc_images'][:total_images]).float()
                self.legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()
        if self.idx_list is None:
            self.num_images = total_images
        print(f"Loaded {self.num_images} images into memory, "
        f"shape: ({self.num_images}, {self.num_channels}, {self.crop_size}, {self.crop_size})")
        print(f"Memory usage: ~{2 * self.hsc_images.numel() * 4 / (1024**3):.3f} GB")

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):
        """
        Returns an example with anchor image, same galaxy on the other instrument, and the anchor image with center masked.

        Args:
            idx: Either an int (dataset index) or a tuple (idx, anchor_survey) when using BalancedAnchorBatchSampler.
                 If tuple, anchor_survey will be used instead of random choice.

        Returns:
            tuple: (anchor_image, same_galaxy, same_instrument, metadata)
                - anchor_image: torch.Tensor, shape (C, H, W) - normalized anchor image
                - same_galaxy: torch.Tensor, shape (C, H, W) - same galaxy from other instrument, normalized
                - same_instrument: torch.Tensor, shape (1, C, H, W) - anchor image with center 32x32 pixels masked (set to 0)
                - metadata: dict with keys:
                    - 'anchor_survey': str, either 'hsc' or 'legacy'
                    - 'idx': int, the dataset index used
                    - 'num_same_instrument': int, always 1 for this dataset
        """
        # Handle tuple from BalancedAnchorBatchSampler: (idx, anchor_survey)
        if isinstance(idx, tuple):
            idx, anchor_survey = idx
        else:
            anchor_survey = None  # Will be randomly chosen below

        if idx < 0 or idx >= self.num_images:
            raise IndexError(f"Index {idx} out of range [0, {self.num_images})")
        hsc_image = self.hsc_images[idx]
        legacy_image = self.legacy_images[idx]

        # Normalize HSC image
        if self.is96:
            mean_hsc, std_hsc = self.norm_dict['hsc96']
        else:
            mean_hsc, std_hsc = self.norm_dict['hsc']
        hsc_image = (hsc_image - mean_hsc) / std_hsc

        # Zoom and normalize legacy image
        if self.is96:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy96_zoom']
        else:
            mean_legacy_zoom, std_legacy_zoom = self.norm_dict['legacy_zoom']
        legacy_image = zoom_legacy_image(legacy_image)
        legacy_image = (legacy_image - mean_legacy_zoom) / std_legacy_zoom

        # Use provided anchor_survey or choose deterministically/randomly
        if anchor_survey is None:
            if self.deterministic_anchor_survey:
                # Deterministic assignment: even idx -> 'hsc', odd idx -> 'legacy'
                anchor_survey = 'hsc' if idx % 2 == 0 else 'legacy'
            else:
                anchor_survey = random.choice(['hsc', 'legacy'])

        anchor_image = None
        same_galaxy = None
        same_instrument = None

        if anchor_survey == 'hsc':
            anchor_image = hsc_image
            same_galaxy = legacy_image
            same_instrument = center_mask(anchor_image)
            # Add dimension to match expected shape (k, C, H, W) where k=1
            same_instrument = same_instrument.unsqueeze(0)  # (1, C, H, W)

        elif anchor_survey == 'legacy':
            anchor_image = legacy_image
            same_galaxy = hsc_image
            same_instrument = center_mask(anchor_image)
            # Add dimension to match expected shape (k, C, H, W) where k=1
            same_instrument = same_instrument.unsqueeze(0)  # (1, C, H, W)


        # Metadata dictionary for debugging, analysis, and logging
        metadata = {
            'anchor_survey': anchor_survey,
            'idx': idx,
            'num_same_instrument': 1,
        }

        return anchor_image, same_galaxy, same_instrument, metadata



class BalancedAnchorBatchSampler(Sampler):
    """
    Yields batches of (idx, anchor_survey) tuples so that each batch is exactly 50/50
    in terms of anchor survey. The anchor_survey assignments are randomly shuffled within
    each batch to avoid systematic patterns.

    This ensures balanced training while maintaining randomness in the assignment.
    """
    def __init__(self, num_samples: int, batch_size: int, drop_last: bool = True, seed: int = 0):
        assert batch_size % 2 == 0, "batch_size must be even for 50/50 split"
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Shuffle indices once per epoch
        perm = torch.randperm(self.num_samples, generator=g).tolist()

        # Calculate number of batches
        n_full = self.num_samples // self.batch_size
        n_batches = n_full if self.drop_last else math.ceil(self.num_samples / self.batch_size)

        cursor = 0
        for _ in range(n_batches):
            if cursor + self.batch_size > self.num_samples:
                if self.drop_last:
                    break
                # If not dropping last, stop here (could pad if needed)
                break

            batch_idxs = perm[cursor:cursor + self.batch_size]
            cursor += self.batch_size

            # Create balanced anchor_survey assignments: half hsc, half legacy
            anchor_surveys = ['hsc'] * self.half + ['legacy'] * self.half
            # Shuffle the anchor_survey assignments within the batch for randomness
            anchor_survey_perm = torch.randperm(self.batch_size, generator=g).tolist()
            anchor_surveys_shuffled = [anchor_surveys[i] for i in anchor_survey_perm]

            # Pair each idx with its randomly assigned anchor_survey
            batch = [(idx, anchor_survey) for idx, anchor_survey in zip(batch_idxs, anchor_surveys_shuffled)]
            yield batch

def custom_collate_fn(batch):
    """
    Custom collate function that handles metadata dictionaries properly.
    """
    # Separate tensors from metadata
    anchor_images = [item[0] for item in batch]
    same_galaxies = [item[1] for item in batch]
    same_instruments = [item[2] for item in batch]
    metadata_list = [item[3] for item in batch]

    # Collate tensors normally
    collated_anchor = default_collate(anchor_images)
    collated_same_galaxy = default_collate(same_galaxies)
    collated_same_instrument = default_collate(same_instruments)

    # Keep metadata as a list of dicts (don't try to collate it)
    return collated_anchor, collated_same_galaxy, collated_same_instrument, metadata_list



def center_crop(image, crop_size: int=30):
    _, _, height, width = image.shape
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return image[
        :, :, start_y : start_y + crop_size, start_x : start_x + crop_size
    ]


def center_mask(image, mask_size: int=32):
    """
    Masks the center region of an image by setting pixels to 0.

    Args:
        image: torch.Tensor, shape (C, H, W) or (N, C, H, W)
        mask_size: int, size of the square mask (default: 32)

    Returns:
        torch.Tensor with same shape as input, with center region masked (set to 0)
    """
    # Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
    is_3d = len(image.shape) == 3
    if is_3d:
        image = image.unsqueeze(0)  # Add batch dimension

    _, _, height, width = image.shape
    start_x = (width - mask_size) // 2
    start_y = (height - mask_size) // 2

    # Create a copy to avoid modifying the original
    masked_image = image.clone()
    masked_image[:, :, start_y : start_y + mask_size, start_x : start_x + mask_size] = 0

    if is_3d:
        masked_image = masked_image.squeeze(0)  # Remove batch dimension

    return masked_image

def zoom_legacy_image(leg_im: torch.Tensor, factor = 0.64) -> torch.Tensor:
    """
    Zoom in the legacy images to have the same FoV as HSC

    Args:
        leg_im: torch.Tensor, shape (C, H, W) or (N, C, H, W)
        factor: zoom factor (default: 0.64)

    Returns:
        torch.Tensor with same shape as input, zoomed to match HSC FoV
    """
    # Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
    is_3d = len(leg_im.shape) == 3
    if is_3d:
        leg_im = leg_im.unsqueeze(0)  # Add batch dimension

    im_size = leg_im.shape[-1]
    new_size = round(factor * im_size)

    cropped_im = center_crop(leg_im, new_size)

    y = F.interpolate(
        cropped_im, size=(im_size, im_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,   # great for downsampling
    )

    if is_3d:
        y = y.squeeze(0)  # Remove batch dimension

    return y


def calculate_legacy_stats_before_after_zoom(
    hdf5_path: str,
    zoom_factor: float = 0.64,
    batch_size: int = 1000,
):
    """
    Load all legacy images from a 48x48 dataset, calculate mean/std before zoom,
    apply zoom, then calculate mean/std after zoom, and print both.

    Args:
        hdf5_path: Path to HDF5 file containing the dataset
        zoom_factor: Factor to use for zoom_legacy_image (default: 0.64)
        batch_size: Batch size for processing images (default: 1000)
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    start_time = time.time()
    print(f"Loading legacy images from {hdf5_path}")

    # Load all legacy images
    load_start = time.time()
    with h5py.File(hdf5_path, 'r') as f:
        total_images = f.attrs['num_images']
        crop_size = f.attrs['crop_size']
        num_channels = f.attrs['num_channels']

        print(f"Dataset info: {total_images} images, {crop_size}x{crop_size}, {num_channels} channels")

        # Load all legacy images
        legacy_images = torch.from_numpy(f['legacy_images'][:total_images]).float()

    load_time = time.time() - load_start
    print(f"Loaded {total_images} legacy images in {load_time:.2f} seconds")
    print(f"Image shape: {legacy_images.shape}")

    # Calculate mean and std BEFORE zoom
    stats_start = time.time()
    mean_before = legacy_images.mean().item()
    std_before = legacy_images.std().item()
    stats_time = time.time() - stats_start

    print(f"\n=== BEFORE ZOOM ===")
    print(f"Mean: {mean_before:.6f}")
    print(f"Std:  {std_before:.6f}")
    print(f"(Calculated in {stats_time:.2f} seconds)")

    # Apply zoom to all images in batches
    print(f"\nApplying zoom with factor {zoom_factor}...")
    zoomed_images = []

    num_batches = (total_images + batch_size - 1) // batch_size
    zoom_start = time.time()
    batch_times = []

    for batch_idx, i in enumerate(range(0, total_images, batch_size), 1):
        batch_start = time.time()
        end_idx = min(i + batch_size, total_images)
        batch = legacy_images[i:end_idx]

        # Apply zoom (batch is already 4D: (N, C, H, W))
        batch_zoomed = zoom_legacy_image(batch, factor=zoom_factor)
        zoomed_images.append(batch_zoomed)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Calculate progress and time estimates
        elapsed = time.time() - zoom_start
        avg_time_per_batch = elapsed / batch_idx
        remaining_batches = num_batches - batch_idx
        estimated_remaining = avg_time_per_batch * remaining_batches

        progress_pct = (end_idx / total_images) * 100

        print(f"Batch {batch_idx}/{num_batches} ({progress_pct:.1f}%): "
              f"Processed {end_idx}/{total_images} images | "
              f"Elapsed: {elapsed:.1f}s | "
              f"ETA: {estimated_remaining:.1f}s | "
              f"Speed: {batch_size/batch_time:.1f} img/s")

    # Concatenate all zoomed images
    concat_start = time.time()
    legacy_images_zoomed = torch.cat(zoomed_images, dim=0)
    concat_time = time.time() - concat_start
    zoom_total_time = time.time() - zoom_start
    print(f"Zoomed images shape: {legacy_images_zoomed.shape}")
    print(f"Zoom processing completed in {zoom_total_time:.2f} seconds (concat: {concat_time:.2f}s)")

    # Calculate mean and std AFTER zoom
    stats_start = time.time()
    mean_after = legacy_images_zoomed.mean().item()
    std_after = legacy_images_zoomed.std().item()
    stats_time = time.time() - stats_start

    print(f"\n=== AFTER ZOOM ===")
    print(f"Mean: {mean_after:.6f}")
    print(f"Std:  {std_after:.6f}")
    print(f"(Calculated in {stats_time:.2f} seconds)")

    total_time = time.time() - start_time

    print(f"\n=== SUMMARY ===")
    print(f"Before zoom - Mean: {mean_before:.6f}, Std: {std_before:.6f}")
    print(f"After zoom  - Mean: {mean_after:.6f}, Std: {std_after:.6f}")
    print(f"Mean change: {mean_after - mean_before:.6f}")
    print(f"Std change:  {std_after - std_before:.6f}")
    print(f"\n=== TIMING ===")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  - Loading: {load_time:.2f}s")
    print(f"  - Zoom processing: {zoom_total_time:.2f}s")
    print(f"  - Stats calculation: {stats_time:.2f}s")

    return {
        'mean_before': mean_before,
        'std_before': std_before,
        'mean_after': mean_after,
        'std_after': std_after,
    }


if __name__ == "__main__":
    # Default path to 48x48 dataset
    # hdf5_path = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"

    # calculate_legacy_stats_before_after_zoom(
    #     hdf5_path=hdf5_path,
    #     zoom_factor=0.64,
    #     batch_size=1000,
    # )

    # neighbors_path = "/data/vision/billf/scratch/pablomer/data/test_neighbours.h5"
    # neighbors_path = "/data/vision/billf/scratch/pablomer/data/test_neighbours_v2.h5"
    neighbors_path = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"

    with h5py.File(neighbors_path, 'r') as f:
        print(f.keys())
