"""
Dataset class for paired Euclid (VIS) x COSMOS (F115W) images.

Reads from the HDF5 file produced by build_hdf5.py. Images are already
preprocessed (ZP-rescaled, arcsinh range-compressed for COSMOS). This class
applies the final normalization step (zero-mean / unit-variance) so that
images are ready for the model.

HDF5 layout expected (from build_hdf5.py):
    euclid_images             — (N, 1, H_euc, W_euc) float32
    cosmos_images             — (N, 1, H_cos, W_cos) float32
    cosmos_images_downscaled  — (N, 1, H_euc, W_euc) float32
    catalog/euclid_paths      — string array (N,)
    catalog/cosmos_paths      — string array (N,)
    attrs: num_pairs, euclid_shape, cosmos_shape

Usage:
    from dataset import EuclidCosmosDataset, collate_pairs
    dataset = EuclidCosmosDataset(hdf5_path="euclid_cosmos_pairs.h5")
    loader = DataLoader(dataset, batch_size=64, shuffle=True,
                        num_workers=4, collate_fn=collate_pairs,
                        persistent_workers=True, pin_memory=True)
    # Batch: (euclid_imgs, cosmos_imgs, metadata)
    # euclid_imgs: (B, 1, H_euc, W_euc)
    # cosmos_imgs: (B, 1, H_euc, W_euc)  ← downscaled to match Euclid
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Per-survey [mean, std] of preprocessed pixel values.
NORM_DICT = {
    "euclid": [0.037, 0.024],
    "cosmos": [0.007, 0.158],
}


class EuclidCosmosDataset(Dataset):
    """
    Lazy-loading dataset for paired Euclid/COSMOS cutouts.

    Each sample is one galaxy observed by both instruments.
    Returns (euclid_img, cosmos_img, metadata) where cosmos_img is the
    downscaled version matching Euclid's spatial resolution.
    """

    def __init__(self, hdf5_path: str, norm_dict: dict = NORM_DICT):
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.file = None  # opened lazily, once per worker

        with h5py.File(hdf5_path, "r") as f:
            self.N = int(f.attrs["num_pairs"])

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._open_file()

        euc = torch.from_numpy(self.file["euclid_images"][idx].copy())
        cos = torch.from_numpy(self.file["cosmos_images_downscaled"][idx].copy())

        euc_mean, euc_std = self.norm_dict["euclid"]
        cos_mean, cos_std = self.norm_dict["cosmos"]
        euc = (euc - euc_mean) / euc_std
        cos = (cos - cos_mean) / cos_std

        metadata = {"idx": idx}
        return euc, cos, metadata


def collate_pairs(batch):
    """Stack (euclid, cosmos, metadata) samples into a batch."""
    euclid = torch.stack([b[0] for b in batch])
    cosmos = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return euclid, cosmos, metadata


def compute_norm_stats(hdf5_path: str, n_samples: int = 10_000) -> dict:
    """
    Estimate mean and std for each survey from a random subset of the data.
    Run this once after build_hdf5.py completes and update NORM_DICT.

    Example:
        stats = compute_norm_stats("euclid_cosmos_pairs.h5")
        print(stats)
        # {'euclid': [mean, std], 'cosmos': [mean, std]}
    """
    rng = np.random.default_rng(42)
    with h5py.File(hdf5_path, "r") as f:
        N = int(f.attrs["num_pairs"])
        idx = np.sort(rng.choice(N, size=min(n_samples, N), replace=False))
        euc = f["euclid_images"][idx].reshape(-1)
        cos = f["cosmos_images_downscaled"][idx].reshape(-1)

    stats = {
        "euclid": [float(euc.mean()), float(euc.std())],
        "cosmos": [float(cos.mean()), float(cos.std())],
    }
    print("Measured normalization stats:")
    for survey, (mean, std) in stats.items():
        print(f"  {survey}: mean={mean:.5f}, std={std:.5f}")
    return stats


def main():
    from torch.utils.data import DataLoader

    H5_PATH = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"

    print("Computing normalization stats...")
    stats = compute_norm_stats(H5_PATH)

    print("\nTesting dataset loading...")
    dataset = EuclidCosmosDataset(H5_PATH)
    print(f"  Dataset size: {len(dataset)}")
    euc, cos, meta = dataset[0]
    print(f"  Euclid shape: {euc.shape}, range [{euc.min():.3f}, {euc.max():.3f}]")
    print(f"  COSMOS shape: {cos.shape}, range [{cos.min():.3f}, {cos.max():.3f}]")

    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=2, collate_fn=collate_pairs,
                        persistent_workers=True)
    euc_batch, cos_batch, _ = next(iter(loader))
    print(f"\n  Batch — Euclid: {euc_batch.shape}, COSMOS: {cos_batch.shape}")
    print("Done.")



if __name__ == "__main__":
    main()