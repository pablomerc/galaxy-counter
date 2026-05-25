"""
Neighbors dataset for HSC/Legacy triplets (target, same-galaxy, same-instrument neighbors).

Use from a training script:

  from galaxy_counter.neighbors import (
      NeighborsDataset,
      NeighborsDatasetRawRAM,
      collate_neighbors,
      NORM_DICT,
  )
  train_dataset = NeighborsDataset(hdf5_path="path/to/neighbours.h5", ...)
  train_loader = DataLoader(
      train_dataset,
      batch_size=64,
      shuffle=True,
      num_workers=4,
      collate_fn=collate_neighbors,
      persistent_workers=True,
      pin_memory=True,
  )
  # Batch: (targets, samegals, padded_neighbors, neighbor_masks, metadata)
"""

import time

from galaxy_counter.preprocessing.image_preprocessing import preprocess_image_v2
from galaxy_counter.data import zoom_legacy_image

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

NORM_DICT = {
    "hsc": [0.022, 0.05],
    "legacy": [0.023, 0.063],
    "legacy_zoom": [0.045, 0.078],
    "hsc96": [0.00897, 0.0312],
    "legacy96": [0.0108, 0.050],
    "legacy96_zoom": [0.0173, 0.053],
}


def preprocess_raw_image(image, survey: str = "hsc", crop_size: int = 48, norm_dict: dict = NORM_DICT) -> torch.Tensor:
    """Preprocess raw images: crop, clamp, rescale, range compress; zoom for legacy; then normalize."""
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).float()
    image = preprocess_image_v2(image, crop_size=crop_size, survey=survey)
    if survey == "legacy":
        image = zoom_legacy_image(image)
        mean, std = norm_dict["legacy_zoom"]
    else:
        mean, std = norm_dict["hsc"]
    image = (image - mean) / std
    return image


class NeighborsDataset(Dataset):
    def __init__(self, hdf5_path, norm_dict=NORM_DICT, crop_size=48, max_neighbors=15):
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.crop_size = crop_size
        self.max_neighbors = max_neighbors
        self.file = None  # Handle for lazy loading

        # Pre-filter indices once in __init__ to avoid doing logic in __getitem__
        with h5py.File(self.hdf5_path, 'r') as f:
            sources = f['source_type'][:]
            indexes_mmu = np.where(sources == 0)[0]

            neigh_hsc = f["neighbor_idx_hsc"][indexes_mmu]
            neigh_legacy = f["neighbor_idx_legacy"][indexes_mmu]

            # Vectorized check for "good" indices
            good_both = (~np.all(neigh_hsc == -1, axis=1)) & (~np.all(neigh_legacy == -1, axis=1))
            self.indexes_mmu = indexes_mmu[good_both]

            # Pre-cache the neighbor indices for the filtered set to avoid double-reading
            self.cached_neighbor_hsc = neigh_hsc[good_both]
            self.cached_neighbor_legacy = neigh_legacy[good_both]

    def _open_file(self):
        """Opens the HDF5 file once per worker process."""
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.indexes_mmu)

    def __getitem__(self, idx):
        self._open_file()
        index_mmu = self.indexes_mmu[idx]

        # Decide survey once
        anchor_is_hsc = (idx % 2 == 0)
        anchor_survey = 'hsc' if anchor_is_hsc else 'legacy'

        # Fetch primary images
        img_hsc = self.file['images_hsc'][index_mmu]
        img_legacy = self.file['images_legacy'][index_mmu]

        if anchor_is_hsc:
            target_raw, samegal_raw = img_hsc, img_legacy
            # Use pre-cached neighbor indices
            neighbor_ids = self.cached_neighbor_hsc[idx]
            images_ds = self.file['images_hsc']
            survey_key, pair_key = "hsc", "legacy"
        else:
            target_raw, samegal_raw = img_legacy, img_hsc
            neighbor_ids = self.cached_neighbor_legacy[idx]
            images_ds = self.file['images_legacy']
            survey_key, pair_key = "legacy", "hsc"

        # Filter neighbors
        neighbor_ids = neighbor_ids[neighbor_ids != -1][:self.max_neighbors]

        # Batch-read neighbor images (one HDF5 slice; h5py requires indices in increasing order)
        if len(neighbor_ids) > 0:
            order = np.argsort(neighbor_ids)
            sorted_ids = neighbor_ids[order]
            neigh_imgs_sorted = images_ds[np.asarray(sorted_ids)]
            sameins_list = [
                preprocess_raw_image(neigh_imgs_sorted[i], survey_key, self.crop_size, self.norm_dict)
                for i in range(len(sorted_ids))
            ]
        else:
            sameins_list = []

        target = preprocess_raw_image(target_raw, survey_key, self.crop_size, self.norm_dict)
        samegal = preprocess_raw_image(samegal_raw, pair_key, self.crop_size, self.norm_dict)

        # Slice to 4 channels if HSC
        if anchor_is_hsc:
            target = target[:4]
            sameins = torch.stack(sameins_list, dim=0)[:, :4] if sameins_list else torch.empty(0, 4, self.crop_size, self.crop_size)
        else:
            samegal = samegal[:4]
            sameins = torch.stack(sameins_list, dim=0) if sameins_list else torch.empty(0, 3, self.crop_size, self.crop_size)

        metadata = {"anchor_survey": anchor_survey, "idx": idx, "num_same_instrument": len(sameins_list)}
        return target, samegal, sameins, metadata


def collate_neighbors(batch):
    """Collate list of (target, samegal, sameins, metadata) into batched tensors and padded sameins."""
    targets = torch.stack([item[0] for item in batch])
    samegals = torch.stack([item[1] for item in batch])

    # Handle the variable length 'sameins'
    neighbor_tensors = [item[2] for item in batch] # List of (N_i, C, H, W)

    # Get the max number of neighbors in THIS batch
    max_n = max(t.size(0) for t in neighbor_tensors)

    padded_neighbors = []
    neighbor_masks = []

    for t in neighbor_tensors:
        n_current = t.size(0)
        # Pad (C, H, W) is static, we only pad the first dimension (N)
        # F.pad expects padding from the last dim backwards: (W, W, H, H, C, C, N_top, N_bottom)
        # Easier to just create a zeros tensor and copy
        pad_size = (max_n, *t.shape[1:])
        padded_t = torch.zeros(pad_size, dtype=t.dtype)
        padded_t[:n_current] = t

        padded_neighbors.append(padded_t)

        # Create a mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n_current] = True
        neighbor_masks.append(mask)

    # Re-wrap metadata into a list of dicts
    metadata = [item[3] for item in batch]

    return (
        targets,
        samegals,
        torch.stack(padded_neighbors),
        torch.stack(neighbor_masks),
        metadata,
    )




### OPTION 2: Load pre-computed batches into memory


# --- The Dataset Class ---
class NeighborsPrecomputedDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        Loads the entire dataset into RAM.
        Expects keys: 'targets', 'samegals', 'sameins', 'neighbor_masks', 'meta_idx', etc.
        """
        print(f"Loading {hdf5_path} into RAM...")
        t0 = time.time()

        with h5py.File(hdf5_path, 'r') as f:
            # 1. Load Tensors
            # We use [:] to force reading the whole dataset into a numpy array
            print("Reading targets...")
            self.targets = torch.from_numpy(f['targets'][:])

            print("Reading samegals...")
            self.samegals = torch.from_numpy(f['samegals'][:])

            print("Reading neighbors (this is the big one)...")
            self.sameins = torch.from_numpy(f['sameins'][:])

            print("Reading masks...")
            self.masks = torch.from_numpy(f['neighbor_masks'][:])

            # 2. Load Metadata (Optional, but good to have)
            # We keep these as numpy/list to avoid overhead of converting to tensor if they are strings
            if 'meta_idx' in f:
                self.meta_idx = f['meta_idx'][:]
            else:
                self.meta_idx = np.zeros(len(self.targets))

            if 'meta_survey' in f:
                # Decode bytes to strings if necessary
                raw_survey = f['meta_survey'][:]
                self.meta_survey = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in raw_survey]
            else:
                self.meta_survey = ["unknown"] * len(self.targets)

            if 'meta_num_same_instrument' in f:
                self.meta_num_same = f['meta_num_same_instrument'][:]
            else:
                self.meta_num_same = np.zeros(len(self.targets))

        # 3. Validation
        assert len(self.targets) == len(self.sameins)
        print(f"Loaded {len(self.targets)} samples in {time.time() - t0:.2f}s.")
        print(f"Tensors size in RAM: ~{self.sameins.element_size() * self.sameins.numel() / 1e9:.2f} GB (just for neighbors)")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Access is instant since it's just array slicing

        # Reconstruct the metadata dict to match your old format
        metadata = {
            "idx": self.meta_idx[idx],
            "anchor_survey": self.meta_survey[idx],
            "num_same_instrument": self.meta_num_same[idx]
        }

        return (
            self.targets[idx],
            self.samegals[idx],
            self.sameins[idx],
            self.masks[idx],
            metadata
        )

# --- Collate Function ---
# Since data is ALREADY padded in the file, we don't need complex logic.
# We just stack them.
def simple_collate(batch):
    targets = torch.stack([b[0] for b in batch])
    samegals = torch.stack([b[1] for b in batch])
    sameins = torch.stack([b[2] for b in batch])
    masks = torch.stack([b[3] for b in batch])
    metadata = [b[4] for b in batch] # Keep as list of dicts

    return targets, samegals, sameins, masks, metadata



def plot_option2_first_batch(batch, save_path=None, num_samples=4, max_neighbors_show=5):
    """
    Plot targets, samegals, and neighbors for the first batch from NeighborsPrecomputedDataset.

    batch: (targets, samegals, sameins, masks, metadata) from simple_collate.
    save_path: where to save the figure (default: neighbors_option2_first_batch.png).
    num_samples: number of dataset samples (rows) to show.
    max_neighbors_show: max neighbor columns to display per sample.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets, samegals, sameins, masks, metadata = batch
    # shapes: targets/samegals (B, C, H, W), sameins (B, N, C, H, W), masks (B, N)
    B, N_max, C, H, W = sameins.shape
    num_samples = min(num_samples, B)

    def tensor_to_rgb(img):
        """(C, H, W) -> (H, W, 3) scaled for display."""
        x = img.detach().cpu().float().numpy()
        if x.shape[0] >= 3:
            x = x[:3]  # use first 3 channels as RGB
        else:
            x = np.stack([x[0]] * 3, axis=0)
        x = np.transpose(x, (1, 2, 0))
        lo, hi = np.percentile(x, (2, 98))
        if hi > lo:
            x = (x - lo) / (hi - lo)
        x = np.clip(x, 0, 1)
        return x

    # For each sample we show: 1 target + 1 samegal + up to max_neighbors_show neighbors
    n_cols = 2 + max_neighbors_show
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(2 * n_cols, 2 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    for i in range(num_samples):
        axes[i, 0].imshow(tensor_to_rgb(targets[i]))
        axes[i, 0].set_title("Target")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tensor_to_rgb(samegals[i]))
        axes[i, 1].set_title("Same galaxy")
        axes[i, 1].axis("off")

        n_valid = int(masks[i].sum().item())
        for j in range(max_neighbors_show):
            ax = axes[i, 2 + j]
            if j < n_valid:
                ax.imshow(tensor_to_rgb(sameins[i, j]))
                ax.set_title(f"Neighbor {j + 1}")
            else:
                ax.set_facecolor("0.9")
                ax.set_title(f"Neighbor {j + 1}" if j < N_max else "—")
            ax.axis("off")

    meta = metadata[0]
    fig.suptitle(
        f"Option 2 first batch (survey={meta.get('anchor_survey', '?')}, "
        f"batch_size={B}, {N_max} neighbor slots)",
        fontsize=10,
    )
    plt.tight_layout()
    out = save_path or "neighbors_option2_first_batch.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {out}")





## OPTION 3: NeighborsDatasetRawRAM (raw arrays in RAM, preprocess on the fly)


class NeighborsDatasetRawRAM(Dataset):
    """
    Loads raw HDF5 image arrays and neighbor indices into RAM at init.
    Preprocessing (crop, zoom, normalize) is done on the fly in __getitem__.
    Use when the raw dataset fits in memory; you can use num_workers > 0
    to parallelize preprocessing. Same batch format as NeighborsDataset
    (works with collate_neighbors).
    """

    def __init__(self, hdf5_path, norm_dict=NORM_DICT, crop_size=48, max_neighbors=15):
        self.norm_dict = norm_dict
        self.crop_size = crop_size
        self.max_neighbors = max_neighbors

        print(f"Loading raw data from {hdf5_path} into RAM...")
        t_start = time.time()

        with h5py.File(hdf5_path, 'r') as f:
            print("  - Loading images_hsc...")
            self.images_hsc = f["images_hsc"][:]

            print("  - Loading images_legacy...")
            self.images_legacy = f["images_legacy"][:]

            print("  - Loading neighbor indices...")
            self.all_neigh_hsc = f["neighbor_idx_hsc"][:]
            self.all_neigh_legacy = f["neighbor_idx_legacy"][:]

            print("  - Filtering indices...")
            sources = f["source_type"][:]
            indexes_mmu = np.where(sources == 0)[0]

            neigh_hsc_subset = self.all_neigh_hsc[indexes_mmu]
            neigh_legacy_subset = self.all_neigh_legacy[indexes_mmu]

            good_both = (~np.all(neigh_hsc_subset == -1, axis=1)) & (
                ~np.all(neigh_legacy_subset == -1, axis=1)
            )
            self.indexes_mmu = indexes_mmu[good_both]

            self.cached_neighbor_hsc = neigh_hsc_subset[good_both]
            self.cached_neighbor_legacy = neigh_legacy_subset[good_both]

        t_end = time.time()
        print(f"Loaded {len(self.indexes_mmu)} samples. RAM Load Time: {t_end - t_start:.2f}s")
        size_gb = (self.images_hsc.nbytes + self.images_legacy.nbytes) / 1e9
        print(f"Raw Image Arrays Size: ~{size_gb:.2f} GB")

    def __len__(self):
        return len(self.indexes_mmu)

    def __getitem__(self, idx):
        index_mmu = self.indexes_mmu[idx]

        anchor_is_hsc = (idx % 2 == 0)
        anchor_survey = "hsc" if anchor_is_hsc else "legacy"

        img_hsc = self.images_hsc[index_mmu]
        img_legacy = self.images_legacy[index_mmu]

        if anchor_is_hsc:
            target_raw, samegal_raw = img_hsc, img_legacy
            neighbor_ids = self.cached_neighbor_hsc[idx]
            images_source = self.images_hsc
            survey_key, pair_key = "hsc", "legacy"
        else:
            target_raw, samegal_raw = img_legacy, img_hsc
            neighbor_ids = self.cached_neighbor_legacy[idx]
            images_source = self.images_legacy
            survey_key, pair_key = "legacy", "hsc"

        neighbor_ids = neighbor_ids[neighbor_ids != -1][: self.max_neighbors]

        if len(neighbor_ids) > 0:
            neigh_imgs = images_source[neighbor_ids]
            sameins_list = [
                preprocess_raw_image(img, survey_key, self.crop_size, self.norm_dict)
                for img in neigh_imgs
            ]
        else:
            sameins_list = []

        target = preprocess_raw_image(target_raw, survey_key, self.crop_size, self.norm_dict)
        samegal = preprocess_raw_image(samegal_raw, pair_key, self.crop_size, self.norm_dict)

        if anchor_is_hsc:
            target = target[:4]
            sameins = (
                torch.stack(sameins_list, dim=0)[:, :4]
                if sameins_list
                else torch.empty(0, 4, self.crop_size, self.crop_size)
            )
        else:
            samegal = samegal[:4]
            sameins = (
                torch.stack(sameins_list, dim=0)
                if sameins_list
                else torch.empty(0, 3, self.crop_size, self.crop_size)
            )

        metadata = {
            "anchor_survey": anchor_survey,
            "idx": int(idx),
            "num_same_instrument": len(sameins_list),
        }
        return target, samegal, sameins, metadata


# Keys to exclude from NeighborsSimpleDataset metadata (images and neighbor arrays)
NEIGHBORS_SIMPLE_EXCLUDE_KEYS = frozenset({
    "source_type",
    "images_hsc",
    "images_legacy",
    "neighbor_idx_hsc",
    "neighbor_idx_legacy",
    "neighbor_dist_hsc",
    "neighbor_dist_legacy",
})


def _metadata_value_from_h5(val):
    """Convert an HDF5/dataset slice to a Python type for metadata (scalars and small arrays)."""
    if hasattr(val, "shape") and val.shape == ():
        return val.item()
    if np.isscalar(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, np.bool_):
            return bool(val)
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


class NeighborsSimpleDataset(Dataset):
    """
    Returns samples from the HDF5: hsc image, legacy image, and metadata from all columns
    except source_type, images_hsc, images_legacy, and neighbor_* (neighbor_idx_*, neighbor_dist_*).
    """
    def __init__(self, hdf5_path, norm_dict=NORM_DICT, crop_size=48):
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.crop_size = crop_size
        self.file = None  # Handle for lazy loading

        with h5py.File(self.hdf5_path, 'r') as f:
            sources = f['source_type'][:]
            indexes_mmu = np.where(sources == 0)[0]
            self.indexes_mmu = indexes_mmu
            self._meta_keys = [k for k in f.keys() if k not in NEIGHBORS_SIMPLE_EXCLUDE_KEYS]

    def _open_file(self):
        """Opens the HDF5 file once per worker process."""
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)

    def __len__(self):
        return len(self.indexes_mmu)

    def __getitem__(self, idx):
        self._open_file()
        index_mmu = self.indexes_mmu[idx]

        img_hsc = self.file['images_hsc'][index_mmu]
        img_legacy = self.file['images_legacy'][index_mmu]

        img_hsc = preprocess_raw_image(img_hsc, 'hsc', self.crop_size, self.norm_dict)
        img_legacy = preprocess_raw_image(img_legacy, 'legacy', self.crop_size, self.norm_dict)

        img_hsc = img_hsc[:4]
        img_legacy = img_legacy[:4]

        metadata = {"idx": idx, "index_mmu": int(index_mmu)}
        for key in self._meta_keys:
            val = self.file[key][index_mmu]
            metadata[key] = _metadata_value_from_h5(val)

        return img_hsc, img_legacy, metadata


if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    # neighbors_dataset = NeighborsDataset(
    #     hdf5_path="/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5",
    # )
    # batch_size = 8
    # num_warmup = 25
    # num_measure = 100

    # loader = DataLoader(
    #     neighbors_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=collate_neighbors,
    #     persistent_workers=True,
    #     pin_memory=True,
    # )
    # it = iter(loader)
    # for _ in range(num_warmup):
    #     next(it)
    # start = time.time()
    # for _ in tqdm(range(num_measure)):
    #     next(it)
    # elapsed = time.time() - start
    # print(f"Average time per batch (steady state): {elapsed / num_measure:.4f}s")


    #######################################################
    # TESTING THE MEMORY LOADING OPTION
        # CONFIGURATION
    # Use the MERGED file path here
    # H5_PATH = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5"
    H5_PATH = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"
    BATCH_SIZE = 64

    # 1. Init Dataset (Loads to RAM)
    dataset = NeighborsPrecomputedDataset(H5_PATH)

    # 2. Init Loader
    # Note: num_workers=0 is usually FASTER for in-memory datasets because
    # it avoids pickling overhead between processes.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # We can shuffle freely now!
        num_workers=0,
        collate_fn=simple_collate
    )

    # Visualize first batch
    first_batch = next(iter(loader))
    plot_option2_first_batch(
        first_batch,
        save_path="neighbors_option2_first_batch.png",
        num_samples=4,
        max_neighbors_show=5,
    )

    num_batches_total = len(loader)  # may be small if using a single shard
    NUM_BATCHES = min(100, num_batches_total)
    print(f"\nStarting benchmark with Batch Size {BATCH_SIZE} ({num_batches_total} batches in dataset, timing {NUM_BATCHES})...")

    # Warmup (don't exceed available batches)
    iter_loader = iter(loader)
    for _ in range(min(5, num_batches_total)):
        next(iter_loader)

    # Timing: run up to NUM_BATCHES or until iterator is exhausted
    t_start = time.time()
    count = 0
    for i in tqdm(range(NUM_BATCHES), desc="Benchmark"):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        count += 1
        # simulate transfer to GPU
        # [x.cuda() for x in batch if isinstance(x, torch.Tensor)]

    t_end = time.time()
    if count == 0:
        print("No batches to time.")
    else:
        total_samples = count * BATCH_SIZE
        fps = total_samples / (t_end - t_start)
        print(f"\nDone.")
        print(f"Throughput: {fps:.1f} samples/sec")
        print(f"Time per batch: {(t_end - t_start) / count * 1000:.2f} ms")


    # #########################################################
    # # TESTING OPTION 3: NeighborsDatasetRawRAM
    # SOURCE_H5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
    # # Or test_neighbours.h5 for a quick check
    # print("Initializing NeighborsDatasetRawRAM...")
    # try:
    #     dataset = NeighborsDatasetRawRAM(hdf5_path=SOURCE_H5, max_neighbors=5)
    # except Exception as e:
    #     print(f"Error loading dataset: {e}")
    #     sys.exit(1)

    # BATCH_SIZE = 64
    # NUM_WORKERS = 4
    # loader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=NUM_WORKERS,
    #     collate_fn=collate_neighbors,
    #     persistent_workers=(NUM_WORKERS > 0),
    # )

    # print(f"\nStarting benchmark on {len(dataset)} samples with {NUM_WORKERS} workers...")
    # iter_loader = iter(loader)
    # for _ in range(5):
    #     next(iter_loader)

    # t_start = time.time()
    # count = 0
    # for i, batch in enumerate(tqdm(loader)):
    #     count += 1
    #     if i >= 99:
    #         break
    # t_end = time.time()
    # total_samples = count * BATCH_SIZE
    # duration = t_end - t_start
    # print(f"\n--- Results ---")
    # print(f"Throughput: {total_samples / duration:.1f} samples/sec")
    # print(f"Time per batch: {(duration / count) * 1000:.2f} ms")
