"""
Explore Euclid and COSMOS cutout statistics to calibrate BAND_CENTER_MAX.

In a first step, this script only works with the VIS band (Euclid) and F115W band (COSMOS).
TODO: add other bands (e.g., Y, J, H for Euclid and F150W, F277W for COSMOS).

Edit the CONFIG block below to match your directory layout and band suffixes,
then run:
    python experiments/euclid-cosmos/explore_data.py
"""

import os
import glob
import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------------------
# CONFIG — edit these paths and patterns to match your data layout
# ---------------------------------------------------------------------------

EUCLID_DIR_VIS = "/n03data/fontirro/euclid/40_cutouts/40_cutouts-vis"  # directory containing Euclid FITS files
COSMOS_DIR_F115W = "/n03data/fontirro/cosmos/120_cutouts/f115w"  # directory containing COSMOS FITS files

# glob pattern to find Euclid files inside their respective directories
EUCLID_PATTERN = "*.fits"

 # glob pattern to find COSMOS files.
COSMOS_PATTERN = "*.fits"            

# How many files to sample per survey (set to None to use all)
N_SAMPLE = 200

# ---------------------------------------------------------------------------


def load_euclid(path: str) -> np.ndarray:
    """Load a single Euclid cutout from a FITS file.
    path: full path to the FITS file
    Returns: (1, H, W) array of pixel values (1 channel, height, width)
    """
    with fits.open(path) as hdul:
        data = hdul[1].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]  # (H, W) -> (1, H, W) where H is height and W is width.
    return data


def load_cosmos(path: str) -> np.ndarray:
    """Load F115W band COSMOS galaxy from a FITS files.
    path: full path to the F115W band FITS file
    Returns: (1, H, W) array of pixel values (1 channel, height, width)
    """
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]  # (H, W) -> (1, H, W) where H is height and W is width.
    return data



def channel_stats(stack: np.ndarray) -> dict:
    """Return per-channel statistics for an (N, C, H, W) array, where:
    N = number of samples, C = number of channels, H = height, W = width.
    stats: 
    min: min pixel value, 
    max: max pixel value, 
    p99: 99th percentile pixel value (i.e. the bright tail, typical bright pixel values), 
    p99.9: 99.9th percentile pixel value (i.e the near-maximum excluding outliers like cosmic rays 
                    — this is what gets used to set BAND_CENTER_MAX), 
    p0.1: 0.1th percentile pixel value i.e. the faint/negative tail (useful for spotting bad pixels).
    """
    n, c = stack.shape[:2]
    stats = {}
    for ci in range(c):
        flat = stack[:, ci].ravel()
        finite = flat[np.isfinite(flat)]
        stats[ci] = {
            "min":   float(np.min(finite)),
            "max":   float(np.max(finite)),
            "p99":   float(np.percentile(finite, 99)),
            "p99.9": float(np.percentile(finite, 99.9)),
            "p0.1":  float(np.percentile(finite, 0.1)),
        }
    return stats


def sample_files(files: list[str], n: int | None) -> list[str]:
    """Randomly sample n files from the list (or return all if n is None or too large).
    """
    if n is None or n >= len(files):
        return files
    rng = np.random.default_rng(42)
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[i] for i in sorted(idx)]


def main():
    # --- Euclid ---
    #reading each file from directory and sorting them.
    euclid_files = sorted(glob.glob(os.path.join(EUCLID_DIR_VIS, EUCLID_PATTERN)))
    print(f"Found {len(euclid_files)} Euclid files.")

    #randomly sample files if N_SAMPLE is set.
    euclid_files = sample_files(euclid_files, N_SAMPLE)

    #adding N_SAMPLE files' pixel values to stack.
    euclid_stack = []
    for f in euclid_files:
        try:
            euclid_stack.append(load_euclid(f)) #remember that load_euclid adds one channel dimension, so each entry is (1, H, W)
        except Exception as e:
            print(f"  [WARN] skipping {f}: {e}")
    euclid_stack = np.stack(euclid_stack, axis=0)  # (N, 1, H, W)
    print(euclid_stack)
    print(f"Euclid array shape: {euclid_stack.shape}")

    
    # --- COSMOS ---
    cosmos_files = sorted(glob.glob(os.path.join(COSMOS_DIR_F115W, COSMOS_PATTERN)))
    print(f"\nFound {len(cosmos_files)} COSMOS files")
    cosmos_files = sample_files(cosmos_files, N_SAMPLE)

    cosmos_stack = []
    for f in cosmos_files:
        try:
            cosmos_stack.append(load_cosmos(f)) #remember that load_cosmos adds one channel dimension, so each entry is (1, H, W).
        except Exception as e:
            print(f"  [WARN] skipping {f}: {e}")
    cosmos_stack = np.stack(cosmos_stack, axis=0)  # (N, 1, H, W)
    print(f"COSMOS array shape: {cosmos_stack.shape}")

    # --- Report ---
    print("\n" + "=" * 55)
    print("EUCLID  (1 channel - VIS band)")
    print("=" * 55)
    euclid_stats = channel_stats(euclid_stack)
    for ci, s in euclid_stats.items():
        print(f"  ch{ci}:  min={s['min']:.4g}  max={s['max']:.4g}"
                f"  p0.1={s['p0.1']:.4g}  p99={s['p99']:.4g}  p99.9={s['p99.9']:.4g}")

    print("\n" + "=" * 55)
    print("COSMOS  (1 channels - F115W band)")
    print("=" * 55)
    cosmos_stats = channel_stats(cosmos_stack)
    for ci, s in cosmos_stats.items():
        print(f"  ch{ci}:  min={s['min']:.4g}  max={s['max']:.4g}"
              f"  p0.1={s['p0.1']:.4g}  p99={s['p99']:.4g}  p99.9={s['p99.9']:.4g}")

    print("\n" + "=" * 55)
    print("Suggested BAND_CENTER_MAX  (based on p99.9)")
    print("Set these in image_preprocessing.py")
    print("=" * 55)
    for ci, s in euclid_stats.items():
        print(f'  "EUCLID": {s["p99.9"]:.4g},')
    for ci, s in cosmos_stats.items():
        print(f'  "COSMOS": {s["p99.9"]:.4g},')


if __name__ == "__main__":
    main()
