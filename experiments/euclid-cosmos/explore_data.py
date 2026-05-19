"""
Explore Euclid and COSMOS cutout statistics to calibrate BAND_CENTER_MAX.

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

EUCLID_DIR = "/n03data/fontirro/euclid/40_cutouts"  # directory containing Euclid FITS files
COSMOS_DIR = "/n03data/fontirro/cosmos/120_cutouts"  # directory containing COSMOS FITS files

# Glob pattern to find Euclid files inside their respective directories
EUCLID_PATTERN = "*.fits"

# Each COSMOS galaxy has one file per band. List the band suffixes so the script
# can reconstruct all three filenames from a single base name.
# Example: if your files are galaxy_001_g.fits, galaxy_001_r.fits, galaxy_001_i.fits
# set COSMOS_BAND_SUFFIXES = ["_g", "_r", "_i"] and COSMOS_PATTERN = "*_g.fits"
COSMOS_BAND_SUFFIXES = ["F115W", "F150W", "F277W"]   # suffixes before .fits
COSMOS_PATTERN = "F115W_*.fits"                  # anchor band to glob for unique galaxies

# How many files to sample per survey (set to None to use all)
N_SAMPLE = 200

# Which HDU index to read from (0 = primary; adjust if your data uses extensions)
HDU_INDEX = 0

# ---------------------------------------------------------------------------


def load_euclid(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[1].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]  # (H, W) -> (1, H, W) where H is height and W is width.
    return data


def load_cosmos(base_path: str, suffixes: list[str]) -> np.ndarray:
    """Load one multi-band COSMOS galaxy from separate per-band FITS files."""
    stem = base_path[: base_path.rfind(suffixes[0])]  # strip anchor suffix
    bands = []
    for suf in suffixes:
        p = stem + suf + ".fits"
        with fits.open(p) as hdul:
            ch = hdul[0].data.astype(np.float32)
        if ch.ndim == 3:
            ch = ch[0]  # drop degenerate leading axis if present
        bands.append(ch)
    return np.stack(bands, axis=0)  # (3, H, W)


def channel_stats(stack: np.ndarray) -> dict:
    """Return per-channel statistics for an (N, C, H, W) array."""
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
    if n is None or n >= len(files):
        return files
    rng = np.random.default_rng(42)
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[i] for i in sorted(idx)]


def main():
    # --- Euclid ---
    euclid_files = sorted(glob.glob(os.path.join(EUCLID_DIR, EUCLID_PATTERN)))
    print(f"Found {len(euclid_files)} Euclid files.")
    euclid_files = sample_files(euclid_files, N_SAMPLE)

    euclid_stack = []
    for f in euclid_files:
        try:
            euclid_stack.append(load_euclid(f))
        except Exception as e:
            print(f"  [WARN] skipping {f}: {e}")
    euclid_stack = np.stack(euclid_stack, axis=0)  # (N, 1, H, W)
    print(f"Euclid array shape: {euclid_stack.shape}")

    # --- COSMOS ---
    cosmos_anchors = sorted(glob.glob(os.path.join(COSMOS_DIR, COSMOS_PATTERN)))
    print(f"\nFound {len(cosmos_anchors)} COSMOS galaxies (anchored on '{COSMOS_PATTERN}').")
    cosmos_anchors = sample_files(cosmos_anchors, N_SAMPLE)

    cosmos_stack = []
    for f in cosmos_anchors:
        try:
            cosmos_stack.append(load_cosmos(f, COSMOS_BAND_SUFFIXES))
        except Exception as e:
            print(f"  [WARN] skipping {f}: {e}")
    cosmos_stack = np.stack(cosmos_stack, axis=0)  # (N, 3, H, W)
    print(f"COSMOS array shape: {cosmos_stack.shape}")

    # --- Report ---
    print("\n" + "=" * 55)
    print("EUCLID  (1 channel)")
    print("=" * 55)
    euclid_stats = channel_stats(euclid_stack)
    for ci, s in euclid_stats.items():
        print(f"  ch{ci}:  min={s['min']:.4g}  max={s['max']:.4g}"
              f"  p0.1={s['p0.1']:.4g}  p99={s['p99']:.4g}  p99.9={s['p99.9']:.4g}")

    print("\n" + "=" * 55)
    print("COSMOS  (3 channels — indices match COSMOS_BAND_SUFFIXES order)")
    print("=" * 55)
    cosmos_stats = channel_stats(cosmos_stack)
    for ci, s in cosmos_stats.items():
        suf = COSMOS_BAND_SUFFIXES[ci] if ci < len(COSMOS_BAND_SUFFIXES) else f"ch{ci}"
        print(f"  {suf}:  min={s['min']:.4g}  max={s['max']:.4g}"
              f"  p0.1={s['p0.1']:.4g}  p99={s['p99']:.4g}  p99.9={s['p99.9']:.4g}")

    print("\n" + "=" * 55)
    print("Suggested BAND_CENTER_MAX  (based on p99.9)")
    print("Set these in image_preprocessing.py")
    print("=" * 55)
    for ci, s in euclid_stats.items():
        print(f'  "EUCLID": {s["p99.9"]:.4g},')
    for ci, s in cosmos_stats.items():
        suf = COSMOS_BAND_SUFFIXES[ci] if ci < len(COSMOS_BAND_SUFFIXES) else f"COSMOS-ch{ci}"
        key = f"COSMOS{suf.upper()}"
        print(f'  "{key}": {s["p99.9"]:.4g},')


if __name__ == "__main__":
    main()
