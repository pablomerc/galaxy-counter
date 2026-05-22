"""
Build an HDF5 file with paired Euclid (VIS) and COSMOS (F115W) cutouts.

The native spatial size of the first file in each instrument is used to
pre-allocate the datasets. All images of the same instrument are assumed
to share the same size. Datasets follow the (N, C, H, W) layout from
new_dataset_guide.md.

HDF5 layout:
    euclid_images         — (N, 1, H_euc, W_euc) float32
    cosmos_images         — (N, 1, H_cos, W_cos) float32
    catalog/euclid_paths  — string array (N,)
    catalog/cosmos_paths  — string array (N,)
    attrs: num_pairs, num_channels, euclid_shape, cosmos_shape

Preprocessing (via preprocess_image_v2):
    Euclid : ZP rescaling (ZP 26.2 → 23.9), no range compression
    COSMOS : no ZP rescaling (already at 23.9), range compression applied

Edit the CONFIG block below, then run:
    python experiments/euclid-cosmos/build_hdf5.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

CATALOG_PATH = "/n03data/fontirro/data_files/cat_crossmatch_mag27.csv"  # path to the paired catalog

EUCLID_COL = "file_euclid_vis"    # column name for the Euclid FITS file path
COSMOS_COL = "file_cosmos_f115w"    # column name for the COSMOS FITS file path

EUCLID_DIR_PATH = "/n03data/fontirro/euclid/40_cutouts/40_cutouts-vis/"  # base directory for Euclid VIS cutouts.
COSMOS_DIR_PATH = "/n03data/fontirro/cosmos/cosmos/120_cutouts/f115w/"  # base directory for COSMOS F115W cutouts.

EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension)
COSMOS_HDU = 0   # HDU index for COSMOS data (usually 0)

OUTPUT_H5 = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
import torch

from image_preprocessing import preprocess_image_v2


def load_fits(path: str, hdu: int) -> torch.Tensor:
    """Load a FITS file and return a (1, 1, H, W) float32 tensor."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]        # (H, W) → (1, H, W)
    return torch.from_numpy(data).unsqueeze(0)  # (1, 1, H, W)


def get_spatial_size(path: str, hdu: int) -> tuple[int, int]:
    """Return (H, W) of the first valid file."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data
    return data.shape[-2], data.shape[-1]


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Catalog loaded: {len(catalog)} pairs")
    print(f"Columns: {list(catalog.columns)}")

    euclid_paths = [os.path.join(EUCLID_DIR_PATH, p) for p in catalog[EUCLID_COL]]
    cosmos_paths = [os.path.join(COSMOS_DIR_PATH, p) for p in catalog[COSMOS_COL]]
    N = len(euclid_paths)

    # Determine spatial sizes from first file of each instrument
    H_euc, W_euc = get_spatial_size(euclid_paths[0], EUCLID_HDU)
    H_cos, W_cos = get_spatial_size(cosmos_paths[0], COSMOS_HDU)
    print(f"Euclid image size : {H_euc} × {W_euc}")
    print(f"COSMOS image size : {H_cos} × {W_cos}")

    with h5py.File(OUTPUT_H5, "w") as f:
        euc_ds = f.create_dataset(
            "euclid_images", shape=(N, 1, H_euc, W_euc),
            dtype=np.float32, compression="gzip", compression_opts=4,
        )
        cos_ds = f.create_dataset(
            "cosmos_images", shape=(N, 1, H_cos, W_cos),
            dtype=np.float32, compression="gzip", compression_opts=4,
        )
        cat_grp = f.create_group("catalog")
        dt = h5py.string_dtype()
        cat_grp.create_dataset("euclid_paths", data=np.array(euclid_paths, dtype=object), dtype=dt)
        cat_grp.create_dataset("cosmos_paths", data=np.array(cosmos_paths, dtype=object), dtype=dt)
        f.attrs["num_pairs"] = N
        f.attrs["num_channels"] = 1
        f.attrs["euclid_shape"] = [H_euc, W_euc]
        f.attrs["cosmos_shape"] = [H_cos, W_cos]

        skipped = 0
        for i, (ep, cp) in enumerate(zip(euclid_paths, cosmos_paths)):
            if i % 100 == 0:
                print(f"  [{i}/{N}] processing...")

            try:
                euc_tensor = load_fits(ep, EUCLID_HDU)
                cos_tensor = load_fits(cp, COSMOS_HDU)

                euc_processed = preprocess_image_v2(euc_tensor, bands=["VIS"])    # ZP rescale only
                cos_processed = preprocess_image_v2(cos_tensor, bands=["F115W"])  # range compress only
            except Exception as e:
                print(f"  [WARN] skipping pair {i}: {e}")
                skipped += 1
                continue

            # store: squeeze batch dim (1, 1, H, W) → (1, H, W)
            euc_ds[i] = euc_processed.squeeze(0).numpy()
            cos_ds[i] = cos_processed.squeeze(0).numpy()

    print(f"\nDone. {N - skipped}/{N} pairs written to {OUTPUT_H5}")
    if skipped:
        print(f"  {skipped} pairs skipped due to errors.")


if __name__ == "__main__":
    main()
