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

EUCLID_COL = "file_euclid_vis"              # column name for the Euclid FITS file path
COSMOS_COL = "file_cosmos_f115w"            # column name for the COSMOS FITS file path

EUCLID_EXISTS_COL = "cutout_euc_40_vis"    # boolean column: True if Euclid cutout exists
COSMOS_EXISTS_COL = "cutout_cos_120_115w"  # boolean column: True if COSMOS cutout exists

EUCLID_DIR_PATH = { path_vis: "/n03data/fontirro/euclid/40_cutouts/40_cutouts-vis/"  # base directory for Euclid VIS cutouts.
}

COSMOS_DIR_PATH = { path_f115w: "/n03data/fontirro/cosmos/120_cutouts/f115w/",  # base directory for COSMOS F115W cutouts.
                    path_f150w: "/n03data/fontirro/cosmos/120_cutouts/f150w/"
}


EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension)
COSMOS_HDU = 0   # HDU index for COSMOS data (usually 0)

OUTPUT_H5 = "/n03data/fontirro/data_files/euclid_cosmos_pairs_v2.h5"

NUM_WORKERS = 16  # parallel threads for loading + preprocessing

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
import torch.nn.functional as F
from image_preprocessing import preprocess_image_v2


def load_fits(path: str, hdu: int) -> torch.Tensor:
    """Load a FITS file and return a (1, 1, H, W) float32 tensor."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]
    return torch.from_numpy(data).unsqueeze(0)  # (1, 1, H, W)


def get_spatial_size(path: str, hdu: int) -> tuple[int, int]:
    """Return (H, W) of the first valid file."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data
    return data.shape[-2], data.shape[-1]


def process_pair(args: tuple) -> tuple:
    """Load, preprocess, and downscale one pair. Returns (i, euc, cos, cos_down, error)."""
    i, ep, cp, h_euc, w_euc = args
    try:
        euc_tensor = load_fits(ep, EUCLID_HDU)
        cos_tensor = load_fits(cp, COSMOS_HDU)
        euc = preprocess_image_v2(euc_tensor, bands=["VIS"]).squeeze(0).numpy()
        cos = preprocess_image_v2(cos_tensor, bands=["F115W"]).squeeze(0).numpy()
        cos_down = F.interpolate(
            torch.from_numpy(cos).unsqueeze(0), size=(h_euc, w_euc),
            mode="bilinear", align_corners=False,
        ).squeeze(0).numpy()
        return i, euc, cos, cos_down, None
    except Exception as e:
        return i, None, None, None, str(e)


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Catalog loaded: {len(catalog)} pairs")
    #print(f"Columns: {list(catalog.columns)}")

    mask = (catalog[EUCLID_EXISTS_COL] == 1) & (catalog[COSMOS_EXISTS_COL] == 1)
    catalog = catalog[mask]
    print(f"Pairs with both cutouts present: {len(catalog)} / {len(mask)}")

    euclid_paths = [os.path.join(EUCLID_DIR_PATH, p) for p in catalog[EUCLID_COL]]
    cosmos_paths = [os.path.join(COSMOS_DIR_PATH, p) for p in catalog[COSMOS_COL]]
    N = len(euclid_paths)

    H_euc, W_euc = get_spatial_size(euclid_paths[0], EUCLID_HDU)
    H_cos, W_cos = get_spatial_size(cosmos_paths[0], COSMOS_HDU)
    print(f"Euclid image size : {H_euc} x {W_euc}")
    print(f"COSMOS image size : {H_cos} x {W_cos}")

    args_list = [(i, ep, cp, H_euc, W_euc) for i, (ep, cp) in enumerate(zip(euclid_paths, cosmos_paths))]

    with h5py.File(OUTPUT_H5, "w") as f:
        euc_ds = f.create_dataset("euclid_images", shape=(N, 1, H_euc, W_euc), dtype=np.float32)
        cos_ds = f.create_dataset("cosmos_images", shape=(N, 1, H_cos, W_cos), dtype=np.float32)
        cos_down_ds = f.create_dataset("cosmos_images_downscaled", shape=(N, 1, H_euc, W_euc), dtype=np.float32)
        cat_grp = f.create_group("catalog")
        dt = h5py.string_dtype()
        cat_grp.create_dataset("euclid_paths", data=np.array(euclid_paths, dtype=object), dtype=dt)
        cat_grp.create_dataset("cosmos_paths", data=np.array(cosmos_paths, dtype=object), dtype=dt)
        f.attrs["num_pairs"] = N
        f.attrs["num_channels"] = 1
        f.attrs["euclid_shape"] = [H_euc, W_euc]
        f.attrs["cosmos_shape"] = [H_cos, W_cos]

        skipped = 0
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = executor.map(process_pair, args_list, chunksize=100)
            for result in tqdm(results, total=N, desc="Processing", mininterval=60, dynamic_ncols=False):
                i, euc, cos, cos_down, err = result
                if err:
                    print(f"\n  [WARN] skipping pair {i}: {err}")
                    skipped += 1
                    continue
                euc_ds[i] = euc
                cos_ds[i] = cos
                cos_down_ds[i] = cos_down

    print(f"\nDone. {N - skipped}/{N} pairs written to {OUTPUT_H5}")
    if skipped:
        print(f"  {skipped} pairs skipped due to errors.")


if __name__ == "__main__":
    main()
