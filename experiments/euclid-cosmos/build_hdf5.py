"""
Build an HDF5 file with paired Euclid (VIS) and COSMOS (F115W) cutouts.

The native spatial size of the first file in each instrument is used to
pre-allocate the datasets. All images of the same instrument are assumed
to share the same size. Datasets follow the (N, C, H, W) layout from
new_dataset_guide.md.

HDF5 layout:
    euclid_images         — (N, 1, H_euc, W_euc) float32
    cosmos_images         — (N, 2, H_cos, W_cos) float32 #we add now one more filter: F150W
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

CATALOG_PATH = "/n03data/fontirro/data_files/cat_crossmatch_mag27_mag25.csv"  # path to the paired catalog

EUCLID_COL = "file_euclid_vis"              # column name for the Euclid FITS file path
COSMOS_COL = "file_cosmos_f115w"            # column name for the COSMOS FITS file path

EUCLID_EXISTS_COL = ["cutout_euc_40_vis"]    # boolean column: True if Euclid cutout exists
COSMOS_EXISTS_COL = ["cutout_cos_120_115w", "cutout_cos_120_150w"]  # boolean column: True if COSMOS cutout exists

EUCLID_DIR_PATH = {
    'path_vis': "/n03data/fontirro/cutouts/euclid/40_cutouts/40_cutouts-vis/"  # base directory for Euclid VIS cutouts.
}

COSMOS_DIR_PATH = {
    'path_f115w': "/n03data/fontirro/cutouts/cosmos/120_cutouts/f115w/",  # base directory for COSMOS F115W cutouts.
    'path_f150w': "/n03data/fontirro/cutouts/cosmos/120_cutouts/f150w/"
}


EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension)
COSMOS_HDU = 0   # HDU index for COSMOS data (usually 0)

OUTPUT_H5 = "/n03data/fontirro/data_files/euclid_cosmos_pairs_v2.h5"

NUM_WORKERS = 16  # parallel threads for loading + preprocessing

H_SIZE = 64  # target spatial size for both Euclid and COSMOS (COSMOS will be downscaled to match Euclid)
W_SIZE = 64  # target spatial size for both Euclid and COSMOS (COSMOS will be downscaled to match Euclid)
# ---------------------------------------------------------------------------

import traceback
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
    """Load, preprocess, and downscale one pair. Returns (i, euc, cos, cos_down, euc_up, error).
    cos and cos_down are (2, H, W) arrays with F115W and F150W stacked."""
    i, ep, cp_f115w, cp_f150w = args
    try:

        euc_tensor = load_fits(ep, EUCLID_HDU)
        cos_f115w = preprocess_image_v2(load_fits(cp_f115w, COSMOS_HDU), bands=["F115W"]).squeeze(0).numpy()
        cos_f150w = preprocess_image_v2(load_fits(cp_f150w, COSMOS_HDU), bands=["F150W"]).squeeze(0).numpy()
        euc = preprocess_image_v2(euc_tensor, bands=["VIS"]).squeeze(0).numpy()
        cos = np.concatenate([cos_f115w, cos_f150w], axis=0)  # (2, H, W)
        cos_down = F.interpolate(
            torch.from_numpy(cos).unsqueeze(0), size=(H_SIZE, W_SIZE),
            mode="area",
        ).squeeze(0).numpy()  # (2, H_size, W_size)
        euc_up = F.interpolate(
            torch.from_numpy(euc).unsqueeze(0), size=(H_SIZE, W_SIZE),
            mode="bilinear", align_corners=True,
        ).squeeze(0).numpy()  # (1, H_size, W_size)

        return i, euc, cos, cos_down, euc_up, None

    except Exception:
        return i, None, None, None, None, traceback.format_exc()


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Catalog loaded: {len(catalog)} pairs")
    #print(f"Columns: {list(catalog.columns)}")

    euclid_mask = catalog[EUCLID_EXISTS_COL] if isinstance(EUCLID_EXISTS_COL, str) else catalog[EUCLID_EXISTS_COL].all(axis=1)
    cosmos_mask = catalog[COSMOS_EXISTS_COL] if isinstance(COSMOS_EXISTS_COL, str) else catalog[COSMOS_EXISTS_COL].all(axis=1)
    mask = euclid_mask & cosmos_mask #although the sample already has been checked, I'd like to keep this just to be sure.
    catalog = catalog[mask]
    print(f"Pairs with both cutouts present: {len(catalog)} / {len(mask)}")

    euclid_paths = [os.path.join(EUCLID_DIR_PATH['path_vis'], p) for p in catalog[EUCLID_COL]]
    cosmos_paths_f115w = [os.path.join(COSMOS_DIR_PATH['path_f115w'], p) for p in catalog[COSMOS_COL]]
    cosmos_paths_f150w = [os.path.join(COSMOS_DIR_PATH['path_f150w'], p) for p in catalog["file_cosmos_f150w"]]
    N = len(euclid_paths)

    print(f"First Euclid path: {euclid_paths[0]}")
    print(f"First COSMOS F115W path: {cosmos_paths_f115w[0]}")
    print(f"First COSMOS F150W path: {cosmos_paths_f150w[0]}")

    print("\nEuclid FITS structure (first file):")
    with fits.open(euclid_paths[0]) as hdul:
        hdul.info()
        for k, hdu in enumerate(hdul):
            if hdu.data is not None:
                d = hdu.data.astype(np.float32)
                print(f"  HDU {k}: shape={d.shape}  range=[{d.min():.4f}, {d.max():.4f}]  nonzero={np.count_nonzero(d)}")

    print("\nSampling Euclid files for zero-data rate:")
    sample_idx = np.linspace(0, N - 1, min(20, N), dtype=int)
    zero_count = 0
    for si in sample_idx:
        with fits.open(euclid_paths[si]) as hdul:
            d = hdul[EUCLID_HDU].data.astype(np.float32)
            nz = np.count_nonzero(d)
            if nz == 0:
                zero_count += 1
    print(f"  {zero_count}/{len(sample_idx)} sampled files are all-zero")
    if zero_count == len(sample_idx):
        print("  [WARN] All sampled Euclid files are zero — the cutout files may be empty.")
        print("  Check EUCLID_HDU or re-examine how the cutouts were generated.")

    print("\nTesting first pair (sequential)...")
    _, t_euc, t_cos, t_cos_down, t_euc_up, t_err = process_pair((0, euclid_paths[0], cosmos_paths_f115w[0], cosmos_paths_f150w[0]))
    if t_err:
        print(f"[ERROR] First pair failed:\n{t_err}")
        sys.exit(1)
    print(f"  euc      shape={t_euc.shape}  range=[{t_euc.min():.4f}, {t_euc.max():.4f}]")
    print(f"  euc_up   shape={t_euc_up.shape}  range=[{t_euc_up.min():.4f}, {t_euc_up.max():.4f}]")
    print(f"  cos      shape={t_cos.shape}  range=[{t_cos.min():.4f}, {t_cos.max():.4f}]")
    print(f"  cos_down shape={t_cos_down.shape}  range=[{t_cos_down.min():.4f}, {t_cos_down.max():.4f}]")
    print("First pair OK.\n")

    H_euc, W_euc = get_spatial_size(euclid_paths[0], EUCLID_HDU)
    H_cos, W_cos = get_spatial_size(cosmos_paths_f115w[0], COSMOS_HDU)
    print(f"Euclid image size : {H_euc} x {W_euc}")
    print(f"COSMOS image size : {H_cos} x {W_cos}")

    return

    args_list = [
        (i, ep, cp115, cp150)
        for i, (ep, cp115, cp150) in enumerate(zip(euclid_paths, cosmos_paths_f115w, cosmos_paths_f150w))
    ]

    with h5py.File(OUTPUT_H5, "w") as f:
        euc_ds = f.create_dataset("euclid_images", shape=(N, 1, H_euc, W_euc), dtype=np.float32)
        cos_ds = f.create_dataset("cosmos_images", shape=(N, 2, H_cos, W_cos), dtype=np.float32)
        cos_down_ds = f.create_dataset("cosmos_images_downscaled", shape=(N, 2, H_SIZE, W_SIZE), dtype=np.float32)
        euc_up_ds = f.create_dataset("euclid_images_upscaled", shape=(N, 1, H_SIZE, W_SIZE), dtype=np.float32)
        cat_grp = f.create_group("catalog")
        dt = h5py.string_dtype()
        cat_grp.create_dataset("euclid_paths", data=np.array(euclid_paths, dtype=object), dtype=dt)
        cat_grp.create_dataset("cosmos_paths_f115w", data=np.array(cosmos_paths_f115w, dtype=object), dtype=dt)
        cat_grp.create_dataset("cosmos_paths_f150w", data=np.array(cosmos_paths_f150w, dtype=object), dtype=dt)
        f.attrs["num_pairs"] = N
        f.attrs["num_channels"] = 2
        f.attrs["euclid_shape"] = [H_euc, W_euc]
        f.attrs["cosmos_shape"] = [H_cos, W_cos]
        f.attrs["cosmos_downscaled_shape"] = [H_SIZE, W_SIZE]

        skipped = 0
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = executor.map(process_pair, args_list, chunksize=100)
            for result in tqdm(results, total=N, desc="Processing", mininterval=60, dynamic_ncols=False):
                i, euc, cos, cos_down, euc_up, err = result
                if err:
                    print(f"\n  [WARN] skipping pair {i}: {err}")
                    skipped += 1
                    continue
                euc_ds[i] = euc
                cos_ds[i] = cos
                cos_down_ds[i] = cos_down
                euc_up_ds[i] = euc_up
    print(f"\nDone. {N - skipped}/{N} pairs written to {OUTPUT_H5}")
    if skipped:
        print(f"  {skipped} pairs skipped due to errors.")


if __name__ == "__main__":
    main()
