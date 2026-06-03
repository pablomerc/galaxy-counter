# Easy plot a galaxy pair from the HDF5 file to check they look correct.

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.visualization import astropy_mpl_style, ImageNormalize, PercentileInterval, AsinhStretch


H5_PATH = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
OUT_PATH = "/n03data/fontirro/output_plots/"

def plot_pair(h5_path: str, idx: int, out_path: str):
    with h5py.File(h5_path, "r") as f:
        euc_data = f["euclid_images"][idx, 0]
        cos_data_down = f["cosmos_images_downscaled"][idx, 0]
        cos_data = f["cosmos_images"][idx, 0]
        euc_path = f["catalog/euclid_paths"][idx].decode()
        cos_path = f["catalog/cosmos_paths"][idx].decode()


    id_cos = cos_path.split("/")[-1].replace(".fits", "")
    id_euc = euc_path.split("/")[-1].split("_")[-2].replace("_cutout.fits", "")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(euc_data, cmap="gray")
    axes[0].set_title(f"Euclid  (idx={idx})\n{id_euc}")
    axes[1].imshow(cos_data_down, cmap="gray")
    axes[1].set_title(f"COSMOS (downscaled)  (idx={idx})\n{id_cos}")
    axes[2].imshow(cos_data, cmap="gray", norm=ImageNormalize(cos_data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
    axes[2].set_title(f"COSMOS  (idx={idx})\n{id_cos}")
    plt.tight_layout()

    out_file = os.path.join(out_path, f"pair_{idx}_{id_euc}_{id_cos}.png")

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot a galaxy pair from the HDF5 file.")
    parser.add_argument("--h5", default=H5_PATH, help="Path to the HDF5 file")
    parser.add_argument("--idx", type=int, default=0, help="Pair index to plot")
    parser.add_argument("--out_path", default=OUT_PATH, help="Directory to save the plot")
    args = parser.parse_args()

    print(f"Plotting pair {args.idx} from {args.h5}")
    plot_pair(args.h5, args.idx, args.out_path)


if __name__ == "__main__":
    main()
