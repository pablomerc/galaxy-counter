# Easy plot a galaxy pair from the HDF5 file to check they look correct.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

H5_PATH = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"


def plot_pair(h5_path: str, idx: int):
    with h5py.File(h5_path, "r") as f:
        euc_data = f["euclid_images"][idx, 0]
        cos_data = f["cosmos_images_downscaled"][idx, 0]
        euc_path = f["catalog/euclid_paths"][idx].decode()
        cos_path = f["catalog/cosmos_paths"][idx].decode()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(euc_data, cmap="gray")
    axes[0].set_title(f"Euclid  (idx={idx})\n{euc_path}")
    axes[1].imshow(cos_data, cmap="gray")
    axes[1].set_title(f"COSMOS  (idx={idx})\n{cos_path}")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot a galaxy pair from the HDF5 file.")
    parser.add_argument("--h5", default=H5_PATH, help="Path to the HDF5 file")
    parser.add_argument("--idx", type=int, default=0, help="Pair index to plot")
    args = parser.parse_args()

    print(f"Plotting pair {args.idx} from {args.h5}")
    plot_pair(args.h5, args.idx)


if __name__ == "__main__":
    main()
