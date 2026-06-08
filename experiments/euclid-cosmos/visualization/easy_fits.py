# Plot galaxy pairs from the HDF5 file.

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

H5_PATH = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
OUT_DIR = "/n03data/fontirro/downsampled_plots_example/"


def _render_pair(axes, euc_data, cos_data_down, cos_data, idx, id_euc, id_cos):
    for ax in axes:
        ax.cla()
    axes[0].imshow(euc_data, cmap="gray")
    axes[0].set_title(f"Euclid  (idx={idx})\n{id_euc}")
    axes[1].imshow(cos_data_down, cmap="gray")
    axes[1].set_title(f"COSMOS (downscaled)  (idx={idx})\n{id_cos}")
    axes[2].imshow(cos_data, cmap="gray",
                   norm=ImageNormalize(cos_data, interval=PercentileInterval(99.5), stretch=AsinhStretch()))
    axes[2].set_title(f"COSMOS  (idx={idx})\n{id_cos}")


def main():
    parser = argparse.ArgumentParser(description="Plot galaxy pairs from the HDF5 file.")
    parser.add_argument("--h5", default=H5_PATH)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--start", type=int, default=0, help="First index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last index (exclusive); defaults to all")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    with h5py.File(args.h5, "r") as f:
        N = int(f.attrs["num_pairs"])
        end = min(args.end, N) if args.end is not None else N

        for idx in range(args.start, end):
            euc_data   = f["euclid_images"][idx, 0]
            cos_data_d = f["cosmos_images_downscaled"][idx, 0]
            cos_data   = f["cosmos_images"][idx, 0]
            euc_path   = f["catalog/euclid_paths"][idx].decode()
            cos_path   = f["catalog/cosmos_paths"][idx].decode()

            id_cos = cos_path.split("/")[-1].replace(".fits", "")
            id_euc = euc_path.split("/")[-1].split("_")[-2].replace("_cutout.fits", "")

            out_file = os.path.join(args.out_dir, f"pair_{idx}_{id_euc}_{id_cos}.png")
            if os.path.exists(out_file):
                continue

            _render_pair(axes, euc_data, cos_data_d, cos_data, idx, id_euc, id_cos)
            plt.tight_layout()
            plt.savefig(out_file, dpi=150, bbox_inches="tight")

    plt.close(fig)
    print(f"Done [{args.start}, {end})")


if __name__ == "__main__":
    main()
