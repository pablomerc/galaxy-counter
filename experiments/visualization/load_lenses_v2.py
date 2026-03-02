"""
Load trained double_train_fm model and lens examples the same way as in double_train_fm.py,
then generate validation-style plots for all lenses (row-scaled grid + astropy triple).
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Path setup: add galaxy_model so we can import data and double_train_fm
current_path = os.path.abspath(__file__)
galaxy_model_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, galaxy_model_dir)

from data import HSCLegacyTripletDatasetZoomLenses, custom_collate_fn
from double_train_fm import ConditionalFlowMatchingModule

# --- Config (same as double_train_fm.py for 48x48 lenses) ---
HDF5_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
LENSE_INDICES = [
    3199, 3298, 4368, 4556, 8357, 9503, 19076, 20869, 26247,
    40506, 51839, 53037, 60565, 60980, 64245, 72326, 74053, 77857, 99695,
]
IMAGE_SIZE = 48
NUM_SAMPLES_PER_COND = 5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _row_scale_rgb(x_chw, vmin, vmax):
    """Scale (3,H,W) to (H,W,3) in [0,1] using per-channel vmin/vmax."""
    x = x_chw[:3]
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    return y.permute(1, 2, 0)


def save_row_scaled_lens_grid(
    anchor_batch,
    samegal_batch,
    sameins_batch,
    samples_per_lens,
    save_path,
    device,
):
    """
    Save grid: each row = one lens; cols = SameGal | SameIns (1st) | Target | Sample1..5 | Mean.
    samples_per_lens: (num_lenses, num_samples_per_cond, C, H, W).
    """
    num_lens = anchor_batch.shape[0]
    num_cols = 3 + NUM_SAMPLES_PER_COND + 1
    col_titles = (
        ["SameGal", "SameIns (1st)", "Target"]
        + [f"Sample {j+1}" for j in range(NUM_SAMPLES_PER_COND)]
        + ["Mean"]
    )

    fig, axes = plt.subplots(
        num_lens,
        num_cols,
        figsize=(2 * num_cols, 2 * num_lens),
        squeeze=False,
    )
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for i in range(num_lens):
        samegal = samegal_batch[i : i + 1].to(device)
        target = anchor_batch[i : i + 1].to(device)
        sameins = sameins_batch[i : i + 1].to(device)
        sameins_first = sameins[:, 0:1]

        # Precomputed samples for this lens: (num_samples_per_cond, C, H, W)
        samples = samples_per_lens[i]  # already on device or cpu
        if isinstance(samples, torch.Tensor) and samples.device != device:
            samples = samples.to(device)
        mean_sample = samples.mean(dim=0, keepdim=True)

        target_chw = target[0, :3]
        vmin = target_chw.amin(dim=(1, 2))
        vmax = target_chw.amax(dim=(1, 2))

        samegal_vis = _row_scale_rgb(samegal[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 0].imshow(samegal_vis)
        axes[i, 0].axis("off")

        sameins_first_vis = (
            _row_scale_rgb(sameins_first[0, 0, :3], vmin, vmax).detach().cpu().numpy()
        )
        axes[i, 1].imshow(sameins_first_vis)
        axes[i, 1].axis("off")

        target_vis = _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 2].imshow(target_vis)
        axes[i, 2].axis("off")

        for j in range(NUM_SAMPLES_PER_COND):
            samp_vis = (
                _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
            )
            axes[i, 3 + j].imshow(samp_vis)
            axes[i, 3 + j].axis("off")

        mean_vis = (
            _row_scale_rgb(mean_sample[0, :3], vmin, vmax).detach().cpu().numpy()
        )
        axes[i, -1].imshow(mean_vis)
        axes[i, -1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved row-scaled lens grid to {save_path}")


def save_astropy_lens_triple(
    anchor_batch,
    samegal_batch,
    sameins_batch,
    samples_list,
    save_path,
):
    """
    One block per lens: 3 rows (Target HSC, Sample, Legacy), 5 cols (4 bands + RGB).
    samples_list: list of (C,H,W) tensors, one per lens.
    """
    try:
        from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
    except ImportError:
        print("Skipping astropy lens triple (astropy not installed).")
        return

    def _normed_imshow_lens(ax, data2d, interval_obj, stretch_obj, title=None):
        data = np.asarray(data2d, dtype=np.float32)
        mask = np.isfinite(data)
        if not np.any(mask):
            ax.axis("off")
            return
        vmin, vmax = interval_obj.get_limits(data[mask])
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_obj, clip=True)
        ax.imshow(data, origin="lower", norm=norm, cmap="magma")
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    def make_rgb_lens(img, r, g, b, interval_obj, stretch_obj):
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        channels_available = img.shape[0]
        r, g, b = [min(x, channels_available - 1) for x in [r, g, b]]
        rgb = np.stack([img[r], img[g], img[b]], axis=-1).astype(np.float32)
        out = np.zeros_like(rgb)
        for k in range(3):
            ch = rgb[..., k]
            mask = np.isfinite(ch)
            if not np.any(mask):
                continue
            vmin, vmax = interval_obj.get_limits(ch[mask])
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax, stretch=stretch_obj, clip=True
            )
            out[..., k] = norm(ch)
        mx = np.nanmax(out)
        if mx > 0:
            out /= mx
        return np.clip(out, 0, 1)

    interval_obj = PercentileInterval(99.5)
    stretch_obj = AsinhStretch()
    band_names = ["g", "r", "i", "z"]

    n_lens = len(samples_list)
    n_rows = 3 * n_lens
    row_height = 10 / 3
    fig, gs = plt.subplots(
        n_rows, 5, figsize=(14, row_height * n_rows), constrained_layout=True
    )
    fig.suptitle(
        "Lens validation: Target (HSC) | Sample | Legacy (samegal)",
        fontsize=12,
        y=1.02,
    )
    if n_rows == 1:
        gs = gs.reshape(1, -1)

    for L in range(n_lens):
        target_np = anchor_batch[L].cpu().numpy()
        sample_np = samples_list[L].cpu().numpy()
        legacy_np = samegal_batch[L].cpu().numpy()

        base = L * 3
        for c in range(4):
            _normed_imshow_lens(
                gs[base + 0, c],
                target_np[c],
                interval_obj,
                stretch_obj,
                title=f"Target {band_names[c]}",
            )
        gs[base + 0, 4].imshow(
            make_rgb_lens(target_np, 2, 1, 0, interval_obj, stretch_obj),
            origin="lower",
        )
        gs[base + 0, 4].set_title("Target RGB (irg)", fontsize=9, fontweight="bold")
        gs[base + 0, 4].axis("off")

        for c in range(4):
            _normed_imshow_lens(
                gs[base + 1, c],
                sample_np[c],
                interval_obj,
                stretch_obj,
                title=f"Sample {band_names[c]}",
            )
        gs[base + 1, 4].imshow(
            make_rgb_lens(sample_np, 2, 1, 0, interval_obj, stretch_obj),
            origin="lower",
        )
        gs[base + 1, 4].set_title("Sample RGB (irg)", fontsize=9, fontweight="bold")
        gs[base + 1, 4].axis("off")

        for c in range(4):
            _normed_imshow_lens(
                gs[base + 2, c],
                legacy_np[c],
                interval_obj,
                stretch_obj,
                title=f"Legacy {band_names[c]}",
            )
        gs[base + 2, 4].imshow(
            make_rgb_lens(legacy_np, 2, 1, 0, interval_obj, stretch_obj),
            origin="lower",
        )
        gs[base + 2, 4].set_title("Legacy RGB (irg)", fontsize=9, fontweight="bold")
        gs[base + 2, 4].axis("off")

    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved astropy lens triple to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Load model and plot lens validation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="lens_plots_v2",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default=HDF5_PATH,
        help="HDF5 path for lens dataset",
    )
    parser.add_argument(
        "--no_astropy",
        action="store_true",
        help="Skip astropy triple plot (if astropy not available)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load lens dataset (same as double_train_fm)
    lense_dataset = HSCLegacyTripletDatasetZoomLenses(
        hdf5_path=args.hdf5,
        lense_indices=LENSE_INDICES,
        is96=False,
    )
    lense_loader = DataLoader(
        lense_dataset,
        batch_size=len(LENSE_INDICES),
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )
    batch = next(iter(lense_loader))
    anchor_image, same_galaxy, same_instrument, metadata = batch
    anchor_image = anchor_image.to(device)
    same_galaxy = same_galaxy.to(device)
    same_instrument = same_instrument.to(device)
    num_lenses = anchor_image.shape[0]
    print(f"Loaded {num_lenses} lens examples.")

    # 2. Load model
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.to(device)
    model.eval()

    print(f"Model loaded from checkpoint: {args.checkpoint}")
    print(f"Model device: {model.device}")
    # 3. Generate samples per lens (5 samples per lens for grid; 1 for triple)
    num_samples_per_cond = NUM_SAMPLES_PER_COND
    samples_per_lens = []
    with torch.no_grad():
        for i in range(num_lenses):
            samegal = same_galaxy[i : i + 1]
            sameins = same_instrument[i : i + 1]
            samegal_rep = samegal.repeat(num_samples_per_cond, 1, 1, 1)
            sameins_rep = sameins.repeat(num_samples_per_cond, 1, 1, 1, 1)
            samples = model.sample(samegal_rep, sameins_rep)
            samples_per_lens.append(samples)
            print(f"Generated sample for lens {i}")
    # Stack: list of (num_samples_per_cond, C, H, W) -> keep as list for now
    samples_stacked = torch.stack(
        [s for s in samples_per_lens], dim=0
    )  # (num_lenses, num_samples_per_cond, C, H, W)

    print(f"Samples stacked: {samples_stacked.shape}")
    # 4. Row-scaled grid for all lenses
    save_row_scaled_lens_grid(
        anchor_image,
        same_galaxy,
        same_instrument,
        samples_stacked,
        os.path.join(args.out_dir, "lens_grid_row_scaled.png"),
        device,
    )

    # 5. Astropy triple (Target | Sample | Legacy) for all lenses
    if not args.no_astropy:
        with torch.no_grad():
            sample_single_list = []
            for i in range(num_lenses):
                samegal = same_galaxy[i : i + 1]
                sameins = same_instrument[i : i + 1]
                s = model.sample(samegal, sameins)
                sample_single_list.append(s[0])
        save_astropy_lens_triple(
            anchor_image,
            same_galaxy,
            same_instrument,
            sample_single_list,
            os.path.join(args.out_dir, "lens_triple_target_sample_legacy.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
