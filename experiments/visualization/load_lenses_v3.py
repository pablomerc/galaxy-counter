"""
Load trained channel-conditioning model (train_fm.py style) and lens examples
the same way as in train_fm.py, then generate validation-style plots for all lenses.

Model: single conditioning image (Legacy) -> target (HSC). sample(cond_images) only.
Data: HSCLegacyDataset with lens indices (returns hsc, legacy) with same normalization as train_fm.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Path setup: add galaxy_model so we can import data and train_fm
current_path = os.path.abspath(__file__)
galaxy_model_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, galaxy_model_dir)

from data import HSCLegacyDataset
from train_fm import ConditionalFlowMatchingModule

# --- Config (same as train_fm.py for 48x48 lenses) ---
HDF5_PATH = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5"
LENSE_INDICES = [
    3199, 3298, 4368, 4556, 8357, 9503, 19076, 20869, 26247,
    40506, 51839, 53037, 60565, 60980, 64245, 72326, 74053, 77857, 99695,
]
NUM_SAMPLES_PER_COND = 5

# Default checkpoint: channel-conditioning model
DEFAULT_CHECKPOINT = "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/2meictrr/checkpoints/epoch=201-step=75000.ckpt"


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
    target_batch,
    cond_batch,
    samples_per_lens,
    save_path,
    device,
):
    """
    Save grid: each row = one lens; cols = Cond (Legacy) | Target (HSC) | Sample1..5 | Mean.
    samples_per_lens: (num_lenses, num_samples_per_cond, C, H, W).
    """
    num_lens = target_batch.shape[0]
    num_cols = 2 + NUM_SAMPLES_PER_COND + 1
    col_titles = (
        ["Cond (Legacy)", "Target (HSC)"]
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
        cond = cond_batch[i : i + 1].to(device)
        target = target_batch[i : i + 1].to(device)
        samples = samples_per_lens[i]
        if isinstance(samples, torch.Tensor) and samples.device != device:
            samples = samples.to(device)
        mean_sample = samples.mean(dim=0, keepdim=True)

        target_chw = target[0, :3]
        vmin = target_chw.amin(dim=(1, 2))
        vmax = target_chw.amax(dim=(1, 2))

        cond_vis = _row_scale_rgb(cond[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 0].imshow(cond_vis)
        axes[i, 0].axis("off")

        target_vis = _row_scale_rgb(target[0, :3], vmin, vmax).detach().cpu().numpy()
        axes[i, 1].imshow(target_vis)
        axes[i, 1].axis("off")

        for j in range(NUM_SAMPLES_PER_COND):
            samp_vis = (
                _row_scale_rgb(samples[j, :3], vmin, vmax).detach().cpu().numpy()
            )
            axes[i, 2 + j].imshow(samp_vis)
            axes[i, 2 + j].axis("off")

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
    target_batch,
    cond_batch,
    samples_list,
    save_path,
):
    """
    One block per lens: 3 rows (Target HSC, Sample, Legacy cond), 5 cols (4 bands + RGB).
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
        "Lens validation: Target (HSC) | Sample | Legacy (cond)",
        fontsize=12,
        y=1.02,
    )
    if n_rows == 1:
        gs = gs.reshape(1, -1)

    for L in range(n_lens):
        target_np = target_batch[L].cpu().numpy()
        sample_np = samples_list[L].cpu().numpy()
        legacy_np = cond_batch[L].cpu().numpy()

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
    parser = argparse.ArgumentParser(
        description="Load channel-conditioning model (train_fm) and plot lens validation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="lens_plots_v3",
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

    # 1. Load lens dataset (same as train_fm: HSCLegacyDataset, batch = (hsc, legacy))
    lense_dataset = HSCLegacyDataset(
        hdf5_path=args.hdf5,
        idx_list=LENSE_INDICES,
    )
    lense_loader = DataLoader(
        lense_dataset,
        batch_size=len(LENSE_INDICES),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(lense_loader))
    # HSCLegacyDataset returns (hsc_image, legacy_image) -> batch is (target, cond)
    target_batch, cond_batch = batch
    target_batch = target_batch.to(device)
    cond_batch = cond_batch.to(device)
    num_lenses = target_batch.shape[0]
    print(f"Loaded {num_lenses} lens examples (target=HSC, cond=Legacy).")

    # 2. Load model (train_fm ConditionalFlowMatchingModule)
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.to(device)
    model.eval()

    # 3. Generate samples per lens (5 samples per lens for grid; 1 for triple)
    samples_per_lens = []
    with torch.no_grad():
        for i in range(num_lenses):
            cond = cond_batch[i : i + 1]
            cond_rep = cond.repeat(NUM_SAMPLES_PER_COND, 1, 1, 1)
            samples = model.sample(cond_rep)
            samples_per_lens.append(samples)
    samples_stacked = torch.stack(samples_per_lens, dim=0)

    # 4. Row-scaled grid for all lenses
    save_row_scaled_lens_grid(
        target_batch,
        cond_batch,
        samples_stacked,
        os.path.join(args.out_dir, "lens_grid_row_scaled.png"),
        device,
    )

    # 5. Astropy triple (Target | Sample | Legacy) for all lenses
    if not args.no_astropy:
        sample_single_list = []
        with torch.no_grad():
            for i in range(num_lenses):
                cond = cond_batch[i : i + 1]
                s = model.sample(cond)
                sample_single_list.append(s[0])
        save_astropy_lens_triple(
            target_batch,
            cond_batch,
            sample_single_list,
            os.path.join(args.out_dir, "lens_triple_target_sample_legacy.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
