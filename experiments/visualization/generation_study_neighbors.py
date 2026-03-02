"""
Load a pretrained neighbors model and run a generation study on precomputed neighbor data.

- Loads one batch from NeighborsPrecomputedDataset.
- For each target: 5 reconstructions (each using 1 of the 5 sameins) + 1 with all 5.
- Uses a fixed x_noise per target (reproducible comparison across columns).
- Plot: NUM_EXAMPLES rows (default 8).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from double_train_fm_neighbors import ConditionalFlowMatchingModule
from neighbors import NeighborsPrecomputedDataset, simple_collate

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CHECKPOINT_OPTIONS = [
    ("64 no geo", "tess-ml/galaxy-flow-matching-neighbours/n8szckjq",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/n8szckjq/checkpoints/latest-step=step=56000.ckpt",
     64, False, 56000),
    ("64 geo", "tess-ml/galaxy-flow-matching-neighbours/lr451mnx",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=61000.ckpt",
     64, True, 58000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/g2g9kvr4",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt",
     16, False, 75000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/x1xf4cym",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/x1xf4cym/checkpoints/latest-step=step=53000.ckpt",
     16, False, 53000),
]

# HDF5_NEIGHBORS = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5"
HDF5_NEIGHBORS = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0001.h5"
CHECKPOINT_IDX = 1  # 0=64 no geo, 1=64 geo, 2=16 no geo, ...
NUM_EXAMPLES = 8  # number of targets (rows) in the plot
NUM_SAMEINS = 5  # use first 5 sameins: 5 single-neighbor recons + 1 recon with all 5
SEED_EXAMPLES = 42  # seed for which batch/examples to use (change to get different targets)
SEED_NOISE = 42  # base seed for x_noise (per-row seed = SEED_NOISE + row_idx)

NEIGHBORS_VIZ_DIR = Path(__file__).parent / "neighbors_visualization"
GEN_STUDY_DIR = NEIGHBORS_VIZ_DIR / "gen_study"

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cpu")
if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        try:
            test_tensor = torch.tensor([1.0], device=f"cuda:{gpu_id}")
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}")
            break
        except RuntimeError:
            continue
if device.type == "cpu":
    print("No working GPU found, using CPU")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_label, _wandb_id, checkpoint_path, dim, _geom, step = CHECKPOINT_OPTIONS[CHECKPOINT_IDX]
mode_tag = "geom" if _geom else ""
print(f"Checkpoint: {_label}, path={checkpoint_path}, dim={dim}, geom={_geom}")

model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

# -----------------------------------------------------------------------------
# Load one batch from precomputed neighbors dataset (shuffle seeded for reproducibility)
# -----------------------------------------------------------------------------
dataset = NeighborsPrecomputedDataset(HDF5_NEIGHBORS)
g_examples = torch.Generator().manual_seed(SEED_EXAMPLES)
loader = DataLoader(
    dataset,
    batch_size=min(64, len(dataset)),
    shuffle=True,
    generator=g_examples,
    num_workers=0,
    collate_fn=simple_collate,
)
batch = next(iter(loader))
targets, samegals, sameins, masks, metadata = batch
# shapes: targets (B,C,H,W), samegals (B,C,H,W), sameins (B,k,C,H,W), masks (B,k)

targets = targets.to(device)
samegals = samegals.to(device)
sameins = sameins.to(device)
masks = masks.to(device)

B, k_max, C, H, W = sameins.shape
k_use = min(NUM_SAMEINS, k_max)
n_examples = min(NUM_EXAMPLES, B)
print(f"Batch: B={B}, sameins (B, k, C, H, W)=({B}, {k_max}, {C}, {H}, {W}), using k={k_use}, plotting {n_examples} examples")

# -----------------------------------------------------------------------------
# Reconstructions: for each of n_examples targets, 5 single-neighbor + 1 all
# -----------------------------------------------------------------------------
def _row_scale_rgb(x_chw, vmin, vmax):
    """Scale (3,H,W) to (H,W,3) in [0,1] with per-channel vmin/vmax."""
    x = x_chw[:3]
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    return y.permute(1, 2, 0)

all_rows = []  # list of dicts: target_image, samegal_image, samples (list of k_use+1 tensors)

for idx in range(n_examples):
    target_image = targets[idx : idx + 1]
    samegal_image = samegals[idx : idx + 1]
    sameins_one = sameins[idx, :k_use]
    mask_one = masks[idx, :k_use]

    torch.manual_seed(SEED_NOISE + idx)
    x_noise = torch.randn(
        1, model.hparams.in_channels, model.hparams.image_size, model.hparams.image_size,
        device=device,
    )

    samples = []
    for i in range(k_use):
        cond_sameins_i = sameins_one[i : i + 1].unsqueeze(0)
        out = model.sample(
            cond_image_samegal=samegal_image,
            cond_image_sameins=cond_sameins_i,
            masks=None,
            x_noise=x_noise,
        )
        samples.append(out[0])
    cond_sameins_all = sameins_one.unsqueeze(0)
    mask_all = mask_one.unsqueeze(0)
    out_all = model.sample(
        cond_image_samegal=samegal_image,
        cond_image_sameins=cond_sameins_all,
        masks=mask_all,
        x_noise=x_noise,
    )
    samples.append(out_all[0])
    all_rows.append({
        "target_image": target_image,
        "samegal_image": samegal_image,
        "samples": samples,
    })

print("Reconstructions done.")

# -----------------------------------------------------------------------------
# Visualization: n_examples rows x (target | samegal | recon 0..k_use-1 | recon all)
# -----------------------------------------------------------------------------
num_cols = 2 + k_use + 1
fig, axes = plt.subplots(n_examples, num_cols, figsize=(2 * num_cols, 2.2 * n_examples))
if n_examples == 1:
    axes = axes.reshape(1, -1)

for row_idx, row in enumerate(all_rows):
    target_image = row["target_image"]
    samegal_image = row["samegal_image"]
    samples = row["samples"]
    target_chw = target_image[0, :3]
    vmin = target_chw.amin(dim=(1, 2))
    vmax = target_chw.amax(dim=(1, 2))

    axes[row_idx, 0].imshow(_row_scale_rgb(target_image[0, :3], vmin, vmax).detach().cpu().numpy())
    axes[row_idx, 0].set_title("Target" if row_idx == 0 else "", fontsize=10)
    axes[row_idx, 0].axis("off")

    axes[row_idx, 1].imshow(_row_scale_rgb(samegal_image[0, :3], vmin, vmax).detach().cpu().numpy())
    axes[row_idx, 1].set_title("SameGal" if row_idx == 0 else "", fontsize=10)
    axes[row_idx, 1].axis("off")

    for i in range(k_use):
        axes[row_idx, 2 + i].imshow(_row_scale_rgb(samples[i][:3], vmin, vmax).detach().cpu().numpy())
        axes[row_idx, 2 + i].set_title(f"Recon\n(n{i})" if row_idx == 0 else "", fontsize=9)
        axes[row_idx, 2 + i].axis("off")

    axes[row_idx, 2 + k_use].imshow(_row_scale_rgb(samples[k_use][:3], vmin, vmax).detach().cpu().numpy())
    axes[row_idx, 2 + k_use].set_title("Recon\n(all)" if row_idx == 0 else "", fontsize=9)
    axes[row_idx, 2 + k_use].axis("off")

plt.suptitle(
    f"Generation study ({n_examples} targets, fixed x_noise per row). z_dim={dim} {mode_tag}, step={step}",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()

GEN_STUDY_DIR.mkdir(parents=True, exist_ok=True)
out_path = GEN_STUDY_DIR / f"gen_study_{n_examples}_targets_zdim{dim}{mode_tag}_seed{SEED_EXAMPLES}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
