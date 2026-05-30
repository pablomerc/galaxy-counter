"""
Generate sample images from the trained Euclid x COSMOS model and save to HDF5.

For each galaxy, generates num_samples reconstructions in each requested direction:
  cosmos-to-euclid: condition on COSMOS, generate Euclid
  euclid-to-cosmos: condition on Euclid, generate COSMOS

Results are saved to an HDF5 file for fast re-plotting without re-running the model.
A quick preview PNG is also saved immediately.

Usage:
    python experiments/euclid-cosmos/generate.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2/best-....ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --out-dir    /n03data/fontirro/plots_model/euclid-cosmos-phase1-v2/generated \
        --indices    /n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2/test_indices.npy \
        --n-images   16 \
        --num-samples 5 \
        --direction  both
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def _scale(arr):
    """Clip and scale a (H, W) array to [0, 1] using 1–99 percentile."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def generate_samples(model, cond, sameins, masks, num_samples, device):
    """
    Generate num_samples reconstructions per galaxy.
    Returns (N, num_samples, 1, H, W) tensor on CPU.
    """
    N = cond.shape[0]
    all_samples = []
    for i in range(N):
        cond_i    = cond[i:i+1].to(device).repeat(num_samples, 1, 1, 1)
        sins_i    = sameins[i:i+1].to(device).repeat(num_samples, 1, 1, 1, 1)
        masks_i   = masks[i:i+1].to(device).repeat(num_samples, 1)
        with torch.no_grad():
            s = model.sample(cond_image_samegal=cond_i,
                             cond_image_sameins=sins_i,
                             masks=masks_i,
                             num_steps=args_num_steps)
        all_samples.append(s.cpu().unsqueeze(0))   # (1, num_samples, 1, H, W)
    return torch.cat(all_samples, dim=0)


def save_preview(cond, samples, mean, target, col_labels, out_path):
    """Save a quick n_images × (1 + num_samples + 1 + 1) preview grid."""
    N, S = samples.shape[0], samples.shape[1]
    num_cols = 1 + S + 1 + 1   # cond | sample_1..S | mean | real

    fig, axes = plt.subplots(N, num_cols, figsize=(2 * num_cols, 2.2 * N), squeeze=False)

    titles = [col_labels[0]] + [f"Sample {j+1}" for j in range(S)] + ["Mean", col_labels[1]]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=8)

    for i in range(N):
        imgs = ([cond[i, 0].numpy()] +
                [samples[i, j, 0].numpy() for j in range(S)] +
                [mean[i, 0].numpy(), target[i, 0].numpy()])
        for j, arr in enumerate(imgs):
            axes[i, j].imshow(_scale(arr), cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Preview saved: {out_path}")


def run_direction(model, cond, target, sameins, masks, num_samples, device,
                  key, out_dir, col_labels, h5file):
    """Generate samples for one direction and write to open HDF5 file."""
    print(f"\n--- Direction: {col_labels[0]} → {col_labels[1]} ---")
    samples  = generate_samples(model, cond, sameins, masks, num_samples, device)
    mean_img = samples.mean(dim=1)

    # Save to HDF5
    grp = h5file.create_group(key)
    grp.create_dataset("cond",    data=cond.numpy(),         compression="gzip")
    grp.create_dataset("target",  data=target.numpy(),       compression="gzip")
    grp.create_dataset("samples", data=samples.numpy(),      compression="gzip")
    grp.create_dataset("mean",    data=mean_img.numpy(),     compression="gzip")
    grp.attrs["from_survey"] = col_labels[0]
    grp.attrs["to_survey"]   = col_labels[1]

    # Quick preview
    preview_path = os.path.join(out_dir, f"{key}_preview.png")
    save_preview(cond, samples, mean_img, target, col_labels, preview_path)


# Global so generate_samples can access it without threading complexity
args_num_steps = 100


def main():
    global args_num_steps

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--h5",           required=True)
    p.add_argument("--out-dir",      default="generated")
    p.add_argument("--indices",      default=None,
                   help="Optional .npy file (e.g. test_indices.npy) to sample from.")
    p.add_argument("--n-images",     type=int, default=16,
                   help="Number of galaxy rows to generate.")
    p.add_argument("--num-samples",  type=int, default=5,
                   help="Number of samples generated per galaxy.")
    p.add_argument("--num-steps",    type=int, default=100,
                   help="ODE integration steps.")
    p.add_argument("--direction",    default="both",
                   choices=["cosmos-to-euclid", "euclid-to-cosmos", "both"])
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    args_num_steps = args.num_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval().to(device)

    dataset = EuclidCosmosDataset(args.h5)   # bidirectional=False → always (euclid, cosmos)

    if args.indices is not None:
        pool = np.load(args.indices)
        print(f"Drawing from {len(pool)} indices ({args.indices})")
    else:
        pool = np.arange(len(dataset))

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(pool, size=min(args.n_images, len(pool)), replace=False).tolist()
    subset = Subset(dataset, chosen)
    loader = DataLoader(subset, batch_size=args.n_images, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    euclid, cosmos, sameins_orig, masks, _ = next(iter(loader))

    os.makedirs(args.out_dir, exist_ok=True)
    data_path = os.path.join(args.out_dir, "generation_data.h5")

    with h5py.File(data_path, "w") as f:
        f.create_dataset("euclid",  data=euclid.numpy(),  compression="gzip")
        f.create_dataset("cosmos",  data=cosmos.numpy(),  compression="gzip")
        f.create_dataset("indices", data=np.array(chosen, dtype=np.int64))
        f.attrs["num_images"]  = len(chosen)
        f.attrs["num_samples"] = args.num_samples
        f.attrs["num_steps"]   = args.num_steps
        f.attrs["seed"]        = args.seed
        f.attrs["directions"]  = args.direction

        if args.direction in ("cosmos-to-euclid", "both"):
            sameins = cosmos.unsqueeze(1)
            run_direction(model, cosmos, euclid, sameins, masks,
                          args.num_samples, device,
                          key="cosmos_to_euclid",
                          out_dir=args.out_dir,
                          col_labels=("COSMOS", "Euclid"),
                          h5file=f)

        if args.direction in ("euclid-to-cosmos", "both"):
            sameins = euclid.unsqueeze(1)
            run_direction(model, euclid, cosmos, sameins, masks,
                          args.num_samples, device,
                          key="euclid_to_cosmos",
                          out_dir=args.out_dir,
                          col_labels=("Euclid", "COSMOS"),
                          h5file=f)

    print(f"\nData saved: {data_path}")
    print("Done. Re-plot anytime with replot_generate.py --data", data_path)


if __name__ == "__main__":
    main()
