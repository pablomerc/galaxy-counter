"""
TEST!!!!!!!!

Train the flow-matching model on paired Euclid (VIS) x COSMOS (F115W) cutouts.

Phase 1 (this script): simple pairs, no same-instrument neighbors.
  - encoder_1 conditions on the COSMOS counterpart of the same galaxy.
  - encoder_2 receives a dummy copy of the COSMOS image (k=1 stand-in).
    It will be replaced by real Euclid neighbors in Phase 2 after neighbors
    are computed from the trained model's embeddings.
  - lambda_geometric=0 because without real Euclid neighbors, the geometric
    loss has no meaningful signal for encoder_2.

Run locally (single GPU, for a quick sanity check):
    python experiments/euclid-cosmos/train.py

Submit on HPC via:
    sbatch experiments/euclid-cosmos/run_train.sh
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root or from the experiment directory
_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))
from dataset import EuclidCosmosDataset

from galaxy_counter.models.double_train_fm_neighbors import ConditionalFlowMatchingModule


class EuclidCosmosModel(ConditionalFlowMatchingModule):
    """Subclass that disables wandb-specific hooks from the base class.
    The base class assumes a wandb logger and 3-channel images; we use a CSV
    logger and 1-channel images, so those hooks are replaced with no-ops.
    """

    def __init__(self, *args, sample_dir=None, n_val_steps=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dir = sample_dir
        self.n_val_steps = n_val_steps
        self._fixed_val_batch = None

    def on_train_start(self) -> None:
        import time
        self._train_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Training started - Target: {self.trainer.max_steps} steps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

    def validation_step(self, batch, batch_idx):
        if self._fixed_val_batch is None and batch_idx == 0:
            euclid, cosmos, sameins, masks, _ = batch
            n = min(8, euclid.shape[0])
            self._fixed_val_batch = (
                euclid[:n].detach().clone(),
                cosmos[:n].detach().clone(),
                sameins[:n].detach().clone(),
                masks[:n].detach().clone(),
            )
        return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self._fixed_val_batch is None or self.sample_dir is None:
            return

        euclid, cosmos, _, masks = [t.to(self.device) for t in self._fixed_val_batch]
        os.makedirs(self.sample_dir, exist_ok=True)
        step = self.trainer.global_step
        n = euclid.shape[0]

        directions = [
            ("COSMOS → Euclid", cosmos, euclid, "COSMOS input",  "Generated Euclid", "Real Euclid"),
            ("Euclid → COSMOS", euclid, cosmos, "Euclid input",  "Generated COSMOS", "Real COSMOS"),
        ]

        for dir_label, cond, target, t0, t1, t2 in directions:
            sameins = cond.unsqueeze(1)
            with torch.no_grad():
                generated = self.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=self.n_val_steps,
                )

            fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n))
            if n == 1:
                axes = axes[None, :]
            for j, title in enumerate([t0, t1, t2]):
                axes[0, j].set_title(title, fontsize=9)
            for i in range(n):
                for j, img in enumerate([cond[i], generated[i], target[i]]):
                    arr = img.squeeze().cpu().float().numpy()
                    vmin, vmax = np.percentile(arr, [1, 99])
                    axes[i, j].imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
                    axes[i, j].axis("off")

            tag = dir_label.replace(" ", "").replace("→", "-")
            fig.suptitle(f"{dir_label}  |  step {step}", fontsize=10)
            plt.tight_layout()
            fname = os.path.join(self.sample_dir, f"{tag}_step={step:07d}.png")
            plt.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"Saved samples: {fname}")

# ---------------------------------------------------------------------------
# CONFIG — edit before running
# ---------------------------------------------------------------------------
H5_PATH     = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
CKPT_DIR    = "/n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2"

BATCH_SIZE  = 64
NUM_WORKERS = 16
VAL_RATIO   = 0.05
TEST_RATIO  = 0.001
NUM_STEPS   = 200_000
IMAGE_SIZE  = 40      #Euclid cutout spatial size
LR          = 1e-4    #learning rate for AdamW optimizer

N_GPUS      = 1       #set to number of GPUs on the node
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """
    Builds the 5-tuple the model expects:
      (anchor, samegal, sameins, masks, metadata)

    Direction (which survey is anchor vs condition) is determined by the
    dataset: even indices → Euclid anchor, odd indices → COSMOS anchor.
    """
    anchor = torch.stack([b[0] for b in batch])   # (B, 1, H, W)
    cond   = torch.stack([b[1] for b in batch])   # (B, 1, H, W)
    B = anchor.shape[0]

    sameins  = cond.unsqueeze(1)                  # (B, 1, 1, H, W)
    masks    = torch.ones(B, 1, dtype=torch.bool)
    metadata = [b[2] for b in batch]
    return anchor, cond, sameins, masks, metadata


def main():
    pl.seed_everything(42, workers=True)

    dataset   = EuclidCosmosDataset(H5_PATH, bidirectional=True)
    n_total   = len(dataset)
    n_test    = int(n_total * TEST_RATIO)
    n_val     = int(n_total * VAL_RATIO)
    n_train   = n_total - n_val - n_test
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
    print(f"Dataset: {n_total} pairs → {n_train} train / {n_val} val / {n_test} test")

    # Save test indices so testing.py uses the exact same held-out set
    os.makedirs(CKPT_DIR, exist_ok=True)
    test_indices_path = os.path.join(CKPT_DIR, "test_indices.npy")
    np.save(test_indices_path, np.array(test_ds.indices))
    print(f"Test indices saved to: {test_indices_path}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    model = EuclidCosmosModel(
        sample_dir=os.path.join(CKPT_DIR, "samples"),
        in_channels=1,            # Euclid VIS: 1 channel
        cond_channels=1,          # COSMOS F115W: 1 channel
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=LR,
        num_sample_images=8,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.0,     # no neighbors yet → geometric loss disabled
        mask_center=False,
    )

    csv_logger = CSVLogger(save_dir=CKPT_DIR, name="logs")

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_checkpoint = ModelCheckpoint(
        dirpath=CKPT_DIR,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=CKPT_DIR,
        every_n_train_steps=2000,
        save_top_k=1,
        filename="latest-step={step}",
    )

    trainer = pl.Trainer(
        max_steps=max(1, int(NUM_STEPS / N_GPUS)),
        logger=csv_logger,
        accelerator="auto",
        devices=N_GPUS,
        strategy="ddp_find_unused_parameters_true" if N_GPUS > 1 else "auto",
        log_every_n_steps=10,
        precision="bf16-mixed",
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=2,
    )

    # Set to the latest checkpoint path to resume, or None to start fresh
    RESUME_FROM = None
    trainer.fit(model, train_loader, val_loader, ckpt_path=RESUME_FROM)


if __name__ == "__main__":
    main()
