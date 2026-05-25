"""
TEST!!!!!!!!

Train the flow-matching model on paired Euclid (VIS) × COSMOS (F115W) cutouts.

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
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Allow running from repo root or from the experiment directory
sys.path.insert(0, os.path.dirname(__file__))
from dataset import EuclidCosmosDataset

from galaxy_counter.models.double_train_fm_neighbors import ConditionalFlowMatchingModule

# ---------------------------------------------------------------------------
# CONFIG — edit before running
# ---------------------------------------------------------------------------
H5_PATH     = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
CKPT_DIR    = "/n03data/fontirro/checkpoints/euclid-cosmos-phase1"

BATCH_SIZE  = 64
NUM_WORKERS = 8
VAL_RATIO   = 0.05
NUM_STEPS   = 200_000
IMAGE_SIZE  = 40      # Euclid cutout spatial size
LR          = 1e-4

N_GPUS      = 1       # set to number of GPUs on the node
WANDB_PROJECT = "galaxy-flow-matching-euclid-cosmos"
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """
    Builds the 5-tuple the model expects:
      (anchor, samegal, sameins, masks, metadata)

    anchor  = Euclid image         (B, 1, H, W)
    samegal = COSMOS counterpart    (B, 1, H, W)
    sameins = dummy stand-in (k=1) (B, 1, 1, H, W)  ← same as samegal for now
    masks   = all True             (B, 1)
    """
    euclid = torch.stack([b[0] for b in batch])   # (B, 1, H, W)
    cosmos = torch.stack([b[1] for b in batch])   # (B, 1, H, W)
    B = euclid.shape[0]

    sameins = cosmos.unsqueeze(1)                 # (B, 1, 1, H, W)
    masks   = torch.ones(B, 1, dtype=torch.bool)
    metadata = [{"anchor_survey": "euclid", **b[2]} for b in batch]
    return euclid, cosmos, sameins, masks, metadata


def main():
    pl.seed_everything(42, workers=True)

    dataset   = EuclidCosmosDataset(H5_PATH)
    n_total   = len(dataset)
    n_val     = int(n_total * VAL_RATIO)
    n_train   = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"Dataset: {n_total} pairs → {n_train} train / {n_val} val")

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

    model = ConditionalFlowMatchingModule(
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

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=f"phase1-pairs-img{IMAGE_SIZE}-ch128",
        log_model=False,
        config={
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "dataset": H5_PATH,
            "phase": 1,
        },
    )

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
        logger=wandb_logger,
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

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
