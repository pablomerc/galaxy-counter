"""
Train dual-encoder contrastive baseline on precomputed neighbors data.
"""

import os
from pathlib import Path
from datetime import datetime

# Avoid matplotlib trying to write to non-writable $HOME/.config on cluster nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# Avoid numba cache writes to read-only environment paths when UMAP is enabled.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from galaxy_counter.neighbors import NeighborsPrecomputedDataset, simple_collate
from galaxy_counter.baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)


PRECOMPUTED_H5 = os.getenv(
    "GALAXY_COUNTER_PRECOMPUTED_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5",
)
VAL_RATIO = 0.05
BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_STEPS = 300_000
LR = 1e-4

WANDB_PROJECT = "galaxy-contrastive-neighbours-baseline"
RUN_NAME = "dual-encoder-contrastive-resnet18"
CHECKPOINT_DIR = os.getenv("GALAXY_COUNTER_CHECKPOINT_DIR", "./outputs/contrastive_baseline")


class VerboseModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that prints saved checkpoint paths to terminal."""

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            print(f"[checkpoint] saved: {filepath}", flush=True)


def is_h100_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    for i in range(torch.cuda.device_count()):
        if "h100" in torch.cuda.get_device_name(i).lower():
            return True
    return False


def main():
    seed = 42
    pl.seed_everything(seed, workers=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_run_dir = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_{run_stamp}")
    os.makedirs(checkpoint_run_dir, exist_ok=True)
    print(f"[checkpoint] run directory: {checkpoint_run_dir}", flush=True)

    is_h100 = is_h100_gpu()
    precision_setting = "bf16-mixed" if is_h100 else "16-mixed"
    batch_size = BATCH_SIZE

    dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
    total_size = len(dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=simple_collate,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=simple_collate,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    model = DualEncoderContrastiveModule(
        in_channels=4,
        embedding_dim=256,
        projection_dim=64,
        projection_hidden_dim=128,
        pretrained_encoder=False,
        temperature_galaxy=0.1,
        temperature_instrument=0.1,
        lambda_galaxy=1.0,
        lambda_instrument=1.0,
        lr=LR,
        weight_decay=1e-4,
        enable_umap_logging=True,
        num_umap_batches=16,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        log_model=False,
        config={
            "batch_size": batch_size,
            "precision": precision_setting,
            "dataset": PRECOMPUTED_H5,
            "val_ratio": VAL_RATIO,
            "max_steps": NUM_STEPS,
        },
    )

    best_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = VerboseModelCheckpoint(
        dirpath=checkpoint_run_dir,
        every_n_train_steps=1000,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    n_devices = 1
    trainer = pl.Trainer(
        max_steps=NUM_STEPS,
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true" if n_devices > 1 else "auto",
        precision=precision_setting,
        log_every_n_steps=10,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
