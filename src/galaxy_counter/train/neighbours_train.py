"""
Train the double encoder flow matching model on the Neighbors dataset (48x48 only).

Uses NeighborsDataset from neighbors.py. No lenses, no masking, no multi-size.
"""

import os

# Print immediately so you see the script started (heavy imports below can take 1–2 min)
print("neighbours_train: loading imports...", flush=True)

from torch.utils.data import DataLoader, Subset, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from galaxy_counter.neighbors import (
    NeighborsDataset,
    NeighborsPrecomputedDataset,
    simple_collate,
    collate_neighbors,
)
from galaxy_counter.models.double_train_fm_neighbors import (
    ConditionalFlowMatchingModule,
    is_h100_gpu,
)


# Model expects (x_1, cond_image_samegal, cond_image_sameins, metadata) — no masks.

# For NeighborsDataset / NeighborsDatasetRawRAM: variable-length sameins → use collate_neighbors (pads), then drop masks.
def collate_for_model(batch):
    targets, samegals, padded_neighbors, _, metadata = collate_neighbors(batch)
    return (targets, samegals, padded_neighbors, metadata)


# For NeighborsPrecomputedDataset: data is already padded → use simple_collate, then drop masks.
def collate_for_model_precomputed(batch):
    # Don't use the underscore _; grab the masks!
    targets, samegals, sameins, masks, metadata = simple_collate(batch)
    return (targets, samegals, sameins, masks, metadata)


# --- Config ---

# If using neighbors file directly
# NEIGHBORS_H5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
# MAX_NEIGHBORS = 15



#USING PRECOMPUTED BATCHES
# PRECOMPUTED_H5 = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/train_neighbors.vds"
# PRECOMPUTED_H5 = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"
# VAL_PRECOMPUTED_H5 = "/data/vision/billf/scratch/pablomer/data/neighbor_batches/val_neighbors.vds"
PRECOMPUTED_H5 = os.getenv(
    "GALAXY_COUNTER_PRECOMPUTED_H5",
    "/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5",
)  # contains all
BATCH_SIZE = 64  # 64 OOM on V100 32GB with 48x48 + 2 encoders + UNet; reduce if still OOM
NUM_WORKERS = 0
DATALOADER_MODE = "precomputed"  # "precomputed" (NeighborsPrecomputedDataset + simple_collate) or "neighbors" (NeighborsDataset/RawRAM + collate_neighbors)

VAL_RATIO = 0.05  # 5% for validation
NUM_STEPS = 300_000 * 5
IMAGE_SIZE = 48
LR = 1e-4
LAMBDA_GENERATIVE = 1.0
# LAMBDA_GEOMETRIC = 7.5e-4
LAMBDA_GEOMETRIC = 0.0

WANDB_PROJECT = "galaxy-flow-matching-neighbours"


def main():
    seed = 42
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    is_h100 = is_h100_gpu()
    batch_size = BATCH_SIZE
    precision_setting = "16-mixed"
    num_steps = NUM_STEPS
    if is_h100:
        batch_size = 64
        precision_setting = "bf16-mixed"
        print(f"H100 detected: batch_size={batch_size}, precision={precision_setting}")

    # Single dataset, then train/val split by index
    dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)

    total_size = len(dataset)
    val_size = int(total_size * VAL_RATIO)
    train_size = total_size - val_size


    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    collate_fn = collate_for_model_precomputed if DATALOADER_MODE == "precomputed" else collate_for_model

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )

    model = ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=LR,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=LAMBDA_GENERATIVE,
        lambda_geometric=LAMBDA_GEOMETRIC,
        mask_center=False,
    )

    # Pass config here; Lightning handles DDP so only rank 0 gets real wandb (avoids .config.update() on placeholder)
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name="neighbours-48x48-zdim64-geom0.0-longtraining",
        log_model=False,
        config={
            "batch_size": batch_size,
            "precision": precision_setting,
            "is_h100": is_h100,
            "dataset": "NeighborsDataset",
            "image_size": IMAGE_SIZE,
        },
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    periodic_checkpoint = ModelCheckpoint(
        every_n_train_steps=1000,
        save_top_k=1,
        filename="latest-step={step}",
        save_last=False,
    )

    n_devices = 4  # set to 4 or more for multi-GPU
    trainer = pl.Trainer(
        max_steps=max(1, int(num_steps / n_devices)),
        logger=wandb_logger,
        accelerator="auto",
        devices=n_devices,
        strategy="ddp_find_unused_parameters_true" if n_devices > 1 else "auto",
        log_every_n_steps=10,
        precision=precision_setting,
        val_check_interval=1000, # bring back to 1000
        check_val_every_n_epoch=None,
        callbacks=[best_checkpoint, periodic_checkpoint],
        num_sanity_val_steps=0,  # skip sanity check to start training faster (first val batch has heavy CUDA init)
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
