"""
Regenerate and save the train/val/test split indices without running training.
Uses the same seed and ratios as train.py so the indices are identical.

Usage:
    python experiments/euclid-cosmos/save_split_indices.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import random_split

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset

# Must match train.py exactly
H5_PATH    = "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
CKPT_DIR   = "/n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2"
VAL_RATIO  = 0.05
TEST_RATIO = 0.001

dataset  = EuclidCosmosDataset(H5_PATH)
n_total  = len(dataset)
n_test   = int(n_total * TEST_RATIO)
n_val    = int(n_total * VAL_RATIO)
n_train  = n_total - n_val - n_test

generator = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

print(f"Dataset: {n_total} pairs → {n_train} train / {n_val} val / {n_test} test")

os.makedirs(CKPT_DIR, exist_ok=True)
np.save(os.path.join(CKPT_DIR, "train_indices.npy"), np.array(train_ds.indices))
np.save(os.path.join(CKPT_DIR, "val_indices.npy"),   np.array(val_ds.indices))
np.save(os.path.join(CKPT_DIR, "test_indices.npy"),  np.array(test_ds.indices))

print(f"Saved to {CKPT_DIR}:")
print(f"  train_indices.npy  ({n_train} indices)")
print(f"  val_indices.npy    ({n_val} indices)")
print(f"  test_indices.npy   ({n_test} indices)")
