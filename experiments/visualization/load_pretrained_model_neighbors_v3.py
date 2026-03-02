"""
Load a pretrained neighbors model and compare average distance in the instrument latent space:
  - "Instrument neighbors": (target, sameins) from NeighborsPrecomputedDataset — curated
    same-instrument, same-survey neighbors.
  - "Instrument non-neighbors (same survey)": (target, legacy_bunch) where legacy_bunch
    is a bunch of Legacy images from HSCLegacyDatasetZoom (same survey = Legacy).

We filter precomputed to Legacy anchors so targets and sameins are Legacy; non-neighbors
are Legacy images from HSCLegacyDatasetZoom.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from neighbors import NeighborsPrecomputedDataset, simple_collate
from data import HSCLegacyDatasetZoom

# Set True to load old double-encoder model (double_train_fm), False for neighbors model (double_train_fm_neighbors)
USE_OLD_MODEL = False

PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbours_vds.h5'
# PRECOMPUTED_H5 = '/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5'

NUM_EXAMPLES = 2048 # 2048
BATCH_SIZE = 256

HDF5_PATH = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5'
# Legacy images from HSCLegacyDatasetZoom for non-neighbors (same survey)
IDX_LEGACY_BUNCH = list(range(80_000, 85_000))  # 5000 Legacy images
NUM_NON_NEIGHBOR_SAMPLES = 50000

# Neighbors model checkpoints (double_train_fm_neighbours)
CHECKPOINT_OPTIONS = [
    ("64 no geo", "tess-ml/galaxy-flow-matching-neighbours/n8szckjq",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/n8szckjq/checkpoints/latest-step=step=56000.ckpt",
     64, False, 56000),
    ("64 geo", "tess-ml/galaxy-flow-matching-neighbours/lr451mnx",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/lr451mnx/checkpoints/latest-step=step=57000.ckpt",
     64, True, 50000),
    ("16 no geo", "tess-ml/galaxy-flow-matching-neighbours/g2g9kvr4",
     "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt",
     16, False, 75000),
]

# Old double-encoder model checkpoints (double_train_fm, same as load_pretrained_model.py)
OLD_CHECKPOINT_OPTIONS = [
    ("16 geom", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt", 16, True),
    ("64 geom", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/twop8sfb/checkpoints/latest-step=step=75000.ckpt", 64, True),
    ("8", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/rach5aeu/checkpoints/latest-step=step=75000.ckpt", 8, False),
    ("8 geom", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wdbsh3rc/checkpoints/latest-step=step=75000.ckpt", 8, True),
    ("64", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/muia8i77/checkpoints/latest-step=step=75000.ckpt", 64, False),
]

if USE_OLD_MODEL:
    from double_train_fm import ConditionalFlowMatchingModule
    idx_old = 1  # index into OLD_CHECKPOINT_OPTIONS
    _label, checkpoint_path, dim, _geom = OLD_CHECKPOINT_OPTIONS[idx_old]
    mode_tag = "geom" if _geom else ""
    print(f"Model: old (double_train_fm). Checkpoint: {_label}, path={checkpoint_path}, dim={dim}, geom={_geom}")
else:
    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    idx = 1  # index into CHECKPOINT_OPTIONS
    _label, _wandb_id, checkpoint_path, dim, _geom, step = CHECKPOINT_OPTIONS[idx]
    mode_tag = "geom" if _geom else ""
    print(f"Model: neighbors (double_train_fm_neighbors). Checkpoint: {_label} (wandb {_wandb_id}), path={checkpoint_path}, dim={dim}, geom={_geom}")
print(f"Neighbors: NeighborsPrecomputedDataset (target ↔ sameins)")
print(f"Non-neighbors: HSCLegacyDatasetZoom Legacy bunch ({len(IDX_LEGACY_BUNCH)} images, same survey)")
print("Metric: instrument latent (encoder_2)\n")

device = torch.device('cpu')
if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        try:
            test_tensor = torch.tensor([1.0], device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
            break
        except RuntimeError:
            continue

model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
model.eval()
torch.set_grad_enabled(False)
model = model.to(device)

def encode_encoder2(images):
    """images: (B, C, H, W). Returns (B, D) after spatial mean."""
    with torch.no_grad():
        out = model.encoder_2(images)
    return torch.mean(out, dim=1)

# --- Load NeighborsPrecomputedDataset; keep only Legacy anchors (same survey as Legacy bunch) ---
full_dataset = NeighborsPrecomputedDataset(PRECOMPUTED_H5)
survey_arr = np.array(full_dataset.meta_survey)
legacy_idx = np.where(np.char.lower(survey_arr.astype(str)) == 'legacy')[0]
if len(legacy_idx) == 0:
    print("Warning: no 'legacy' in meta_survey; using first NUM_EXAMPLES (survey match may not hold).")
    legacy_idx = np.arange(min(NUM_EXAMPLES, len(full_dataset)))
else:
    legacy_idx = legacy_idx[:min(len(legacy_idx), NUM_EXAMPLES)]
subset = Subset(full_dataset, legacy_idx)
loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=simple_collate,
)
targets_list, sameins_list, masks_list = [], [], []
for batch in loader:
    t, _, si, m, _ = batch
    targets_list.append(t)
    sameins_list.append(si)
    masks_list.append(m)
    if sum(x.shape[0] for x in targets_list) >= NUM_EXAMPLES:
        break

targets = torch.cat(targets_list, dim=0)[:NUM_EXAMPLES].to(device)
sameins = torch.cat(sameins_list, dim=0)[:NUM_EXAMPLES].to(device)
masks = torch.cat(masks_list, dim=0)[:NUM_EXAMPLES]

N = targets.shape[0]
_, N_max, C, H, W = sameins.shape
print(f"Precomputed (Legacy anchors): {N} targets, sameins shape {sameins.shape}, masks {masks.shape}")

# --- Load bunch of Legacy from HSCLegacyDatasetZoom (for non-neighbors) ---
dataset_bunch = HSCLegacyDatasetZoom(hdf5_path=HDF5_PATH, idx_list=IDX_LEGACY_BUNCH)
loader_bunch = DataLoader(dataset_bunch, batch_size=2048, shuffle=False, num_workers=0)
legacy_bunch_list = []
for batch in loader_bunch:
    _, leg = batch
    legacy_bunch_list.append(leg)
legacy_bunch = torch.cat(legacy_bunch_list, dim=0).to(device)
M = legacy_bunch.shape[0]
print(f"Legacy bunch (HSCLegacyDatasetZoom): {M} images")

# Encode
target_emb = encode_encoder2(targets)
embed_dim = target_emb.shape[1]

sameins_flat = sameins.view(N * N_max, C, H, W)
samein_emb_flat = encode_encoder2(sameins_flat)
samein_emb = samein_emb_flat.view(N, N_max, embed_dim)

bunch_emb = encode_encoder2(legacy_bunch).cpu().numpy()

# --- Instrument neighbors: target[i] ↔ sameins[i][k] for valid k ---
neighbor_dists = []
for i in range(N):
    valid = masks[i] > 0
    if valid.any():
        d = torch.norm(target_emb[i : i + 1] - samein_emb[i], dim=-1)
        d_valid = d[valid].cpu().numpy()
        neighbor_dists.append(d_valid)
neighbor_dists = np.concatenate(neighbor_dists) if neighbor_dists else np.array([])
avg_neighbor_dist = float(np.mean(neighbor_dists)) if len(neighbor_dists) > 0 else float('nan')
n_neighbor_pairs = len(neighbor_dists)

# --- Instrument non-neighbors: legacy_bunch[i] vs legacy_bunch[j] for i != j ---
np.random.seed(42)
non_neighbor_dists = []
for _ in range(NUM_NON_NEIGHBOR_SAMPLES):
    i, j = np.random.randint(0, M, 2)
    if i != j:
        non_neighbor_dists.append(np.linalg.norm(bunch_emb[i] - bunch_emb[j]))
non_neighbor_dists = np.array(non_neighbor_dists)
avg_non_neighbor_dist = float(np.mean(non_neighbor_dists))

# --- Report ---
print("\n" + "=" * 60)
print("Instrument latent space (encoder_2): neighbor vs non-neighbor distances")
print("=" * 60)
print(f"\n1) Instrument neighbors (target ↔ sameins, NeighborsPrecomputedDataset):")
print(f"   Average distance: {avg_neighbor_dist:.4f}")
print(f"   Number of (target, samein) pairs: {n_neighbor_pairs}")

print(f"\n2) Instrument non-neighbors, same survey (legacy_bunch[i] vs legacy_bunch[j], i≠j, HSCLegacyDatasetZoom):")
print(f"   Average distance: {avg_non_neighbor_dist:.4f}")
print(f"   Number of pairs sampled: {len(non_neighbor_dists)}")

ratio = avg_neighbor_dist / avg_non_neighbor_dist
print(f"\n3) Ratio (neighbors / non-neighbors): {ratio:.4f}")
print(f"   → Neighbors are {'closer' if ratio < 1 else 'farther'} in instrument space.")
print("=" * 60)
