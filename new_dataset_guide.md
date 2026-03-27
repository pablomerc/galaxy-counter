# Using galaxy-counter on a New Dataset

This guide walks you through adapting this codebase to your own data.
The model is a **conditional flow-matching** model that learns to translate
images of the same object across two different instruments/surveys.
It uses **spatial neighbors** (nearby galaxies observed by the same instrument)
as an additional conditioning signal.

The five steps below mirror the pipeline used for the HSC × Legacy Survey dataset
this repo was built on.

---

## Step 1 — Get your data

You need **paired images**: the same set of objects observed by two different
instruments (e.g. two telescopes, two filters, two epochs). You also need a way
to identify, for each object, which other objects in the dataset are its spatial
neighbors (i.e. galaxies that happen to appear nearby on the sky and were therefore
observed under similar instrumental conditions).

In our case the data came from HuggingFace:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Smith42/legacysurvey_hsc_crossmatched",
    repo_type="dataset",
    local_dir="/path/to/data",
    allow_patterns=["data/train-*.parquet"],
)
```

For your dataset the source will be different, but the goal is the same: end up
with a set of cross-matched image pairs.

---

## Step 2 — Preprocess your images

Raw astronomical images span a very large dynamic range and different surveys use
different flux zeropoints. Before training you need to normalize all images into a
roughly comparable range.

Look at `src/galaxy_counter/preprocessing/image_preprocessing.py` for the full
implementation. The pipeline we apply is:

1. **CenterCrop** — crop to a fixed spatial size (we use 48×48 or 96×96 px).
2. **Clamp** — per-band clip to physically reasonable flux limits. This removes
   cosmic rays and saturated pixels. The limits (`BAND_CENTER_MAX`) are
   survey-specific; set them based on your data.
3. **Zeropoint rescaling** — if your two instruments use different flux zeropoints,
   divide one of them by the appropriate scale factor so both are on the same scale.
   In our case HSC images are divided by `10^((27.0 − 22.5) / 2.5)` to match the
   Legacy Survey scale.
4. **Range compression** — apply `arcsinh(x / 0.01) × 0.01 × 10.0` to compress
   the dynamic range while preserving sign. This makes faint and bright structures
   more equally visible to the network.

After preprocessing, compute the **per-survey mean and std** over a representative
sample and store them in `NORM_DICT` (see `src/galaxy_counter/data.py`). These are
used inside the Dataset classes to normalize images to roughly zero mean / unit
variance at load time.

Write a script that reads your raw data, applies these transforms, and saves the
result to an HDF5 file:

```python
import h5py, numpy as np

with h5py.File("my_preprocessed.h5", "w") as f:
    f.attrs["num_images"] = N
    f.attrs["crop_size"]  = 48
    f.attrs["num_channels"] = C
    f.create_dataset("instrument_a_images", shape=(N, C, 48, 48), dtype=np.float32)
    f.create_dataset("instrument_b_images", shape=(N, C, 48, 48), dtype=np.float32)
```

You may also need a **zoom** step if the two instruments have different pixel scales /
field of view. See `zoom_legacy_image` in `src/galaxy_counter/data.py` for the
implementation we use (center-crop to 64% then bilinear upsample back).

---

## Step 3 — Build a custom data loader

> **Why spatial neighbors?** We did this thing with the spatial neighbors because we
> realized they will share observing conditions and have more strongly correlated
> instrument properties. But for a first test you can try running this training with
> just randomly selected instrument pairs and that should still give you good
> reconstructions (but not very useful instrument latents).

The training loop expects batches of the form:

```
(target, same_galaxy, same_instrument_neighbors, neighbor_mask, metadata)
```

- `target` — the anchor image `(C, H, W)` from instrument A or B
- `same_galaxy` — the same object seen by the other instrument `(C, H, W)`
- `same_instrument_neighbors` — up to K images of *other* objects from the same
  instrument, observed nearby on the sky `(K, C, H, W)`
- `neighbor_mask` — boolean mask of length K (True = real neighbor, False = padding)
- `metadata` — dict with any extra info you want to log

See `src/galaxy_counter/neighbors.py` for the full implementation:

- `NeighborsDataset` — reads raw images on-the-fly from an HDF5 file and applies
  preprocessing at load time. Slower but memory-light.
- `NeighborsPrecomputedDataset` — reads pre-baked, pre-padded batches from HDF5
  shards. Much faster for training; requires a preprocessing step up front.
- `collate_neighbors` — custom collate function that pads variable-length neighbor
  lists to the same size within a batch.

For a new dataset you will need to:

1. Build an HDF5 file with the following layout (this is what `NeighborsDataset`
   reads from):

   ```
   f['images_a']          # raw images, instrument A — shape (N, C_a, H, W)
   f['images_b']          # raw images, instrument B — shape (N, C_b, H, W)
   f['neighbor_idx_a']    # kNN indices in instrument A — shape (N, K), -1 = no neighbor
   f['neighbor_idx_b']    # kNN indices in instrument B — shape (N, K)
   f['source_type']       # int8, shape (N,) — used to filter valid pairs
   ```

2. Compute the neighbor indices. In our case this was done as a one-off by a
   collaborator (Carol) — the script is not in this repo. The simplest approach for
   a new dataset is a pixel-level kNN: flatten each image, run sklearn
   `NearestNeighbors` or a FAISS index, and store the resulting indices. We
   pre-compute these once and cache them in the HDF5.

3. Write a `Dataset` subclass modeled on `NeighborsDataset` that reads from your
   HDF5 and calls your preprocessing function inside `__getitem__`.

---

## Step 4 — (Optional) Pre-bake preprocessed batches for fast training

In our case reading the h5 file back and forth according to the neighbor indexes was a huge bottleneck, which is why we "pre-computed" some batches for training  so that at training time we just had to read a file sequentially. This is not necessary if you don't end up using an ellaborate neighbor logic like we did.


Reading raw images and preprocessing on-the-fly is a bottleneck at scale.
If you have a large dataset, pre-bake the preprocessed batches into shards:

1. Run a script that iterates through your `NeighborsDataset` with multiple DataLoader
   workers and writes each batch to a compressed HDF5 shard file. Each shard stores:

   ```
   f['targets']         # (N, C, H, W)
   f['samegals']        # (N, C, H, W)
   f['sameins']         # (N, K, C, H, W)  — padded to global max_neighbors
   f['neighbor_masks']  # (N, K)  bool
   ```

2. Merge the shards into an HDF5 Virtual Dataset (a thin index file, no data copy):

   ```python
   import h5py, glob

   files = sorted(glob.glob("shards/shard_*.h5"))
   # ... see save_neighbors_gemini_merge pattern for the VDS creation loop
   ```

3. Point `GALAXY_COUNTER_PRECOMPUTED_H5` at the VDS file and use
   `NeighborsPrecomputedDataset` in the training script.

---

## Step 5 — Adapt the training script

The model is implemented in `src/galaxy_counter/models/double_train_fm_neighbors.py`
(`ConditionalFlowMatchingModule`). It is a PyTorch Lightning module and should need
minimal changes to work with new data. The parts you are most likely to touch:

- **`in_channels` and `cond_channels`** — set to the number of bands in your
  instrument A and B images. Currently both are 4.
- **`image_size`** — set to your crop size (48 or 96).
- **`cross_attention_dim`** — embedding dimension for the conditioning encoders; 16
  works well at 48×48.

The training entrypoint is `src/galaxy_counter/train/neighbours_train.py`, launched
via:

```bash
scripts/run_neighbors_train.sh
# → python -m galaxy_counter.train.neighbours_train
```

Key environment variables to set:

```bash
export GALAXY_COUNTER_PRECOMPUTED_H5=/path/to/neighbours_vds.h5
export GALAXY_COUNTER_NEIGHBORS_H5=/path/to/neighbours_v2.h5
export GALAXY_COUNTER_CHECKPOINT_DIR=./outputs
```

Or edit the config directly:
`configs/neighbors_default.json` — set `data.precomputed_h5`, `data.neighbors_h5`,
`model.in_channels`, `model.cond_channels`, `model.image_size`.

For multi-GPU training on SLURM, see `scripts/test_galc_gpu.sh` as a template.

---

## Quick checklist

- [ ] Paired images downloaded and accessible
- [ ] `BAND_CENTER_MAX` updated for your survey bands in `image_preprocessing.py`
- [ ] Zeropoint rescaling adapted (or removed) for your flux scale
- [ ] (OPTIONAL) Raw images + kNN neighbor indices written to `neighbours.h5`
- [ ] `NORM_DICT` mean/std measured from your preprocessed data and updated in `data.py` / `neighbors.py`
- [ ] `in_channels`, `cond_channels`, `image_size` set in config
- [ ] (Optional) pre-computed batches + VDS created for fast training
- [ ] Training launched with correct data paths
