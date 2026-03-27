# galaxy-counter

Refactored core galaxy-counter code extracted from `tess-generative`, focused on:
- neighbors double-FM model
- contrastive baseline
- downstream evaluation
- neighbor search

## Layout
- `src/galaxy_counter/`: core package code
- `experiments/aion_benchmark/`: migrated experiment scripts
- `experiments/visualization/`: migrated visualization scripts
- `data/metadata/`: overlap CSVs required by downstream preparation
- `scripts/`: runnable entrypoint wrappers and data-preparation scripts
- `configs/paths.example.yaml`: example external data paths

## Setup

Install the package in editable mode (required before running any scripts — otherwise `galaxy_counter` will not be found as a module):
```bash
cd galaxy-counter   # or wherever you cloned the repo
conda activate torchenv
pip install -e .
```

This is a one-time step per environment. After this, all scripts and SLURM jobs that use `torchenv` will be able to import `galaxy_counter`.

## Environment variables (optional)
- `GALAXY_COUNTER_NEIGHBORS_H5`: neighbors HDF5 path
- `GALAXY_COUNTER_PRECOMPUTED_H5`: precomputed neighbor batches HDF5 path
- `GALAXY_COUNTER_CHECKPOINT_DIR`: output checkpoint directory

## Common commands
```bash
# Data preparation (run once before training)
python scripts/save_neighbor_shards.py   # pre-bake NeighborsDataset into HDF5 shards
python scripts/merge_neighbor_shards.py  # merge shards into train/val Virtual Datasets

# Training and evaluation
scripts/run_neighbors_train.sh
scripts/run_baseline_train.sh
scripts/run_downstream_prepare.sh --checkpoint /path/to/model.ckpt --module double_train_fm_neighbors.py --suffix myrun
scripts/run_downstream_predict.sh --suffix myrun
scripts/run_neighbor_search.sh --checkpoint /path/to/model.ckpt --module double_train_fm_neighbors.py --suffix myrun
```
