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
- `scripts/`: runnable entrypoint wrappers
- `configs/paths.example.yaml`: example external data paths

## Setup
```bash
pip install -r requirements.txt
```

## Environment variables (optional)
- `GALAXY_COUNTER_NEIGHBORS_H5`: neighbors HDF5 path
- `GALAXY_COUNTER_PRECOMPUTED_H5`: precomputed neighbor batches HDF5 path
- `GALAXY_COUNTER_CHECKPOINT_DIR`: output checkpoint directory

## Common commands
```bash
scripts/run_neighbors_train.sh
scripts/run_baseline_train.sh
scripts/run_downstream_prepare.sh --checkpoint /path/to/model.ckpt --module double_train_fm_neighbors.py --suffix myrun
scripts/run_downstream_predict.sh --suffix myrun
scripts/run_neighbor_search.sh --checkpoint /path/to/model.ckpt --module double_train_fm_neighbors.py --suffix myrun
```

## Notes
- Gemini helper scripts were excluded.
- Large generated artifacts are gitignored (`*.ckpt`, `*.h5`, plots, logs, `wandb/`).
