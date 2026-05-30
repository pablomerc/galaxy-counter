#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_umap
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_umap_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_umap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/visualization/umap_latent.py \
    --checkpoint "${CKPT_DIR}/best-epoch=37-step=175000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5" \
    --out        "/n03data/fontirro/plots_model/euclid-cosmos-phase1-v2/umap_test.png" \
    --out-cutouts "/n03data/fontirro/plots_model/euclid-cosmos-phase1-v2/umap_test_cutouts.png" \
    --indices    "${CKPT_DIR}/test_indices.npy"