#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_umap
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_umap_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_umap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-phase1"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/umap_latent.py \
    --checkpoint "${CKPT_DIR}/best-epoch=21-step=98000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5" \
    --out        "/n03data/fontirro/plots_model/euclid-cosmos-phase1/umap.png" \
    --out-cutouts "/n03data/fontirro/plots_model/euclid-cosmos-phase1/umap_cutouts.png" \
    --n-samples 5000
