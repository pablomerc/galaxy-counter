#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_generate
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_generate_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_generate_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:30:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-phase1-v2"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/generate.py \
    --checkpoint "${CKPT_DIR}/best-epoch=37-step=175000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5" \
    --out-dir    "/n03data/fontirro/plots_model/euclid-cosmos-phase1-v2/generated" \
    --indices    "${CKPT_DIR}/test_indices.npy" \
    --n-images   16 \
    --direction  both
