#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_test
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_test_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=01:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-phase1"
PLOT_DIR="/n03data/fontirro/plots_model/euclid-cosmos-phase1"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/testing.py \
    --checkpoint "${CKPT_DIR}/best-epoch=21-step=88000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5" \
    --indices    "${CKPT_DIR}/test_indices.npy" \
    --out        "${PLOT_DIR}/test_results.png" \
    --n-plot     8 \
    --num-steps  100
