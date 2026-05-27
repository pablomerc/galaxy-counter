#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_test
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_test_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:30:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/testing.py \
    --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=21-step=98000.ckpt \
    --h5 /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
    --out /n03data/fontirro/checkpoints/euclid-cosmos-phase1/test.png \
    --n-samples 8 \
    --num-steps 100
