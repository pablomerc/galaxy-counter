#!/usr/bin/env bash
#SBATCH --job-name=check_encoders
#SBATCH --output=/home/fontirro/logs/check_encoders_%j.out
#SBATCH --error=/home/fontirro/logs/check_encoders_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=compl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-phase1"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/check_encoders.py \
    --checkpoint "${CKPT_DIR}/best-epoch=21-step=98000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs.h5"
