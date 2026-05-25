#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_train_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:1             # request 1 GPU; increase and set N_GPUS in train.py for multi-GPU
#SBATCH --time=5:00:00
#SBATCH --partition=pscompl      # change to your GPU partition
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/train.py
