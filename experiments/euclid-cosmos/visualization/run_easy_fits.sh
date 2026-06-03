#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_easy_fits
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_easy_fits_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_easy_fits_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      
#SBATCH --mem=8GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=pscompl      # change to your GPU partition
set -euo pipefail



for idx in $(seq 0 311105); do
    python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/visualization/easy_fits.py --idx $idx
done
