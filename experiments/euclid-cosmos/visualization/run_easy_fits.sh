#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_easy_fits
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_easy_fits_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_easy_fits_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=1-00:00:00
#SBATCH --partition=pscompl
set -euo pipefail

SCRIPT=/n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/visualization/easy_fits.py
#OUT_DIR=/n03data/fontirro/output_plots

seq 0 311105 | parallel -j "$SLURM_CPUS_PER_TASK" \
    python "$SCRIPT" --idx {}
