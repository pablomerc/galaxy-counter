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
TOTAL=311107
CHUNK=1000

python3 -c "
total, chunk = $TOTAL, $CHUNK
for s in range(0, total, chunk):
    print(s, min(s + chunk, total))
" | parallel -j "$SLURM_CPUS_PER_TASK" --colsep ' ' \
    python "$SCRIPT" --start {1} --end {2}
