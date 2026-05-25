#!/usr/bin/env bash
#SBATCH --job-name=build_hdf5
#SBATCH --output=/home/fontirro/logs/build_hdf5_%j.out
#SBATCH --error=/home/fontirro/logs/build_hdf5_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # set NUM_WORKERS in build_hdf5.py to match this
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/build_hdf5.py
