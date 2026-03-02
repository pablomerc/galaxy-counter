#!/usr/bin/env bash
set -euo pipefail
python -m galaxy_counter.train.neighbours_train "$@"
