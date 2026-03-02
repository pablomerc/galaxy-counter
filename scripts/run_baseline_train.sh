#!/usr/bin/env bash
set -euo pipefail
python -m galaxy_counter.baseline.train_dual_encoder_contrastive "$@"
