#!/usr/bin/env bash
set -euo pipefail
python -m galaxy_counter.downstream.predict_all "$@"
