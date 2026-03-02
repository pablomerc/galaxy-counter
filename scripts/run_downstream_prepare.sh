#!/usr/bin/env bash
set -euo pipefail
python -m galaxy_counter.downstream.prepare_all "$@"
