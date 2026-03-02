#!/usr/bin/env bash
set -euo pipefail
python -m galaxy_counter.neighbor_search.make_latents_all "$@"
