#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src/m20_heightmap${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -m pytest -q tests
python3 -m pyflakes src/m20_heightmap/m20_heightmap scripts/*.py tests/*.py
