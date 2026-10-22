#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 1 ]]; then echo "Usage: bash scripts/run_gos.sh /absolute/config.local.json" >&2; exit 2; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
set +u
source install/setup.bash
set -u
exec ros2 run m20_heightmap gos_mapper --ros-args -p "config:=$1"
