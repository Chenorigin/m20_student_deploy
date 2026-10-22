#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ROS_DISTRO:?Source the installed ROS 2 setup.bash first}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1
colcon build --executor sequential --parallel-workers 1 --packages-up-to m20_heightmap --cmake-args -DCMAKE_BUILD_TYPE=Release
