#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
board=$(cat /opt/robot/os_type)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MAKEFLAGS=-j1
set +u
source /opt/ros/foxy/setup.bash
set -u
case "$board" in
  CA9B_GOS) exec taskset -c 4 nice -n 15 colcon build --executor sequential --parallel-workers 1 --packages-up-to m20_heightmap --cmake-args -DCMAKE_BUILD_TYPE=Release ;;
  CA9B_AOS) exec taskset -c 4 nice -n 15 colcon build --executor sequential --parallel-workers 1 --packages-up-to m20_heightmap m20_student_sdk --cmake-args -DCMAKE_BUILD_TYPE=Release "-DM20_VENDOR_SDK=${M20_VENDOR_SDK:-/home/user/sdk_deploy/src/M20_sdk_deploy}" "-DM20_ORT_ROOT=${M20_ORT_ROOT:-/home/user/sdk_deploy/src/M20_sdk_deploy/third_party/onnxruntime/arm}" ;;
  *) echo "Unsupported board: $board" >&2; exit 2 ;;
esac
