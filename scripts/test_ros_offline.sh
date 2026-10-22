#!/usr/bin/env bash
# Local ROS callback integration only. No SDK or motor-command publishers.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f install/setup.bash ]] || { echo 'Build the local message package first.' >&2; exit 2; }
set +u
source install/setup.bash
set -u
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=176
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src/m20_heightmap${PYTHONPATH:+:$PYTHONPATH}"
unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ulimit -v 2097152
exec python3 -m pytest -q tests_ros --basetemp="$PWD/.scratch/ros_offline_$$" "$@"
