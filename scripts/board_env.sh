#!/usr/bin/env bash
# Source this file; it prepares only this shell's environment, starts nothing.
M20_RELEASE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
set +u
source /opt/ros/foxy/setup.bash
source "$M20_RELEASE_ROOT/install/setup.bash"
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$M20_RELEASE_ROOT/runtime/dds_udp.xml"
