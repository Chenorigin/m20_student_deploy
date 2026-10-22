#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile=${1:-v1}
[[ "$profile" == v1 ]] || { echo "This student-v3.5 release requires profile v1 (789D)." >&2; exit 2; }
[[ "$(cat /opt/robot/os_type)" == CA9B_GOS ]] || exit 2
[[ "$EUID" == 0 ]] || { echo "Use sudo -E; native input verifies the root factory process." >&2; exit 2; }
source scripts/board_env.sh
python3 scripts/validate_v35_bundle.py
python3 - "$profile" <<'PY'
import sys
from m20_heightmap.runtime import load_config
load_config('runtime/heightmap_'+sys.argv[1]+'.local.json')
PY
# A copied release must not create a second mapper beside the old release.
python3 - <<'CHECK'
from pathlib import Path
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try:
        args=(entry/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError): continue
    if any(arg.decode(errors='ignore').endswith('/gos_mapper') or arg==b'gos_mapper' for arg in args):
        raise SystemExit('Another GOS mapper is running (PID '+entry.name+'); stop its operator session first.')
CHECK
mkdir -p runtime
exec 9>runtime/perception.lock
flock -n 9 || { echo "This release's perception supervisor is already running." >&2; exit 2; }
children=()
# AF_UNIX paths are limited to 107 bytes; release names must not consume that
# budget. mktemp creates this root-owned private directory with mode 0700.
socket_dir=$(mktemp -d /run/m20-native.XXXXXXXX)
socket=$socket_dir/s
stop_children(){ for pid in "${children[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; wait || true; rm -f -- "$socket"; rmdir -- "$socket_dir"; }
trap 'status=$?; echo "Perception supervisor exit status=$status" >&2; stop_children' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
site=$PWD/deploy/site.json
lio=$(readlink -f /home/user/staircase_gos_ws/lio_release)
python3 scripts/verify_lio_release.py "$lio" "$site"
export LD_LIBRARY_PATH="$lio/lib:${LD_LIBRARY_PATH:-}"
export M20_TERRAIN_WORKER_CPUS=6
# Keep native transport off the fit worker and leave each heavy Python/LIO
# stage its own big core. The lightweight native transport and pose adapter
# use separate little cores; factory process affinity is never changed here.
setsid env -u FASTRTPS_DEFAULT_PROFILES_FILE -u FASTDDS_DEFAULT_PROFILES_FILE LD_LIBRARY_PATH=/usr/local/lib taskset -c 2 install/m20_native_inputs/lib/m20_native_inputs/native_lidar_input "$socket" >runtime/native_cloud.log 2>&1 &
children+=("$!")
setsid taskset -c 7 ros2 run m20_heightmap gos_native_inputs --ros-args -p "socket_path:=$socket" -p "config:=$PWD/runtime/heightmap_$profile.local.json" >runtime/native_inputs.log 2>&1 &
children+=("$!")
setsid taskset -c 5 "$lio/bin/super_lio_node" --ros-args --params-file deploy/gos_lio.yaml \
  -r /lio/imu/odom:=/m20/lio/imu_odom -r /lio/robo/odom:=/m20/lio/base_candidate \
  -r /lio/odom:=/m20/lio/odom -r /lio/cloud_world:=/m20/lio/cloud_world \
  -r /lio/path:=/m20/lio/path -r /tf:=/m20/lio/tf >runtime/lio.log 2>&1 &
children+=("$!")
setsid taskset -c 3 ros2 run m20_heightmap base_pose_adapter --ros-args -p "site_config:=$site" >runtime/pose.log 2>&1 &
children+=("$!")
setsid taskset -c 4 ros2 run m20_heightmap gos_mapper --ros-args -p "config:=$PWD/runtime/heightmap_$profile.local.json" >runtime/heightmap.log 2>&1 &
children+=("$!")
echo "Perception started by operator. Logs: $PWD/runtime. Ctrl-C stops only these child processes."
if wait -n; then status=0; else status=$?; fi
echo "Perception child exited status=$status; stopping this supervisor's children" >&2
for pid in "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then echo "Exited child PID=$pid" >&2; fi
done
exit 1
