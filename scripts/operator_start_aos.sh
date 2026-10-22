#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile=${1:-v1}
[[ "$profile" == v1 ]] || { echo "This student-v3.5 release requires profile v1 (789D)." >&2; exit 2; }
startup=${2:-strict}
preflight_args=()
case "$startup" in
  strict) ;;
  supported-init) preflight_args+=(--supported-init --seconds 3) ;;
  *) echo "Choose strict or supported-init startup" >&2; exit 2 ;;
esac
if [[ -f STAGED_NOT_ACCEPTED.txt ]]; then
  cat STAGED_NOT_ACCEPTED.txt >&2
  echo "Model candidate not accepted; SDK startup blocked before hardware initialization." >&2
  exit 2
fi
[[ -t 0 ]] || { echo "Run from an operator's interactive terminal." >&2; exit 2; }
[[ "$(cat /opt/robot/os_type)" == CA9B_AOS ]] || exit 2
[[ "$EUID" == 0 ]] || { echo "Run inside AOS tmux with sudo -E; passive handset capture requires root." >&2; exit 2; }
ancestor=$PPID
durable=false
while [[ "$ancestor" =~ ^[0-9]+$ ]] && (( ancestor > 1 )); do
    comm=$(cat "/proc/$ancestor/comm")
    if [[ "$comm" == 'tmux: server' ]]; then durable=true;break;fi
    ancestor=$(awk '{print $4}' "/proc/$ancestor/stat")
done
[[ "$durable" == true ]] || { echo "Use a board-local tmux session; SSH must not own the controller lifetime." >&2;exit 2; }
source scripts/board_env.sh
python3 scripts/validate_v35_bundle.py --model-metadata
export M20_VERIFIED_MODEL_CONTRACT_SHA256=$(sha256sum models/v1/observation_contract.json | cut -d " " -f 1)
export M20_GAMEPAD_TRANSPORT=passive M20_GAMEPAD_INTERFACE=${M20_GAMEPAD_INTERFACE:-eth0}
install/m20_student_sdk/lib/m20_student_sdk/gamepad_observe "$M20_GAMEPAD_INTERFACE"
export M20_VERIFIED_VENDOR_PID=$(python3 scripts/check_vendor_owner.py --pid-only)
[[ "$M20_VERIFIED_VENDOR_PID" =~ ^[1-9][0-9]*$ ]] || exit 2
python3 scripts/live_preflight.py --profile "$profile" --control "${preflight_args[@]}"
echo "This starts the SDK, including offset initialization commands. Confirm stable support, physical E-stop and manufacturer SDK mode. Keep the verified vendor data process/basic_server running; its command endpoint must remain silent."
read -r -p "Type START_STUDENT to start; you will operate L1/L2/R2: " reply
[[ "$reply" == START_STUDENT ]] || exit 2
export M20_OPERATOR_START=START_STUDENT M20_IMU_SOURCE=direct M20_REMOTE_CONTROL=gamepad
export M20_CONTROL_TRACE_PATH="$PWD/runtime/control_trace_$(date +%s%N).jsonl"
export M20_TRACE_FULL_OBSERVATION=1
export M20_SDK_MODE_CONFIRMED=I_CONFIRMED
export M20_POLICY_PATH="$PWD/models/$profile/policy.onnx"
export M20_ROBOT_CONFIG="$PWD/deploy/robot.json" M20_HEIGHTMAP_CONFIG="$PWD/runtime/heightmap_$profile.local.json"
export M20_EXPECTED_GEOMETRY_HASH=$(python3 -c 'import os;from m20_heightmap.commission import geometry_hash;print(geometry_hash(os.getcwd(),os.environ["M20_HEIGHTMAP_CONFIG"]))')
exec taskset -c 4,5 install/m20_student_sdk/lib/m20_student_sdk/student_control
