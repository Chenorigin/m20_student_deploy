#!/usr/bin/env bash
# Temporary runtime override; only the operator may invoke apply/rollback.
set -euo pipefail
[[ "$(cat /opt/robot/os_type)" == CA9B_GOS ]] || exit 2
mode=${1:-inspect}
dropin=/run/systemd/system/ptp4l.service.d/90-m20-heightmap.conf
case "$mode" in
  inspect) sudo pmc -u -b 0 'GET PORT_DATA_SET' 'GET CURRENT_DATA_SET' 'GET TIME_PROPERTIES_DATA_SET'; exit ;;
  apply)
    [[ -t 0 ]] || exit 2
    echo "Pause your perception applications first. This restarts only GOS ptp4l; it does not send motor commands."
    read -r -p "Type REPAIR_GOS_CLOCK to apply temporary hardware timestamp timeout: " answer
    [[ "$answer" == REPAIR_GOS_CLOCK ]] || exit 2
    sudo -v
    [[ ! -e "$dropin" ]] || { echo "Override already exists; inspect it instead of overwriting." >&2; exit 2; }
    sudo mkdir -p "$(dirname "$dropin")"
    printf '%s\n' '[Service]' 'ExecStart=' 'ExecStart=/usr/sbin/ptp4l -H -E -4 -m -i eth0 -s --tx_timestamp_timeout=20' | sudo tee "$dropin" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ptp4l.service
    echo "Temporary candidate applied, not yet accepted. Inspect state and run the raw timestamp preflight. If still FAULTY, rollback and do not start control."
    ;;
  rollback)
    if [[ -e "$dropin" ]]; then sudo rm "$dropin"; sudo systemctl daemon-reload; sudo systemctl restart ptp4l.service; fi
    ;;
  *) echo "Usage: $0 inspect|apply|rollback" >&2; exit 2 ;;
esac
