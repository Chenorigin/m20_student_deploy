#!/usr/bin/env python3
"""Check the explicit execution mode before SDK robot initialization."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/m20_heightmap"))
from m20_heightmap.cts import require_hardware_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config else None
    if cfg is not None:
        selected = Path(cfg["aos"]["model_contract"])
        selected = selected if selected.is_absolute() else args.config.parent / selected
        if selected.resolve() != args.contract.resolve():
            raise ValueError("Selected runtime and model contract differ")
    mode = require_hardware_runtime(json.loads(args.contract.read_text()), cfg)
    if mode:
        print(json.dumps(dict(cts_execution_mode=mode, landing_control_enabled=False,
                              policy="CTS student encoder + shared decoder", pd="original nominal PD")))


if __name__ == "__main__":
    main()
