#!/usr/bin/env python3
"""Prepare a new CTS candidate; calibration/startup receipts are never inherited."""
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
from export_cts_checkpoint import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Each CTS checkpoint requires a new release directory")
    # Do not recursively copy a release into its own source checkout.
    if ROOT == destination or ROOT in destination.parents:
        raise ValueError("Use an output outside this source checkout")
    destination.mkdir(parents=True)
    for name in ("src", "scripts", "deploy", "tests", "tests_ros", "docs"):
        if (ROOT / name).is_dir():
            shutil.copytree(ROOT / name, destination / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "README.md", destination / "README.md")
    (destination / "runtime").mkdir()
    (destination / "STAGED_NOT_ACCEPTED.txt").write_text(
        "CTS nominal model candidate: offline parity does not authorize hardware. "
        "Mandatory landing controller contact/actuator binding is unavailable.\n")
    record = export(args.checkpoint, args.training, destination / "models/v1")
    # Production board preparation supplies site-local calibration and links.
    # The normal GOS mapper still publishes terrain v1; no raw-command field is
    # added to terrain messages because the operator command is AOS-local.
    print(json.dumps({"release": str(destination), "model": record,
                      "hardware_ready": False, "commissioning_inherited": False}, indent=2))


if __name__ == "__main__":
    main()
