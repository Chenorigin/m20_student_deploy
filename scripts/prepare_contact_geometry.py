#!/usr/bin/env python3
"""Export the original M20 URDF's contact geometry/inertials; no robot access."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/m20_heightmap'))
from m20_heightmap.contact_estimator import save_geometry

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    save_geometry(args.urdf, args.output)
