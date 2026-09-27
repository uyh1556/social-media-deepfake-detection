#!/usr/bin/env python3
"""Run one family-rotation selection/protocol for multiple training seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=[f"S{i}" for i in range(1, 7)], required=True)
    parser.add_argument("--protocol", choices=("fixed_q95", "mixed_jpeg"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=[f"M{i}" for i in range(1, 8)])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = Path(__file__).resolve().with_name("train_xception_family_rotation.py")
    for seed in args.seeds:
        print(
            f"\n######## {args.selection} / {args.protocol} / seed {seed} ########",
            flush=True,
        )
        command = [
            sys.executable, "-u", str(trainer),
            "--selection", args.selection,
            "--protocol", args.protocol,
            "--data-root", str(args.data_root),
            "--manifest-root", str(args.manifest_root),
            "--output-root", str(args.output_root),
            "--models", *args.models,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--seed", str(seed),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
