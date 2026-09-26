#!/usr/bin/env python3
"""Train/reuse the six single-method experts for one family rotation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, choices=[f"S{i}" for i in range(1, 7)])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_root / args.selection.lower()
    protocol_path = manifest_dir / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    trainer = Path(__file__).resolve().with_name("train_xception_method_transfer.py")

    for method in protocol["evaluation_methods"]:
        print(f"\n===== {args.selection} / {method} =====", flush=True)
        command = [
            sys.executable, "-u", str(trainer),
            "--method", method,
            "--data-root", str(args.data_root),
            "--manifest-dir", str(manifest_dir),
            "--output-root", str(args.output_root),
            "--protocol-config", str(protocol_path),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
