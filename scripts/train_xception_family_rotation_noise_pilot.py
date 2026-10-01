#!/usr/bin/env python3
"""Train isolated M7 Mixed-JPEG + weak-noise seed42 pilots without altering baselines."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from train_xception_family_coverage import FIXED_BUDGETS, validate_manifest
from train_xception_family_rotation import completed, fake_methods


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=["S2", "S3", "S4"], default=["S2", "S3", "S4"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    trainer = root / "scripts/train_xception_letterbox.py"
    for selection in args.selections:
        manifest = args.manifest_root / selection.lower() / "m7_seed42.csv"
        frame = validate_manifest(manifest, fake_methods(config, selection, "M7"), FIXED_BUDGETS)
        missing = [path for path in frame.source_path if not (args.data_root / path).is_file()]
        if missing:
            raise FileNotFoundError(f"{selection}: {len(missing)} training/validation images missing; first: {missing[0]}")
        run = args.output_root / f"xception_{selection.lower()}_m7_mixed_jpeg_noise_p25_std2_seed42"
        last, best = run / "last.pt", run / "best.pt"
        print(f"\n===== {selection} / M7 / Mixed-JPEG + noise =====", flush=True)
        if completed(last, best, args.epochs, args.patience):
            print(f"Already complete: {best}", flush=True)
            continue
        command = [sys.executable, "-u", str(trainer),
            "--data-root", str(args.data_root), "--manifest", str(manifest),
            "--output-dir", str(run), "--split-protocol", "family_rotation_mixed_jpeg_noise_pilot_v1",
            "--model", "xception", "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay), "--patience", str(args.patience),
            "--seed", "42", "--experiment-family", "family_rotation_noise_pilot_v1",
            "--condition-name", f"{selection.lower()}_m7_mixed_jpeg_noise_p25_std2",
            "--canonical-size", "256", "--jpeg-quality", "95", "--jpeg-subsampling", "2",
            "--train-jpeg-qualities", "75", "80", "85", "90", "95",
            "--train-noise-probability", "0.25", "--train-noise-max-std", "2.0", "--persistent-progress"]
        if last.is_file():
            command.extend(["--resume", str(last)])
            print(f"Resuming: {last}", flush=True)
        else:
            print("Starting from ImageNet-pretrained Xception", flush=True)
        subprocess.run(command, check=True)
        if not best.is_file():
            raise FileNotFoundError(f"Training ended without best.pt: {best}")


if __name__ == "__main__":
    main()
