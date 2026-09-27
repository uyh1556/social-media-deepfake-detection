#!/usr/bin/env python3
"""Train one M7 with transfer-derived loss weights and Mixed-JPEG."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True, choices=[f"S{i}" for i in range(1, 7)])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest-root", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/family_rotation_v1/selections.json").read_text())
    manifest = args.manifest_root / args.selection.lower() / "m7_seed42.csv"
    weights = args.weights_dir / f"{args.selection.lower()}_weights.json"
    for path in (manifest, weights):
        if not path.is_file(): raise FileNotFoundError(path)
    run = f"xception_{args.selection.lower()}_m7_transfer_loss_weighted_mixed_jpeg_v1_seed{args.seed}"
    command = [sys.executable, "-u", str(Path(__file__).with_name("train_xception_letterbox.py")),
        "--data-root", str(args.data_root), "--manifest", str(manifest),
        "--output-dir", str(args.output_root / run),
        "--split-protocol", "family_rotation_transfer_loss_weighted_v1",
        "--model", "xception", "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--learning-rate", "1e-4", "--weight-decay", "1e-4", "--patience", "3",
        "--seed", str(args.seed), "--experiment-family", "family_rotation_transfer_loss_weighted_v1",
        "--condition-name", f"{args.selection.lower()}_m7_transfer_loss_weighted_mixed_jpeg",
        "--canonical-size", str(config["preprocessing"]["canonical_size"]),
        "--jpeg-quality", str(config["preprocessing"]["fixed_q95"]),
        "--jpeg-subsampling", str(config["preprocessing"]["jpeg_subsampling"]),
        "--train-jpeg-qualities", *map(str, config["preprocessing"]["mixed_train_qualities"]),
        "--method-loss-weights-json", str(weights), "--persistent-progress"]
    last = args.output_root / run / "last.pt"
    if last.is_file(): command.extend(["--resume", str(last)])
    subprocess.run(command, check=True)


if __name__ == "__main__": main()
