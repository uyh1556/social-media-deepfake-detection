#!/usr/bin/env python3
"""Train one S2-S6 selection's M1-M7 under fixed-Q95 or Mixed-JPEG."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from train_xception_family_coverage import FIXED_BUDGETS, validate_manifest


MODELS = tuple(f"M{i}" for i in range(1, 8))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=[f"S{i}" for i in range(2, 7)], required=True)
    parser.add_argument("--protocol", choices=("fixed_q95", "mixed_jpeg"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/family_rotation_v1/selections.json",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fake_methods(config: dict, selection: str, model: str) -> list[str]:
    letters = config["selections"][selection]
    return [
        config["families"][family][letter]
        for family in config["models"][model]["families"]
        for letter in letters
    ]


def completed(last: Path, best: Path, epochs: int, patience: int) -> bool:
    if not last.is_file() or not best.is_file():
        return False
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    return (
        int(checkpoint["epoch"]) >= epochs
        or int(checkpoint.get("epochs_without_improvement", 0)) >= patience
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prep = config["preprocessing"]
    trainer = Path(__file__).resolve().with_name("train_xception_letterbox.py")
    for model in args.models:
        print(
            f"\n===== Preparing {args.selection} / {model} / "
            f"{args.protocol} =====",
            flush=True,
        )
        methods = fake_methods(config, args.selection, model)
        manifest = (
            args.manifest_root / args.selection.lower() / f"{model.lower()}_seed42.csv"
        )
        frame = validate_manifest(manifest, methods, FIXED_BUDGETS)
        print(f"Manifest validated: {manifest}", flush=True)
        missing = [
            value for value in frame["source_path"]
            if not (args.data_root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"First missing image: {missing[0]}")
        print(f"Image paths verified: {len(frame)}", flush=True)

        model_name = config["models"][model]["name"]
        run_name = (
            f"xception_{args.selection.lower()}_{model.lower()}_{model_name}_"
            f"{args.protocol}_family_rotation_v1_seed{args.seed}"
        )
        run_dir = args.output_root / run_name
        last, best = run_dir / "last.pt", run_dir / "best.pt"
        print(f"===== Training {args.selection} / {model} / {args.protocol} =====", flush=True)
        if completed(last, best, args.epochs, args.patience):
            print(f"Skipping completed run: {best}", flush=True)
            continue
        command = [
            sys.executable, "-u", str(trainer),
            "--data-root", str(args.data_root),
            "--manifest", str(manifest),
            "--output-dir", str(run_dir),
            "--split-protocol", f"family_rotation_{args.protocol}_v1",
            "--model", "xception",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
            "--experiment-family", "family_rotation_v1",
            "--condition-name", f"{args.selection.lower()}_{model.lower()}_{model_name}_{args.protocol}",
            "--canonical-size", str(prep["canonical_size"]),
            "--jpeg-quality", str(prep["fixed_q95"]),
            "--jpeg-subsampling", str(prep["jpeg_subsampling"]),
            "--persistent-progress",
        ]
        if args.protocol == "mixed_jpeg":
            command.extend([
                "--train-jpeg-qualities",
                *[str(value) for value in prep["mixed_train_qualities"]],
            ])
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
