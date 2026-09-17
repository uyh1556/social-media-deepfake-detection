#!/usr/bin/env python3
"""Train M0-M7 Xception with label-independent mixed JPEG augmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from train_xception_family_coverage import (
    FIXED_BUDGETS,
    load_condition,
    validate_manifest,
)


EXPERIMENT_FAMILY = "family_coverage_jpeg_mixed_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=[f"M{index}" for index in range(8)],
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--mixed-config",
        type=Path,
        default=project_root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    return parser.parse_args()


def load_protocol(path: Path, condition_id: str, seed: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != EXPERIMENT_FAMILY:
        raise ValueError("Unexpected mixed-JPEG protocol.")
    if condition_id not in protocol.get("models", []):
        raise ValueError(f"Condition is not enabled: {condition_id}")
    if seed not in protocol.get("training_seeds", []):
        raise ValueError(f"Training seed is not enabled: {seed}")
    preprocessing = protocol["preprocessing"]
    expected = {
        "name": "canonical256_jpegmixed_letterbox299",
        "canonical_size": 256,
        "train_jpeg_qualities": [75, 80, 85, 90, 95],
        "train_jpeg_sampling": "uniform",
        "validation_jpeg_quality": 95,
        "jpeg_subsampling": 2,
        "jpeg_optimize": False,
        "jpeg_progressive": False,
        "model_input_size": 299,
    }
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(
                f"Mixed-JPEG mismatch for {key}: "
                f"expected={value}, actual={preprocessing.get(key)}"
            )
    return protocol


def run_name(condition_id: str, condition: dict, seed: int) -> str:
    return (
        f"xception_{condition_id.lower()}_{condition['name']}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
        f"seed{seed}"
    )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    conditions_config = args.conditions_config.resolve()
    mixed_config = args.mixed_config.resolve()
    protocol = load_protocol(mixed_config, args.condition, args.seed)
    family_protocol, condition = load_condition(
        conditions_config, args.condition
    )
    validate_manifest(
        args.manifest,
        condition["seen_fake_methods"],
        family_protocol.get("budgets", FIXED_BUDGETS),
    )
    expected_hash = protocol["manifest_sha256"][args.condition]
    actual_hash = sha256_file(args.manifest)
    if actual_hash != expected_hash:
        raise ValueError(
            "Training manifest hash mismatch: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    run_dir = args.output_root / run_name(
        args.condition, condition, args.seed
    )
    last_checkpoint = run_dir / "last.pt"
    preprocessing = protocol["preprocessing"]
    trainer = Path(__file__).resolve().with_name(
        "train_xception_letterbox.py"
    )
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--data-root",
        str(args.data_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(run_dir),
        "--split-protocol",
        EXPERIMENT_FAMILY,
        "--model",
        "xception",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
        "--experiment-family",
        EXPERIMENT_FAMILY,
        "--condition-name",
        f"{args.condition.lower()}_{condition['name']}_jpeg_mixed",
        "--canonical-size",
        str(preprocessing["canonical_size"]),
        "--jpeg-quality",
        str(preprocessing["validation_jpeg_quality"]),
        "--train-jpeg-qualities",
        *[str(value) for value in preprocessing["train_jpeg_qualities"]],
        "--jpeg-subsampling",
        str(preprocessing["jpeg_subsampling"]),
        "--persistent-progress",
    ]
    if preprocessing["jpeg_optimize"]:
        command.append("--jpeg-optimize")
    if preprocessing["jpeg_progressive"]:
        command.append("--jpeg-progressive")
    if last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint)])
        print(f"Resuming {args.condition}/seed{args.seed}: {last_checkpoint}")
    else:
        print(
            f"Starting {args.condition}/seed{args.seed} mixed-JPEG run "
            "from ImageNet-pretrained Xception"
        )
    print("Run directory:", run_dir, flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
