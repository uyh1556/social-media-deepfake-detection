#!/usr/bin/env python3
"""Validate and launch one M0-M7 family-coverage Xception run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


FIXED_BUDGETS = {
    "train": {"real": 19_800, "fake": 19_800},
    "val": {"real": 4_680, "fake": 4_680},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=[f"M{i}" for i in range(8)],
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/family_coverage_v1/conditions.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_condition(path: Path, condition_id: str) -> tuple[dict, dict]:
    if condition_id == "M0":
        path = path.with_name("m0.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    try:
        condition = config["conditions"][condition_id]
    except KeyError as error:
        raise ValueError(f"Unknown condition: {condition_id}") from error
    return config, condition


def validate_manifest(
    path: Path,
    expected_methods: list[str],
    expected_budgets: dict,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen manifest is not ready: {path}. "
            "Finish the family_coverage_v1 dataset/split preparation first."
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"sample_id", "split", "label", "method", "source_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in manifest")

    development = frame[frame["split"].isin(["train", "val"])]
    if development.empty:
        raise RuntimeError("Manifest has no train/val rows")
    real_methods = set(development.loc[development["label"] == "real", "method"])
    fake_methods = set(development.loc[development["label"] == "fake", "method"])
    if real_methods != {"original"}:
        raise RuntimeError(f"Expected only original real data, found: {sorted(real_methods)}")
    if fake_methods != set(expected_methods):
        raise RuntimeError(
            "Condition/manifest fake methods do not match: "
            f"expected={sorted(expected_methods)}, actual={sorted(fake_methods)}"
        )

    for split in ("train", "val"):
        label_counts = development[development["split"] == split].groupby(
            "label"
        ).size()
        expected_counts = expected_budgets[split]
        actual_counts = {
            label: int(label_counts.get(label, 0))
            for label in ("real", "fake")
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Fixed budget mismatch in {split}: "
                f"expected={expected_counts}, actual={actual_counts}"
            )
        counts = (
            development[
                (development["split"] == split)
                & (development["label"] == "fake")
            ]
            .groupby("method")
            .size()
        )
        if set(counts.index) != set(expected_methods):
            raise RuntimeError(f"Missing {split} fake method: {counts.to_dict()}")
        if int(counts.max() - counts.min()) > 1:
            raise RuntimeError(
                f"Fake method quotas are not equal in {split}: {counts.to_dict()}"
            )
    return frame


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    protocol, condition = load_condition(
        args.conditions_config.resolve(), args.condition
    )
    manifest = validate_manifest(
        args.manifest,
        condition["seen_fake_methods"],
        protocol.get("budgets", FIXED_BUDGETS),
    )
    missing_images = [
        value
        for value in manifest["source_path"]
        if not (args.data_root / value).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Manifest images are not fully extracted. First missing paths: "
            + ", ".join(missing_images[:5])
        )

    run_name = (
        f"xception_{args.condition.lower()}_{condition['name']}_"
        f"letterbox299_family_coverage_v1_seed{args.seed}"
    )
    run_dir = args.output_root / run_name
    last_checkpoint = run_dir / "last.pt"
    trainer = Path(__file__).resolve().with_name("train_xception_letterbox.py")
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
        protocol["protocol"],
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
        "family_coverage_v1",
        "--condition-name",
        f"{args.condition.lower()}_{condition['name']}",
        "--persistent-progress",
    ]
    if last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint)])
        print(f"Resuming {args.condition}: {last_checkpoint}", flush=True)
    else:
        print(
            f"Starting {args.condition} from ImageNet-pretrained Xception",
            flush=True,
        )
    print("Run directory:", run_dir, flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
