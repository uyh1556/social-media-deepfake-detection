#!/usr/bin/env python3
"""Train fixed-budget nested method-holdout subset models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from train_baseline import sha256_file


PROTOCOL = "method_selection_nested_holdout_v1"
DEFAULT_STRATEGIES = ["random3", "family_balanced3", "transfer3", "all5"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-method", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--strategies", nargs="+", default=DEFAULT_STRATEGIES
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {path}")
    return config


def run_name(outer_slug: str, run_key: str, seed: int) -> str:
    return (
        f"xception_nested_holdout_{outer_slug}_{run_key}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
        f"seed{seed}"
    )


def completed_run(
    last_path: Path,
    best_path: Path,
    *,
    manifest_hash: str,
    condition_name: str,
    seed: int,
    epochs: int,
    patience: int,
) -> bool:
    if not last_path.is_file() or not best_path.is_file():
        return False
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("experiment_family") != PROTOCOL:
        raise ValueError(f"Unexpected existing run: {last_path}")
    if config.get("condition_name") != condition_name:
        raise ValueError(f"Condition mismatch: {last_path}")
    if config.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Manifest mismatch: {last_path}")
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch: {last_path}")
    return (
        int(checkpoint["epoch"]) >= epochs
        or int(checkpoint.get("epochs_without_improvement", 0)) >= patience
    )


def validate_manifest(path: Path, row: pd.Series, config: dict) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != row["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected_methods = set(row["selected_methods"].split(";"))
    if row["outer_holdout"] in expected_methods:
        raise RuntimeError("Outer holdout leaked into training methods")
    actual_methods = set(frame.loc[frame["label"] == "fake", "method"])
    if actual_methods != expected_methods:
        raise RuntimeError(
            f"Method mismatch: expected={expected_methods}, actual={actual_methods}"
        )
    for split in ("train", "val"):
        counts = frame[frame["split"] == split].groupby("label").size().to_dict()
        if counts != config["fixed_budgets"][split]:
            raise RuntimeError(f"Budget mismatch in {path}/{split}: {counts}")
        fake_counts = (
            frame[(frame["split"] == split) & (frame["label"] == "fake")]
            .groupby("method")
            .size()
        )
        if int(fake_counts.max() - fake_counts.min()) != 0:
            raise RuntimeError(f"Unequal method quotas: {fake_counts.to_dict()}")
    return actual_hash


def main() -> None:
    args = parse_args()
    config = load_config(args.protocol_config.resolve())
    if args.outer_method not in config["methods"]:
        raise ValueError(f"Unknown outer method: {args.outer_method}")
    if args.seed not in config["training_seeds"]:
        raise ValueError(f"Seed not enabled by protocol: {args.seed}")
    unknown = set(args.strategies) - set(config["strategies"])
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    data_root = args.data_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output_root = args.output_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    plan_path = manifest_dir / "selection_plan.csv"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False)
    selected = plan[
        (plan["outer_holdout"] == args.outer_method)
        & (plan["strategy"].isin(args.strategies))
    ].copy()
    if set(selected["strategy"]) != set(args.strategies):
        raise RuntimeError("Selection plan does not contain every requested strategy")

    outer_slug = config["methods"][args.outer_method]["slug"]
    trainer = Path(__file__).resolve().with_name("train_xception_letterbox.py")
    completed_keys = set()
    for row in selected.sort_values("strategy").itertuples(index=False):
        if row.run_key in completed_keys:
            print(
                f"Shared subset already handled: {row.strategy} -> {row.run_key}",
                flush=True,
            )
            continue
        manifest = manifest_dir / row.manifest
        manifest_hash = validate_manifest(
            manifest, pd.Series(row._asdict()), config
        )
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        missing = [
            value for value in frame["source_path"]
            if not (data_root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Training archive is incomplete. First missing paths: "
                + ", ".join(missing[:5])
            )
        name = run_name(outer_slug, row.run_key, args.seed)
        run_dir = output_root / name
        last_path = run_dir / "last.pt"
        best_path = run_dir / "best.pt"
        condition_name = f"holdout_{outer_slug}_{row.run_key}_jpeg_mixed"
        if completed_run(
            last_path,
            best_path,
            manifest_hash=manifest_hash,
            condition_name=condition_name,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
        ):
            print("Skipping completed:", best_path, flush=True)
            completed_keys.add(row.run_key)
            continue

        preprocessing = config["preprocessing"]
        command = [
            sys.executable,
            "-u",
            str(trainer),
            "--data-root", str(data_root),
            "--manifest", str(manifest),
            "--output-dir", str(run_dir),
            "--split-protocol", PROTOCOL,
            "--model", "xception",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
            "--experiment-family", PROTOCOL,
            "--condition-name", condition_name,
            "--canonical-size", str(preprocessing["canonical_size"]),
            "--jpeg-quality", str(preprocessing["validation_jpeg_quality"]),
            "--train-jpeg-qualities",
            *[str(value) for value in preprocessing["train_jpeg_qualities"]],
            "--jpeg-subsampling", str(preprocessing["jpeg_subsampling"]),
            "--persistent-progress",
        ]
        if preprocessing["jpeg_optimize"]:
            command.append("--jpeg-optimize")
        if preprocessing["jpeg_progressive"]:
            command.append("--jpeg-progressive")
        if last_path.is_file():
            command.extend(["--resume", str(last_path)])
            print("Resuming:", last_path, flush=True)
        else:
            print(
                f"\n=== Holdout {args.outer_method} / {row.strategy} ===",
                flush=True,
            )
            print("Selected methods:", row.selected_methods, flush=True)
        print("Run directory:", run_dir, flush=True)
        subprocess.run(command, check=True)
        if not best_path.is_file():
            raise FileNotFoundError(f"Training ended without best.pt: {run_dir}")
        completed_keys.add(row.run_key)

    print("Completed outer holdout:", args.outer_method, flush=True)


if __name__ == "__main__":
    main()
