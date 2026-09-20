#!/usr/bin/env python3
"""Train the transfer-weighted six-method Mixed-JPEG Xception pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


PROTOCOL = "transfer_weighted_mixed_jpeg_pilot_v1"
RUN_NAME = (
    "xception_m7_transfer_weighted_"
    "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_seed42"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    return parser.parse_args()


def load_and_validate(
    config_path: Path, manifest_path: Path, summary_path: Path
) -> tuple[dict, pd.DataFrame, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol: {config_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected manifest summary: {summary_path}")
    if summary.get("protocol_config_sha256") != sha256_file(config_path):
        raise RuntimeError("Manifest was generated from another protocol config")
    if summary.get("protected_unseen_used") or summary.get("wilddeepfake_used"):
        raise RuntimeError("Protected evaluation data influenced the manifest")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != summary.get("manifest_sha256"):
        raise ValueError("Manifest hash does not match manifest_summary.json")

    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"sample_id", "split", "label", "method", "source_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in weighted manifest")
    expected_methods = set(config["methods"])
    actual_methods = set(frame.loc[frame["label"] == "fake", "method"])
    if actual_methods != expected_methods:
        raise RuntimeError(
            f"Fake methods differ: expected={expected_methods}, actual={actual_methods}"
        )
    for split in ("train", "val"):
        expected_budget = config["fixed_budgets"][split]
        counts = frame[frame["split"] == split].groupby("label").size()
        actual_budget = {
            label: int(counts.get(label, 0)) for label in ("real", "fake")
        }
        if actual_budget != expected_budget:
            raise RuntimeError(
                f"Budget mismatch for {split}: {actual_budget}"
            )
        actual_quotas = (
            frame[
                (frame["split"] == split) & (frame["label"] == "fake")
            ]
            .groupby("method")
            .size()
            .astype(int)
            .to_dict()
        )
        if actual_quotas != summary["method_quotas"][split]:
            raise RuntimeError(
                f"Method quota mismatch for {split}: {actual_quotas}"
            )
    return config, frame, manifest_hash


def completed_run(
    last_path: Path,
    best_path: Path,
    *,
    manifest_hash: str,
    epochs: int,
    patience: int,
) -> bool:
    if not last_path.is_file() or not best_path.is_file():
        return False
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config", {})
    if saved.get("experiment_family") != PROTOCOL:
        raise ValueError(f"Unexpected existing run: {last_path}")
    if saved.get("condition_name") != "m7_transfer_weighted_jpeg_mixed":
        raise ValueError(f"Condition mismatch: {last_path}")
    if saved.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Manifest mismatch: {last_path}")
    return (
        int(checkpoint["epoch"]) >= epochs
        or int(checkpoint.get("epochs_without_improvement", 0)) >= patience
    )


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "manifest": args.manifest.resolve(),
        "manifest_summary": args.manifest_summary.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    config, manifest, manifest_hash = load_and_validate(
        paths["protocol_config"], paths["manifest"], paths["manifest_summary"]
    )
    missing_images = [
        value
        for value in manifest["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Manifest images are not fully extracted. First missing: "
            + ", ".join(missing_images[:5])
        )

    output_dir = paths["output_root"] / RUN_NAME
    last_checkpoint = output_dir / "last.pt"
    best_checkpoint = output_dir / "best.pt"
    if completed_run(
        last_checkpoint,
        best_checkpoint,
        manifest_hash=manifest_hash,
        epochs=args.epochs,
        patience=args.patience,
    ):
        print("Skipping completed training:", best_checkpoint, flush=True)
        return

    preprocessing = config["preprocessing"]
    trainer = Path(__file__).resolve().with_name("train_xception_letterbox.py")
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--data-root",
        str(paths["data_root"]),
        "--manifest",
        str(paths["manifest"]),
        "--output-dir",
        str(output_dir),
        "--split-protocol",
        PROTOCOL,
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
        PROTOCOL,
        "--condition-name",
        "m7_transfer_weighted_jpeg_mixed",
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
        print("Resuming:", last_checkpoint, flush=True)
    else:
        print(
            "Starting transfer-weighted M7 from ImageNet-pretrained Xception",
            flush=True,
        )
    print("Run directory:", output_dir, flush=True)
    subprocess.run(command, check=True)
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training ended without best.pt: {best_checkpoint}")


if __name__ == "__main__":
    main()
