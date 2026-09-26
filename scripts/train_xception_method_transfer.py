#!/usr/bin/env python3
"""Train one frozen single-method Xception for the transfer matrix pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


PROTOCOL = "method_transfer_matrix_v1"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=project_root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not protocol.get("protocol"):
        raise ValueError(f"Missing protocol name in {path}")
    return protocol


def run_name(method: str, definition: dict, seed: int, protocol: dict) -> str:
    template = protocol.get("run_name_template")
    if template:
        return template.format(
            method=method,
            slug=definition["slug"],
            seed=seed,
        )
    return (
        f"xception_method_transfer_{definition['slug']}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
        f"seed{seed}"
    )


def condition_name(method: str, definition: dict, protocol: dict) -> str:
    template = protocol.get("condition_name_template")
    if template:
        return template.format(method=method, slug=definition["slug"])
    return f"method_transfer_{definition['slug']}_jpeg_mixed"


def validate_manifest(
    path: Path,
    method: str,
    definition: dict,
    protocol: dict,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_hash = sha256_file(path)
    expected_hash = protocol["manifest_sha256"][method]
    if manifest_hash != expected_hash:
        raise ValueError(
            f"Manifest hash mismatch for {method}: "
            f"expected={expected_hash}, actual={manifest_hash}"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"sample_id", "split", "label", "method", "family", "source_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    if set(frame.loc[frame["label"] == "real", "method"]) != {"original"}:
        raise RuntimeError("Single-method Real rows must be original")
    if set(frame.loc[frame["label"] == "fake", "method"]) != {method}:
        raise RuntimeError(f"Single-method fake rows do not match {method}")
    fake_families = set(frame.loc[frame["label"] == "fake", "family"])
    if fake_families != {definition["family"]}:
        raise RuntimeError(
            f"Family mismatch for {method}: {sorted(fake_families)}"
        )
    for split in ("train", "val"):
        counts = (
            frame[frame["split"] == split].groupby("label").size().to_dict()
        )
        expected = protocol["budgets"][split]
        if counts != expected:
            raise RuntimeError(
                f"Budget mismatch for {method}/{split}: "
                f"expected={expected}, actual={counts}"
            )
    return manifest_hash


def completed_run(
    last_checkpoint: Path,
    best_checkpoint: Path,
    *,
    method: str,
    seed: int,
    epochs: int,
    patience: int,
    manifest_hash: str,
    protocol: dict,
    definition: dict,
) -> bool:
    if not last_checkpoint.is_file() or not best_checkpoint.is_file():
        return False
    checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    expected_condition = condition_name(method, definition, protocol)
    protocol_name = protocol["protocol"]
    if config.get("experiment_family") != protocol_name:
        raise ValueError(f"Unexpected existing run: {last_checkpoint}")
    if config.get("condition_name") != expected_condition:
        raise ValueError(f"Condition mismatch in {last_checkpoint}")
    if int(config.get("seed")) != seed:
        raise ValueError(f"Seed mismatch in {last_checkpoint}")
    if config.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Manifest mismatch in {last_checkpoint}")
    return (
        int(checkpoint["epoch"]) >= epochs
        or int(checkpoint.get("epochs_without_improvement", 0)) >= patience
    )


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "manifest_dir": args.manifest_dir.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
    }
    protocol = load_protocol(paths["protocol_config"])
    if args.method not in protocol["methods"]:
        raise ValueError(
            f"Unknown method {args.method!r}; choose from "
            f"{list(protocol['methods'])}"
        )
    if args.seed not in protocol["training_seeds"]:
        raise ValueError(f"Seed is not enabled by the protocol: {args.seed}")
    definition = protocol["methods"][args.method]
    manifest = paths["manifest_dir"] / f"{definition['slug']}_seed42.csv"
    manifest_hash = validate_manifest(
        manifest, args.method, definition, protocol
    )
    if not paths["data_root"].is_dir():
        raise FileNotFoundError(paths["data_root"])
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    missing = [
        value
        for value in frame["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Training images are incomplete. First missing paths: "
            + ", ".join(missing[:5])
        )

    preprocessing = protocol["preprocessing"]
    run_dir = paths["output_root"] / run_name(
        args.method, definition, args.seed, protocol
    )
    last_checkpoint = run_dir / "last.pt"
    best_checkpoint = run_dir / "best.pt"
    if completed_run(
        last_checkpoint,
        best_checkpoint,
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        manifest_hash=manifest_hash,
        protocol=protocol,
        definition=definition,
    ):
        print("Skipping completed training:", best_checkpoint, flush=True)
        return

    trainer = Path(__file__).resolve().with_name("train_xception_letterbox.py")
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--data-root",
        str(paths["data_root"]),
        "--manifest",
        str(manifest),
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
        protocol["protocol"],
        "--condition-name",
        condition_name(args.method, definition, protocol),
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
            f"Starting {args.method} single-method model from "
            "ImageNet-pretrained Xception",
            flush=True,
        )
    print("Run directory:", run_dir, flush=True)
    subprocess.run(command, check=True)
    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"Training ended without best.pt: {run_dir}")


if __name__ == "__main__":
    main()
