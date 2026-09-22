#!/usr/bin/env python3
"""Evaluate the frozen family-conditioned method-adversarial pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evaluate_xception_jpeg_consistency as shared
from train_baseline import sha256_file
from train_xception_family_conditioned_method_adversarial import (
    CONDITION_NAME,
    EXPERIMENT_FAMILY,
    PREPROCESSING_NAME,
    load_protocol,
    run_name,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    training_manifest: Path,
    protocol: dict,
) -> dict:
    config = checkpoint["config"]
    expected = {
        "experiment_family": EXPERIMENT_FAMILY,
        "split_protocol": EXPERIMENT_FAMILY,
        "condition_name": CONDITION_NAME,
        "preprocessing_name": PREPROCESSING_NAME,
        "manifest_sha256": sha256_file(training_manifest),
        "seed": 42,
        "preprocessing": protocol["preprocessing"],
        "objective": protocol["objective"],
        "optimization": protocol["optimization"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: {checkpoint_path}"
            )
    if checkpoint["model_name"] not in {"xception", "legacy_xception"}:
        raise ValueError("Unexpected checkpoint model.")
    if not isinstance(
        checkpoint.get("method_discriminator_state_dict"), dict
    ):
        raise ValueError("Checkpoint has no method-discriminator state.")
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
    }


def build_report(model_summary: pd.DataFrame) -> str:
    lines = [
        "# M7 Family-Conditioned Method-Adversarial Pilot",
        "",
        (
            "> Seed-42 pilot. The image budget and Mixed-JPEG path match "
            "M7, while one gradient-reversal method head per family "
            "suppresses identity of the two seen methods in that family."
        ),
        "",
        (
            "| Test condition | Real FPR | Fake recall | All-method AUC | "
            "Development-unseen AUC | Without PixArt | PixArt AUC |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.sort_values("evaluation_condition").itertuples(
        index=False
    ):
        lines.append(
            f"| {row.evaluation_condition} | {row.real_fpr:.4f} | "
            f"{row.fake_recall:.4f} | {row.all_methods_macro_auc:.4f} | "
            f"{row.protected_unseen_macro_auc:.4f} | "
            f"{row.protected_unseen_excluding_pixart_macro_auc:.4f} | "
            f"{row.pixart_auc:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The six unseen methods and WildDeepfake have already been "
                "inspected and are development evidence, not a new final "
                "holdout. Do not tune this frozen pilot on these results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    shared.parse_args = parse_args
    shared.validate_checkpoint = validate_checkpoint
    shared.build_report = build_report
    shared.EXPERIMENT_FAMILY = EXPERIMENT_FAMILY
    shared.PREPROCESSING_NAME = PREPROCESSING_NAME
    shared.load_protocol = load_protocol
    shared.run_name = run_name
    shared.main()


if __name__ == "__main__":
    main()
