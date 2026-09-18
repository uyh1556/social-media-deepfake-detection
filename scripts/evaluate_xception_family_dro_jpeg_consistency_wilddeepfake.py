#!/usr/bin/env python3
"""Evaluate M7 family-DRO plus JPEG consistency on WildDeepfake."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evaluate_xception_jpeg_consistency_wilddeepfake as shared
from evaluate_xception_family_dro_jpeg_consistency import (
    validate_checkpoint,
)
from train_xception_family_dro_jpeg_consistency import (
    EXPERIMENT_FAMILY,
    load_protocol,
    run_name,
)


PROTOCOL_NAME = "family_dro_jpeg_consistency_pilot"
WILD_EVALUATION_PROTOCOL = (
    "wilddeepfake_m7_family_dro_jpeg_consistency_native_letterbox299_v1"
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--wild-config",
        type=Path,
        default=project_root / "configs/wilddeepfake_evaluation_v1/protocol.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=project_root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
    )
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def build_report(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    return "\n".join(
        [
            "# M7 Family DRO + JPEG Consistency on WildDeepfake",
            "",
            (
                "> Primary results are sequence-level. WildDeepfake was "
                "not used for training, checkpoint selection, threshold "
                "selection, or method tuning."
            ),
            "",
            (
                "| Model | Seed | Sequence AUC | Sequence AUPRC | "
                "Real FPR | Fake recall | Frame AUC |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| M7 family DRO + consistency | 42 | "
                f"{row.sequence_roc_auc:.4f} | {row.sequence_auprc:.4f} | "
                f"{row.sequence_real_fpr:.4f} | "
                f"{row.sequence_fake_recall:.4f} | {row.frame_roc_auc:.4f} |"
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "Compare with the seed-42 M7 Mixed-JPEG and consistency-only "
                "runs using the same native RGB/Letterbox-299 path. Do not "
                "retune either loss weight on WildDeepfake."
            ),
            "",
        ]
    )


def main() -> None:
    shared.parse_args = parse_args
    shared.validate_pilot_checkpoint = validate_checkpoint
    shared.build_report = build_report
    shared.EXPERIMENT_FAMILY = EXPERIMENT_FAMILY
    shared.PROTOCOL_NAME = PROTOCOL_NAME
    shared.WILD_EVALUATION_PROTOCOL = WILD_EVALUATION_PROTOCOL
    shared.load_protocol = load_protocol
    shared.run_name = run_name
    shared.main()


if __name__ == "__main__":
    main()
