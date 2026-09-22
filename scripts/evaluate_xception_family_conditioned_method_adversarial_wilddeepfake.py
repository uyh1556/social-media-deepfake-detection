#!/usr/bin/env python3
"""Evaluate the frozen method-adversarial M7 pilot on WildDeepfake."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import evaluate_xception_method_episodic_wilddeepfake as shared
from evaluate_xception_family_conditioned_method_adversarial import (
    validate_checkpoint,
)
from train_xception_family_conditioned_method_adversarial import (
    EXPERIMENT_FAMILY,
    load_protocol,
    run_name,
)


PROTOCOL_NAME = "family_conditioned_method_adversarial"
WILD_EVALUATION_PROTOCOL = (
    "wilddeepfake_m7_family_conditioned_method_adversarial_"
    "native_letterbox299_v1"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--wild-config",
        type=Path,
        default=root / "configs/wilddeepfake_evaluation_v1/protocol.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
    )
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def build_report(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    return "\n".join(
        [
            "# Method-Adversarial M7 Pilot on WildDeepfake",
            "",
            (
                "> Sequence-level result using the same 16 fixed frames "
                "and native RGB/Letterbox-299 path as the existing M7 "
                "Mixed-JPEG reference."
            ),
            "",
            (
                "| Model | Seed | Sequence AUC | Sequence AUPRC | "
                "Real FPR | Fake recall | Frame AUC |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| M7 method-adversarial | 42 | "
                f"{row.sequence_roc_auc:.4f} | "
                f"{row.sequence_auprc:.4f} | "
                f"{row.sequence_real_fpr:.4f} | "
                f"{row.sequence_fake_recall:.4f} | "
                f"{row.frame_roc_auc:.4f} |"
            ),
            "",
            (
                "WildDeepfake is development characterization only and "
                "must not be used to retune the adversarial strength or "
                "decision threshold."
            ),
            "",
        ]
    )


def main() -> None:
    shared.parse_args = parse_args
    shared.validate_checkpoint = validate_checkpoint
    shared.build_report = build_report
    shared.EXPERIMENT_FAMILY = EXPERIMENT_FAMILY
    shared.PROTOCOL_NAME = PROTOCOL_NAME
    shared.WILD_EVALUATION_PROTOCOL = WILD_EVALUATION_PROTOCOL
    shared.load_protocol = load_protocol
    shared.run_name = run_name
    shared.main()


if __name__ == "__main__":
    main()
