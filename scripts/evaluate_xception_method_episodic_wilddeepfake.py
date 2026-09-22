#!/usr/bin/env python3
"""Evaluate the frozen method-episodic M7 pilot on WildDeepfake."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from evaluate_xception_family_coverage import load_conditions, read_manifest
from evaluate_xception_method_episodic import validate_checkpoint
from evaluate_xception_wilddeepfake import (
    aggregate_seeds,
    atomic_csv,
    build_loaders,
    evaluate_one,
    load_json,
    summary_row,
    training_overlap_audit,
    validate_manifest,
    validate_protocol_config,
    versions,
)
from train_baseline import sha256_file, write_json
from train_xception_method_episodic import (
    CONDITION_ID,
    EXPERIMENT_FAMILY,
    load_protocol,
    run_name,
)


PROTOCOL_NAME = "method_episodic"
WILD_EVALUATION_PROTOCOL = (
    "wilddeepfake_m7_method_episodic_native_letterbox299_v1"
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
            "# Method-Episodic M7 Pilot on WildDeepfake",
            "",
            (
                "> Primary results are sequence-level. Each of 806 source "
                "sequences contributes the mean fake probability of its 16 "
                "fixed frames. WildDeepfake was not used for training, "
                "checkpoint selection, threshold selection, or pilot tuning."
            ),
            "",
            (
                "| Model | Seed | Sequence AUC | Sequence AUPRC | Real FPR | "
                "Fake recall | Frame AUC |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| M7 episodic | 42 | {row.sequence_roc_auc:.4f} | "
                f"{row.sequence_auprc:.4f} | "
                f"{row.sequence_real_fpr:.4f} | "
                f"{row.sequence_fake_recall:.4f} | "
                f"{row.frame_roc_auc:.4f} |"
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "Compare this frozen result with the existing M7 Mixed-JPEG "
                "seed-42 result produced by the same native RGB/Letterbox-299 "
                "test path. Do not tune the episodic objective or threshold "
                "using WildDeepfake."
            ),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "manifest": args.manifest.resolve(),
        "training_manifest": args.training_manifest.resolve(),
        "runs_root": args.runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "wild_config": args.wild_config.resolve(),
        "pilot_config": args.pilot_config.resolve(),
        "conditions_config": args.conditions_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)

    wild_protocol = load_json(paths["wild_config"])
    validate_protocol_config(wild_protocol)
    pilot_protocol = load_protocol(paths["pilot_config"])
    expected_training_hash = pilot_protocol["manifest_sha256"][CONDITION_ID]
    if sha256_file(paths["training_manifest"]) != expected_training_hash:
        raise ValueError("M7 training manifest hash mismatch.")

    conditions = load_conditions(paths["conditions_config"])
    condition = conditions[CONDITION_ID]
    test_frame = validate_manifest(
        paths["manifest"], paths["data_root"], wild_protocol
    )
    training_frame = read_manifest(paths["training_manifest"])
    overlap_audit = training_overlap_audit(test_frame, training_frame)

    checkpoint_path = paths["runs_root"] / run_name() / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_info = validate_checkpoint(
        checkpoint,
        checkpoint_path,
        paths["training_manifest"],
        pilot_protocol,
    )
    data_config = checkpoint["config"]["data_config"]

    identity = {
        "protocol": WILD_EVALUATION_PROTOCOL,
        "wild_config_sha256": sha256_file(paths["wild_config"]),
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "training_manifest_sha256": expected_training_hash,
        "test_manifest_sha256": sha256_file(paths["manifest"]),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "evaluation_preprocessing": "native_letterbox299",
        "model": CONDITION_ID,
        "training_seed": 42,
    }
    output_root = paths["output_root"]
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(
                f"Non-empty output has no identity: {output_root}"
            )
        if load_json(identity_path) != identity:
            raise RuntimeError(
                "Existing WildDeepfake output belongs to another run."
            )
        if (output_root / "evaluation_summary.json").is_file():
            print("Evaluation already complete:", output_root, flush=True)
            return
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required.")
    device = torch.device("cuda")
    loader = build_loaders(
        test_frame,
        paths["data_root"],
        data_config,
        args.batch_size,
        args.workers,
    )["native_letterbox299"]
    entry = {
        "runs_root": paths["runs_root"],
        "seeds": [42],
        "control": pilot_protocol,
        "preprocessing": "native_letterbox299",
    }
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    frame_prediction, sequence_prediction = evaluate_one(
        PROTOCOL_NAME,
        CONDITION_ID,
        42,
        entry,
        checkpoint,
        checkpoint_info,
        loader,
        test_frame,
        paths["manifest"],
        output_root,
        device,
    )
    row = summary_row(
        PROTOCOL_NAME,
        CONDITION_ID,
        condition,
        42,
        "native_letterbox299",
        frame_prediction,
        sequence_prediction,
    )
    summary = pd.DataFrame([row])
    aggregate = aggregate_seeds(summary)
    atomic_csv(summary, output_root / "model_seed_summary.csv")
    atomic_csv(aggregate, output_root / "seed_aggregate_summary.csv")
    (output_root / "REPORT.md").write_text(
        build_report(summary), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "WildDeepfake",
            "protocol": identity["protocol"],
            "training_protocol": EXPERIMENT_FAMILY,
            "model": CONDITION_ID,
            "training_seed": 42,
            "test_images": int(len(test_frame)),
            "test_sequences": int(test_frame["sequence_uid"].nunique()),
            "frames_per_sequence": 16,
            "primary_unit": "sequence",
            "sequence_score": (
                "mean fake probability over 16 selected frames"
            ),
            "decision_threshold": 0.5,
            "threshold_tuned_on_wilddeepfake": False,
            "shared_test_preprocessing": (
                "RGB decode -> native-image Letterbox 299 -> Tensor -> "
                "Normalize; no JPEG re-encoding"
            ),
            "training_overlap_audit": overlap_audit,
            "source_images_modified": False,
            "wilddeepfake_used_for_pilot_tuning": False,
            "versions": versions(),
        },
    )
    print("\n=== WildDeepfake method-episodic M7 ===", flush=True)
    print(
        summary[
            [
                "sequence_roc_auc",
                "sequence_auprc",
                "sequence_real_fpr",
                "sequence_fake_recall",
                "frame_roc_auc",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved to:", output_root, flush=True)


if __name__ == "__main__":
    main()
