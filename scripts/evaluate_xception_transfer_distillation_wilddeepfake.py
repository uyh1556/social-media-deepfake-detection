#!/usr/bin/env python3
"""Evaluate frozen M7 Mixed/Uniform-KD/Transfer-KD on WildDeepfake."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from evaluate_xception_family_coverage import load_conditions, read_manifest
from evaluate_xception_transfer_distillation import (
    validate_baseline,
    validate_student,
)
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
from train_xception_jpeg_mixed import run_name as mixed_run_name
from train_xception_transfer_distillation import (
    EXPERIMENT_FAMILY,
    STRATEGIES,
    load_protocol as load_pilot_protocol,
    run_name as kd_run_name,
)


PROTOCOL = "transfer_aware_distillation_external_evaluation_v1"
WILD_PROTOCOL = "transfer_distillation_wilddeepfake_letterbox299_v1"
CONDITION_ID = "M7"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--baseline-runs-root", type=Path, required=True)
    parser.add_argument("--student-runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "--evaluation-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path, expected_protocol: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if expected_protocol is not None and value.get("protocol") != expected_protocol:
        raise ValueError(f"Unexpected protocol in {path}")
    return value


def build_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Transfer-aware KD on WildDeepfake",
        "",
        (
            "> Frozen seed-42 checkpoints use the identical native RGB -> "
            "Letterbox 299 test path. Primary metrics are sequence-level over "
            "806 source sequences. WildDeepfake was not used for training, "
            "checkpoint selection, threshold selection, or teacher weighting."
        ),
        "",
        (
            "| Model | Sequence AUC | Sequence AUPRC | Real FPR | Fake recall | "
            "Frame AUC |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("model").itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.sequence_roc_auc:.4f} | "
            f"{row.sequence_auprc:.4f} | {row.sequence_real_fpr:.4f} | "
            f"{row.sequence_fake_recall:.4f} | {row.frame_roc_auc:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transfer KD deltas",
            "",
            (
                "| Baseline | Sequence AUC delta | AUPRC delta | Real FPR "
                "delta | Fake-recall delta |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.baseline_model} | {row.delta_sequence_roc_auc:+.4f} | "
            f"{row.delta_sequence_auprc:+.4f} | "
            f"{row.delta_sequence_real_fpr:+.4f} | "
            f"{row.delta_sequence_fake_recall:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Do not tune the KD rule or threshold using this result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "manifest": args.manifest.resolve(),
        "training_manifest": args.training_manifest.resolve(),
        "teacher_cache": args.teacher_cache.resolve(),
        "baseline_runs_root": args.baseline_runs_root.resolve(),
        "student_runs_root": args.student_runs_root.resolve(),
        "output_dir": args.output_dir.resolve(),
        "wild_config": args.wild_config.resolve(),
        "pilot_config": args.pilot_config.resolve(),
        "evaluation_config": args.evaluation_config.resolve(),
        "conditions_config": args.conditions_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)

    wild_config = load_json(paths["wild_config"])
    validate_protocol_config(wild_config)
    pilot = load_pilot_protocol(paths["pilot_config"])
    evaluation = read_json(paths["evaluation_config"], PROTOCOL)
    if evaluation["parent_training_protocol"] != EXPERIMENT_FAMILY:
        raise ValueError("External/parent protocol mismatch")
    if evaluation["interpretation"]["future_tuning_from_these_results"]:
        raise ValueError("WildDeepfake results must not be used for tuning")

    manifest_hash = sha256_file(paths["training_manifest"])
    teacher_hash = sha256_file(paths["teacher_cache"])
    if manifest_hash != pilot["manifest_sha256"]:
        raise ValueError("M7 training manifest hash mismatch")
    teacher_summary_path = paths["teacher_cache"].with_name(
        "cache_summary.json"
    )
    teacher_summary = read_json(teacher_summary_path, EXPERIMENT_FAMILY)
    if teacher_summary.get("teacher_targets_sha256") != teacher_hash:
        raise ValueError("Teacher cache hash mismatch")
    if teacher_summary.get("student_manifest_sha256") != manifest_hash:
        raise ValueError("Teacher cache/student manifest mismatch")

    conditions = load_conditions(paths["conditions_config"])
    m7_condition = conditions[CONDITION_ID]
    test_frame = validate_manifest(
        paths["manifest"], paths["data_root"], wild_config
    )
    training_frame = read_manifest(paths["training_manifest"])
    overlap_audit = training_overlap_audit(test_frame, training_frame)

    checkpoints: dict[str, tuple[Path, dict]] = {}
    baseline_path = (
        paths["baseline_runs_root"]
        / mixed_run_name(CONDITION_ID, m7_condition, 42)
        / "best.pt"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline_checkpoint = torch.load(
        baseline_path, map_location="cpu", weights_only=False
    )
    checkpoints["m7_mixed"] = (
        baseline_path,
        validate_baseline(
            baseline_checkpoint,
            baseline_path,
            manifest_hash,
            pilot["preprocessing"],
        ),
    )
    del baseline_checkpoint
    for strategy in STRATEGIES:
        checkpoint_path = (
            paths["student_runs_root"] / kd_run_name(strategy) / "best.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoints[strategy] = (
            checkpoint_path,
            validate_student(
                checkpoint,
                checkpoint_path,
                strategy,
                manifest_hash,
                teacher_hash,
                pilot,
            ),
        )
        del checkpoint

    reference_data_config = checkpoints["m7_mixed"][1]["data_config"]
    for model_name, (_, info) in checkpoints.items():
        if info["data_config"] != reference_data_config:
            raise RuntimeError(f"Model input mismatch: {model_name}")

    identity = {
        "protocol": WILD_PROTOCOL,
        "evaluation_config_sha256": sha256_file(paths["evaluation_config"]),
        "wild_config_sha256": sha256_file(paths["wild_config"]),
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "training_manifest_sha256": manifest_hash,
        "test_manifest_sha256": sha256_file(paths["manifest"]),
        "teacher_cache_sha256": teacher_hash,
        "checkpoints": {
            name: info["sha256"] for name, (_, info) in checkpoints.items()
        },
        "evaluation_preprocessing": "native_letterbox299",
        "interpretation_status": evaluation["interpretation"]["status"],
    }
    output_dir = paths["output_dir"]
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Output has no identity: {output_dir}")
        if load_json(identity_path) != identity:
            raise RuntimeError("Existing WildDeepfake evaluation differs")
        if (output_dir / "evaluation_summary.json").is_file():
            print("Evaluation already complete:", output_dir, flush=True)
            return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required")
    device = torch.device("cuda")
    loader = build_loaders(
        test_frame,
        paths["data_root"],
        reference_data_config,
        args.batch_size,
        args.workers,
    )["native_letterbox299"]

    rows = []
    entry = {"preprocessing": "native_letterbox299"}
    for model_name, (checkpoint_path, info) in checkpoints.items():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        frame_prediction, sequence_prediction = evaluate_one(
            WILD_PROTOCOL,
            model_name,
            42,
            entry,
            checkpoint,
            info,
            loader,
            test_frame,
            paths["manifest"],
            output_dir,
            device,
        )
        row = summary_row(
            WILD_PROTOCOL,
            model_name,
            m7_condition,
            42,
            "native_letterbox299",
            frame_prediction,
            sequence_prediction,
        )
        rows.append(row)
        del checkpoint
        torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    aggregate = aggregate_seeds(summary)
    by_model = summary.set_index("model")
    metric_names = [
        "sequence_roc_auc",
        "sequence_auprc",
        "sequence_real_fpr",
        "sequence_fake_recall",
        "frame_roc_auc",
        "frame_auprc",
    ]
    comparison_rows = []
    for baseline_name in ("m7_mixed", "uniform_kd"):
        row = {"baseline_model": baseline_name}
        for metric in metric_names:
            baseline_value = float(by_model.loc[baseline_name, metric])
            transfer_value = float(by_model.loc["transfer_kd", metric])
            row[f"baseline_{metric}"] = baseline_value
            row[f"transfer_{metric}"] = transfer_value
            row[f"delta_{metric}"] = transfer_value - baseline_value
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    atomic_csv(summary, output_dir / "model_summary.csv")
    atomic_csv(aggregate, output_dir / "seed_aggregate_summary.csv")
    atomic_csv(comparison, output_dir / "comparisons.csv")
    (output_dir / "REPORT.md").write_text(
        build_report(summary, comparison), encoding="utf-8"
    )
    write_json(
        output_dir / "evaluation_summary.json",
        {
            **identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "WildDeepfake",
            "test_images": int(len(test_frame)),
            "test_sequences": int(test_frame["sequence_uid"].nunique()),
            "frames_per_sequence": 16,
            "primary_unit": "sequence",
            "sequence_score": "mean fake probability over 16 fixed frames",
            "decision_threshold": 0.5,
            "threshold_tuned_on_wilddeepfake": False,
            "training_overlap_audit": overlap_audit,
            "source_images_modified": False,
            "test_used_for_training_or_tuning": False,
            "versions": versions(),
            "outputs": {
                "model_summary": "model_summary.csv",
                "comparisons": "comparisons.csv",
                "report": "REPORT.md",
            },
        },
    )
    print("\n=== WildDeepfake summary ===", flush=True)
    print(
        summary[
            [
                "model",
                "sequence_roc_auc",
                "sequence_auprc",
                "sequence_real_fpr",
                "sequence_fake_recall",
                "frame_roc_auc",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSaved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
