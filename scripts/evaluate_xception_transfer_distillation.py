#!/usr/bin/env python3
"""Compare M7 Mixed-JPEG, uniform KD, and transfer-aware KD internally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
from evaluate_xception_family_coverage import (
    cross_split_audit,
    load_conditions,
    read_manifest,
    validate_test_manifest,
)
from evaluate_xception_method_transfer import prediction_frame, summarize_pairs
from train_baseline import sha256_file, write_json
from train_xception_jpeg_mixed import run_name as mixed_run_name
from train_xception_transfer_distillation import (
    EXPERIMENT_FAMILY,
    STRATEGIES,
    load_protocol,
    run_name as kd_run_name,
)
from xception_preprocessing import (
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
)


CONDITION_ID = "M7"
BASELINE_FAMILY = "family_coverage_jpeg_mixed_v1"
EXPECTED_PER_METHOD = 2_000


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--baseline-runs-root", type=Path, required=True)
    parser.add_argument("--student-runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def validate_baseline(
    checkpoint: dict,
    path: Path,
    manifest_hash: str,
    preprocessing: dict,
) -> dict:
    config = checkpoint["config"]
    expected = {
        "experiment_family": BASELINE_FAMILY,
        "condition_name": "m7_fs_fr_efs_jpeg_mixed",
        "manifest_sha256": manifest_hash,
        "seed": 42,
        "preprocessing_name": MIXED_JPEG_REENCODE_NAME,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Baseline mismatch for {key}: {path}")
    for key in [
        "canonical_size",
        "train_jpeg_qualities",
        "train_jpeg_sampling",
        "validation_jpeg_quality",
        "jpeg_subsampling",
        "jpeg_optimize",
        "jpeg_progressive",
    ]:
        if config["preprocessing"].get(key) != preprocessing.get(key):
            raise ValueError(f"Baseline preprocessing mismatch: {key}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
        "data_config": config["data_config"],
        "class_weights": config["class_weights"],
    }


def validate_student(
    checkpoint: dict,
    path: Path,
    strategy: str,
    manifest_hash: str,
    teacher_cache_hash: str,
    protocol: dict,
) -> dict:
    config = checkpoint["config"]
    expected = {
        "experiment_family": EXPERIMENT_FAMILY,
        "split_protocol": EXPERIMENT_FAMILY,
        "condition_name": f"m7_fs_fr_efs_{strategy}",
        "distillation_strategy": strategy,
        "manifest_sha256": manifest_hash,
        "teacher_cache_sha256": teacher_cache_hash,
        "seed": 42,
        "preprocessing": protocol["preprocessing"],
        "objective": protocol["objective"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Student mismatch for {key}: {path}")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
        "data_config": config["data_config"],
        "class_weights": config["class_weights"],
    }


def report(model_summary: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "# M7 Transfer-aware Distillation Internal Pilot",
        "",
        (
            "> This is a training-side seen-method diagnostic. The transfer "
            "weights were derived from these six methods, so this is not final "
            "unseen evidence. Protected unseen methods and WildDeepfake were "
            "not used."
        ),
        "",
        "## Model summary",
        "",
        "| Condition | Model | Mean seen AUC | Worst seen AUC | Mean AUPRC | Mean Real FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.model} | "
            f"{row.mean_auc:.4f} | {row.worst_auc:.4f} | "
            f"{row.mean_auprc:.4f} | {row.mean_real_fpr:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transfer KD differences",
            "",
            "| Condition | Baseline | Mean AUC delta | Worst AUC delta | Wins / 6 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.baseline_model} | "
            f"{row.mean_auc_delta:+.4f} | {row.worst_auc_delta:+.4f} | "
            f"{row.transfer_kd_wins} / {row.methods} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The essential comparison is transfer_kd versus uniform_kd. "
                "An improvement over m7_mixed without improvement over "
                "uniform_kd would support generic distillation, not the transfer matrix."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "training_manifest": args.training_manifest.resolve(),
        "teacher_cache": args.teacher_cache.resolve(),
        "baseline_runs_root": args.baseline_runs_root.resolve(),
        "student_runs_root": args.student_runs_root.resolve(),
        "output_dir": args.output_dir.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "pilot_config": args.pilot_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)
    protocol = load_protocol(paths["pilot_config"])
    manifest_hash = sha256_file(paths["training_manifest"])
    test_hash = sha256_file(paths["test_manifest"])
    teacher_hash = sha256_file(paths["teacher_cache"])
    teacher_summary_path = paths["teacher_cache"].with_name(
        "cache_summary.json"
    )
    if not teacher_summary_path.is_file():
        raise FileNotFoundError(teacher_summary_path)
    teacher_summary = json.loads(
        teacher_summary_path.read_text(encoding="utf-8")
    )
    if teacher_summary.get("protocol") != EXPERIMENT_FAMILY:
        raise ValueError("Teacher cache protocol mismatch")
    if teacher_summary.get("teacher_targets_sha256") != teacher_hash:
        raise ValueError("Teacher cache file hash mismatch")
    if teacher_summary.get("student_manifest_sha256") != manifest_hash:
        raise ValueError("Teacher cache/student manifest mismatch")
    if teacher_summary.get("protected_unseen_used") or teacher_summary.get(
        "wilddeepfake_used"
    ):
        raise ValueError("Final evaluation data leaked into teacher cache")
    if manifest_hash != protocol["manifest_sha256"]:
        raise ValueError("M7 manifest hash mismatch")
    if test_hash != protocol["test_manifest_sha256"]:
        raise ValueError("Test manifest hash mismatch")

    conditions = load_conditions(paths["conditions_config"])
    m7_condition = conditions[CONDITION_ID]
    development = read_manifest(paths["training_manifest"])
    full_test = read_manifest(paths["test_manifest"])
    validate_test_manifest(full_test)
    cross_split_audit(development, full_test)
    methods = protocol["internal_evaluation"]["methods"]
    test_frame = full_test[
        (full_test["method"] == "original")
        | (full_test["method"].isin(methods))
    ].copy().reset_index(drop=True)
    counts = test_frame.groupby("method").size().to_dict()
    if set(counts) != {"original", *methods} or set(counts.values()) != {
        EXPECTED_PER_METHOD
    }:
        raise RuntimeError(f"Unexpected seen test counts: {counts}")
    test_frame["resolved_path"] = test_frame["source_path"]
    test_frame["source_reference_path"] = test_frame["source_path"]
    missing = [
        value
        for value in test_frame["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing test images: " + ", ".join(missing[:5]))

    checkpoints = {}
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
        baseline_checkpoint,
        validate_baseline(
            baseline_checkpoint,
            baseline_path,
            manifest_hash,
            protocol["preprocessing"],
        ),
    )
    for strategy in STRATEGIES:
        path = paths["student_runs_root"] / kd_run_name(strategy) / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[strategy] = (
            checkpoint,
            validate_student(
                checkpoint,
                path,
                strategy,
                manifest_hash,
                teacher_hash,
                protocol,
            ),
        )
    reference_data_config = checkpoints["m7_mixed"][1]["data_config"]
    for model_name, (_, info) in checkpoints.items():
        if info["data_config"] != reference_data_config:
            raise RuntimeError(f"Model input mismatch: {model_name}")

    identity = {
        "protocol": EXPERIMENT_FAMILY,
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "training_manifest_sha256": manifest_hash,
        "test_manifest_sha256": test_hash,
        "teacher_cache_sha256": teacher_hash,
        "teacher_cache_summary_sha256": sha256_file(teacher_summary_path),
        "checkpoints": {
            name: info["sha256"] for name, (_, info) in checkpoints.items()
        },
        "evaluation_methods": methods,
        "evaluation_conditions": [
            item["name"] for item in protocol["evaluation_conditions"]
        ],
        "protected_unseen_used": False,
        "wilddeepfake_used": False,
    }
    output_dir = paths["output_dir"]
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Output has no identity: {output_dir}")
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing evaluation belongs to another run")
        if (output_dir / "evaluation_summary.json").is_file():
            print("Evaluation already complete:", output_dir, flush=True)
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not identity_path.is_file():
        write_json(identity_path, identity)

    loaders = {}
    preprocessing = protocol["preprocessing"]
    for condition in protocol["evaluation_conditions"]:
        transform = canonical_reencode_transform(
            reference_data_config,
            canonical_size=preprocessing["canonical_size"],
            jpeg_quality=condition["jpeg_quality"],
            jpeg_subsampling=preprocessing["jpeg_subsampling"],
            jpeg_optimize=preprocessing["jpeg_optimize"],
            jpeg_progressive=preprocessing["jpeg_progressive"],
        )
        loaders[condition["name"]] = DataLoader(
            EvaluationDataset(
                test_frame,
                paths["data_root"],
                transform,
                MIXED_JPEG_REENCODE_NAME,
                reference_data_config["input_size"][1],
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required")
    rows = []
    for model_name, (checkpoint, info) in checkpoints.items():
        checkpoint = torch.load(
            info["path"], map_location=device, weights_only=False
        )
        model = timm.create_model(
            checkpoint["model_name"],
            pretrained=False,
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        saved_weights = info["class_weights"]
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [saved_weights["real"], saved_weights["fake"]],
                dtype=torch.float32,
                device=device,
            )
        )
        for condition in protocol["evaluation_conditions"]:
            name = condition["name"]
            result_dir = output_dir / model_name / name
            result_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = result_dir / "predictions.csv"
            if prediction_path.is_file():
                prediction = pd.read_csv(prediction_path)
                if prediction["sample_id"].tolist() != test_frame[
                    "sample_id"
                ].tolist():
                    raise RuntimeError(
                        f"Prediction order mismatch: {prediction_path}"
                    )
                print(f"Skipping completed: {model_name}/{name}", flush=True)
            else:
                print(f"\n=== {model_name} / {name} ===", flush=True)
                _, labels, predictions, probabilities = (
                    evaluate_with_predictions(
                        model,
                        loaders[name],
                        test_frame,
                        criterion,
                        device,
                        progress_desc=f"{model_name} {name}",
                        persistent_progress=True,
                    )
                )
                prediction = prediction_frame(
                    test_frame, labels, predictions, probabilities
                )
                temporary = prediction_path.with_suffix(".csv.tmp")
                prediction.to_csv(temporary, index=False)
                temporary.replace(prediction_path)
            pair_rows = summarize_pairs(
                prediction,
                training_method=model_name,
                training_family="FS+FR+EFS",
                evaluation_condition=name,
                protocol={"evaluation_methods": methods, "methods": protocol["methods"]},
            )
            for row in pair_rows:
                row["model"] = model_name
            rows.extend(pair_rows)
        del model, checkpoint
        torch.cuda.empty_cache()

    method_summary = pd.DataFrame(rows).sort_values(
        ["evaluation_condition", "model", "target_method"]
    )
    model_summary = (
        method_summary.groupby(["evaluation_condition", "model"], as_index=False)
        .agg(
            mean_auc=("roc_auc", "mean"),
            worst_auc=("roc_auc", "min"),
            mean_auprc=("auprc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            mean_real_fpr=("real_fpr", "mean"),
        )
        .sort_values(["evaluation_condition", "model"])
    )
    comparison_rows = []
    for condition in identity["evaluation_conditions"]:
        subset = method_summary[
            method_summary["evaluation_condition"] == condition
        ]
        pivot = subset.pivot(
            index="target_method", columns="model", values="roc_auc"
        )
        summary_by_model = model_summary[
            model_summary["evaluation_condition"] == condition
        ].set_index("model")
        for baseline in ["m7_mixed", "uniform_kd"]:
            delta = pivot["transfer_kd"] - pivot[baseline]
            comparison_rows.append(
                {
                    "evaluation_condition": condition,
                    "baseline_model": baseline,
                    "mean_auc_delta": float(delta.mean()),
                    "worst_auc_delta": float(
                        summary_by_model.loc["transfer_kd", "worst_auc"]
                        - summary_by_model.loc[baseline, "worst_auc"]
                    ),
                    "transfer_kd_wins": int((delta > 0).sum()),
                    "methods": int(len(delta)),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    (output_dir / "REPORT.md").write_text(
        report(model_summary, comparisons), encoding="utf-8"
    )
    write_json(
        output_dir / "evaluation_summary.json",
        {
            **identity,
            "purpose": protocol["internal_evaluation"]["purpose"],
            "outputs": {
                "method_summary": "method_summary.csv",
                "model_summary": "model_summary.csv",
                "comparisons": "comparisons.csv",
                "report": "REPORT.md",
            },
        },
    )
    print("Saved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
