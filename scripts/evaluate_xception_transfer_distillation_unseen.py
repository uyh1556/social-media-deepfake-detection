#!/usr/bin/env python3
"""Evaluate frozen M7 Mixed/Uniform-KD/Transfer-KD on all DF40 test methods."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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
from evaluate_xception_reencoding_control import (
    PROTECTED_UNSEEN,
    prediction_frame,
    summarize_prediction,
    versions,
)
from evaluate_xception_transfer_distillation import (
    validate_baseline,
    validate_student,
)
from train_baseline import sha256_file, write_json
from train_xception_jpeg_mixed import run_name as mixed_run_name
from train_xception_transfer_distillation import (
    EXPERIMENT_FAMILY,
    STRATEGIES,
    load_protocol as load_pilot_protocol,
    run_name as kd_run_name,
)
from xception_preprocessing import (
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
)


PROTOCOL = "transfer_aware_distillation_external_evaluation_v1"
CONDITION_ID = "M7"
EXPECTED_TEST_IMAGES = 30_000


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
        default=(
            root
            / f"configs/{EXPERIMENT_FAMILY}/protocol.json"
        ),
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path, expected_protocol: str | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if expected_protocol is not None and value.get("protocol") != expected_protocol:
        raise ValueError(f"Unexpected protocol in {path}")
    return value


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_report(
    model_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    lines = [
        "# Transfer-aware KD: DF40 External Evaluation",
        "",
        (
            "> Frozen seed-42 checkpoints evaluated on all 14 fake methods. "
            "The protected unseen methods were not used for training or KD "
            "weight construction. This is an exploratory post-hoc extension "
            "because the internal advance rule was not satisfied."
        ),
        "",
        (
            "| Condition | Model | Real FPR | All-method AUC | Protected "
            "unseen AUC | Unseen without PixArt | PixArt AUC |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.sort_values(
        ["evaluation_condition", "model"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.model} | "
            f"{row.real_fpr:.4f} | {row.all_methods_macro_auc:.4f} | "
            f"{row.protected_unseen_macro_auc:.4f} | "
            f"{row.protected_unseen_excluding_pixart_macro_auc:.4f} | "
            f"{row.pixart_auc:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transfer KD deltas",
            "",
            (
                "| Condition | Baseline | Unseen AUC delta | Delta without "
                "PixArt | All-method delta | Real FPR delta |"
            ),
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.baseline_model} | "
            f"{row.delta_protected_unseen_macro_auc:+.4f} | "
            f"{row.delta_protected_unseen_excluding_pixart_macro_auc:+.4f} | "
            f"{row.delta_all_methods_macro_auc:+.4f} | "
            f"{row.delta_real_fpr:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Do not modify the KD weight or teacher rule in response to this result.",
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
        "evaluation_config": args.evaluation_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)

    pilot = load_pilot_protocol(paths["pilot_config"])
    evaluation = load_json(paths["evaluation_config"], PROTOCOL)
    if evaluation["parent_training_protocol"] != EXPERIMENT_FAMILY:
        raise ValueError("External/parent protocol mismatch")
    if set(evaluation["protected_unseen_methods"]) != PROTECTED_UNSEEN:
        raise ValueError("Protected unseen method definition mismatch")
    if evaluation["df40_evaluation_conditions"] != pilot[
        "evaluation_conditions"
    ]:
        raise ValueError("Pilot/external JPEG conditions differ")
    if evaluation["interpretation"]["future_tuning_from_these_results"]:
        raise ValueError("External results must not be used for tuning")

    manifest_hash = sha256_file(paths["training_manifest"])
    test_hash = sha256_file(paths["test_manifest"])
    teacher_hash = sha256_file(paths["teacher_cache"])
    if manifest_hash != pilot["manifest_sha256"]:
        raise ValueError("M7 training manifest hash mismatch")
    if test_hash != evaluation["df40_test_manifest_sha256"]:
        raise ValueError("Balanced test manifest hash mismatch")
    teacher_summary_path = paths["teacher_cache"].with_name(
        "cache_summary.json"
    )
    teacher_summary = load_json(teacher_summary_path, EXPERIMENT_FAMILY)
    if teacher_summary.get("teacher_targets_sha256") != teacher_hash:
        raise ValueError("Teacher cache hash mismatch")
    if teacher_summary.get("student_manifest_sha256") != manifest_hash:
        raise ValueError("Teacher cache/student manifest mismatch")

    conditions = load_conditions(paths["conditions_config"])
    m7_condition = conditions[CONDITION_ID]
    development = read_manifest(paths["training_manifest"])
    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise ValueError("Unexpected balanced test image count")
    audit = cross_split_audit(development, test_frame)
    test_frame["source_reference_path"] = test_frame["source_path"]
    test_frame["resolved_path"] = test_frame["source_path"]
    missing = [
        value
        for value in test_frame["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Test archive incomplete: " + ", ".join(missing[:5])
        )

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
        "protocol": PROTOCOL,
        "evaluation_config_sha256": sha256_file(paths["evaluation_config"]),
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "training_manifest_sha256": manifest_hash,
        "test_manifest_sha256": test_hash,
        "teacher_cache_sha256": teacher_hash,
        "checkpoints": {
            name: info["sha256"] for name, (_, info) in checkpoints.items()
        },
        "evaluation_conditions": [
            item["name"] for item in evaluation["df40_evaluation_conditions"]
        ],
        "interpretation_status": evaluation["interpretation"]["status"],
    }
    output_dir = paths["output_dir"]
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Output has no identity: {output_dir}")
        if load_json(identity_path) != identity:
            raise RuntimeError("Existing evaluation belongs to another run")
        if (output_dir / "evaluation_summary.json").is_file():
            print("Evaluation already complete:", output_dir, flush=True)
            return
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    loaders = {}
    preprocessing = pilot["preprocessing"]
    for condition in evaluation["df40_evaluation_conditions"]:
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

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required")
    device = torch.device("cuda")
    model_rows = []
    method_rows = []
    for model_name, (checkpoint_path, info) in checkpoints.items():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
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
        for condition in evaluation["df40_evaluation_conditions"]:
            condition_name = condition["name"]
            result_dir = output_dir / model_name / condition_name
            prediction_path = result_dir / "predictions.csv"
            metrics_path = result_dir / "metrics.json"
            if prediction_path.is_file() and metrics_path.is_file():
                metrics_payload = load_json(metrics_path)
                if metrics_payload.get("checkpoint_sha256") != info["sha256"]:
                    raise RuntimeError(
                        f"Checkpoint changed for {result_dir}"
                    )
                if metrics_payload.get("test_manifest_sha256") != test_hash:
                    raise RuntimeError(f"Test manifest changed for {result_dir}")
                prediction = pd.read_csv(prediction_path)
                if prediction["sample_id"].tolist() != test_frame[
                    "sample_id"
                ].tolist():
                    raise RuntimeError(
                        f"Prediction order mismatch: {prediction_path}"
                    )
                print(
                    f"Skipping completed: {model_name}/{condition_name}",
                    flush=True,
                )
            elif result_dir.exists() and any(result_dir.iterdir()):
                raise RuntimeError(f"Incomplete evaluation: {result_dir}")
            else:
                print(f"\n=== {model_name} / {condition_name} ===", flush=True)
                metrics, labels, predictions, probabilities = (
                    evaluate_with_predictions(
                        model,
                        loaders[condition_name],
                        test_frame,
                        criterion,
                        device,
                        progress_desc=f"{model_name} {condition_name}",
                        persistent_progress=True,
                    )
                )
                prediction = prediction_frame(
                    test_frame, labels, predictions, probabilities
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                atomic_csv(prediction, prediction_path)
                write_json(
                    metrics_path,
                    {
                        "model": model_name,
                        "training_seed": 42,
                        "evaluation_condition": condition,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": info["sha256"],
                        "test_manifest_sha256": test_hash,
                        "cross_split_audit": audit,
                        "metrics": metrics,
                        "decision_rule": "argmax logits (0.5 equivalent)",
                        "test_used_for_tuning": False,
                        "versions": versions(),
                    },
                )
            model_row, current_method_rows = summarize_prediction(
                model_name,
                m7_condition,
                42,
                condition,
                prediction,
            )
            model_row["checkpoint_epoch"] = info["epoch"]
            model_row["best_validation_auc"] = info["best_validation_auc"]
            model_rows.append(model_row)
            method_rows.extend(current_method_rows)
        del model, checkpoint, criterion
        torch.cuda.empty_cache()

    model_summary = pd.DataFrame(model_rows)
    method_summary = pd.DataFrame(method_rows)
    metrics = [
        "protected_unseen_macro_auc",
        "protected_unseen_excluding_pixart_macro_auc",
        "pixart_auc",
        "all_methods_macro_auc",
        "real_fpr",
        "fake_recall",
    ]
    comparison_rows = []
    for condition_name, group in model_summary.groupby(
        "evaluation_condition", sort=True
    ):
        by_model = group.set_index("model")
        for baseline_name in ("m7_mixed", "uniform_kd"):
            row = {
                "evaluation_condition": condition_name,
                "baseline_model": baseline_name,
            }
            for metric in metrics:
                baseline_value = float(by_model.loc[baseline_name, metric])
                transfer_value = float(by_model.loc["transfer_kd", metric])
                row[f"baseline_{metric}"] = baseline_value
                row[f"transfer_{metric}"] = transfer_value
                row[f"delta_{metric}"] = transfer_value - baseline_value
            comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)

    atomic_csv(model_summary, output_dir / "model_summary.csv")
    atomic_csv(method_summary, output_dir / "method_summary.csv")
    atomic_csv(comparison, output_dir / "comparisons.csv")
    (output_dir / "REPORT.md").write_text(
        build_report(model_summary, comparison), encoding="utf-8"
    )
    write_json(
        output_dir / "evaluation_summary.json",
        {
            **identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_images": int(len(test_frame)),
            "protected_unseen_methods": evaluation[
                "protected_unseen_methods"
            ],
            "test_used_for_training_or_tuning": False,
            "outputs": {
                "model_summary": "model_summary.csv",
                "method_summary": "method_summary.csv",
                "comparisons": "comparisons.csv",
                "report": "REPORT.md",
            },
        },
    )
    print("\nSaved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
