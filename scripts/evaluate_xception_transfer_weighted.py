#!/usr/bin/env python3
"""Compare transfer-weighted M7 with the uniform Mixed-JPEG M7 baseline."""

from __future__ import annotations

import argparse
import json
import shutil
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
    build_transform,
    prediction_frame,
    summarize_prediction,
    validate_checkpoint as validate_baseline_checkpoint,
    versions,
)
from train_baseline import sha256_file, write_json
from xception_preprocessing import MIXED_JPEG_REENCODE_NAME


PROTOCOL = "transfer_weighted_mixed_jpeg_pilot_v1"
WEIGHTED_RUN_NAME = (
    "xception_m7_transfer_weighted_"
    "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_seed42"
)
EXPECTED_TEST_IMAGES = 30_000


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--weighted-manifest", type=Path, required=True)
    parser.add_argument("--weighted-manifest-summary", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--weighted-runs-root", type=Path, required=True)
    parser.add_argument("--baseline-runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-evaluation-root",
        type=Path,
        default=None,
        help="Optional completed M7 Mixed-JPEG evaluation to reuse.",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=(
            root
            / "configs/family_coverage_jpeg_mixed_v1/"
            "m0_m7_3seed_protocol.json"
        ),
    )
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_weighted_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    manifest_path: Path,
    config: dict,
) -> dict:
    saved = checkpoint["config"]
    if saved.get("experiment_family") != PROTOCOL:
        raise ValueError(f"Unexpected weighted checkpoint: {checkpoint_path}")
    if saved.get("split_protocol") != PROTOCOL:
        raise ValueError(f"Unexpected split protocol: {checkpoint_path}")
    if saved.get("condition_name") != "m7_transfer_weighted_jpeg_mixed":
        raise ValueError(f"Condition mismatch: {checkpoint_path}")
    if saved.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Manifest mismatch: {checkpoint_path}")
    if saved.get("preprocessing_name") != MIXED_JPEG_REENCODE_NAME:
        raise ValueError(f"Preprocessing mismatch: {checkpoint_path}")
    if int(saved.get("seed", -1)) != 42:
        raise ValueError(f"Seed mismatch: {checkpoint_path}")
    actual = saved.get("preprocessing", {})
    expected = config["preprocessing"]
    for key in [
        "canonical_size", "train_jpeg_qualities", "train_jpeg_sampling",
        "validation_jpeg_quality", "jpeg_subsampling", "jpeg_optimize",
        "jpeg_progressive",
    ]:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Weighted preprocessing mismatch for {key}")
    return {
        "path": checkpoint_path,
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
        "data_config": saved["data_config"],
    }


def validate_prediction(prediction: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    required = {
        "sample_id", "source_path", "label", "method", "family", "role",
        "group_id", "video_id", "true_class", "predicted_class",
        "fake_probability", "correct",
    }
    missing = required - set(prediction.columns)
    if missing:
        raise RuntimeError(f"Prediction columns missing: {sorted(missing)}")
    if prediction["sample_id"].tolist() != test_frame["sample_id"].tolist():
        raise RuntimeError("Prediction order differs from the frozen test manifest")


def maybe_reuse_baseline(
    source_root: Path | None,
    condition_name: str,
    destination: Path,
    checkpoint_hash: str,
    test_hash: str,
) -> bool:
    if source_root is None:
        return False
    source = source_root / "m7" / "seed42" / condition_name
    metrics_path = source / "metrics.json"
    predictions_path = source / "predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return False
    metrics = load_json(metrics_path)
    if metrics.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"Reusable baseline checkpoint differs: {source}")
    if metrics.get("test_manifest_sha256") != test_hash:
        raise RuntimeError(f"Reusable baseline test manifest differs: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in (metrics_path, predictions_path):
        target = destination / source_file.name
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source_file, temporary)
        temporary.replace(target)
    return True


def build_report(model_summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# Transfer-weighted M7 Mixed-JPEG Pilot",
        "",
        (
            "> The proposed model keeps all six seen methods and the same "
            "19,800-image fake budget as M7 Mixed-JPEG. Only method quotas "
            "change, using training-side Q95/Q90/Q75 transfer matrices."
        ),
        "",
        "| Condition | Model | Real FPR | All-method AUC | Protected unseen AUC | Protected unseen without PixArt | PixArt AUC |",
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
            "## Transfer-weighted minus uniform M7",
            "",
            "| Condition | Real FPR delta | All-method AUC delta | Protected unseen delta | Delta without PixArt | PixArt delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.delta_real_fpr:+.4f} | "
            f"{row.delta_all_methods_macro_auc:+.4f} | "
            f"{row.delta_protected_unseen_macro_auc:+.4f} | "
            f"{row.delta_protected_unseen_excluding_pixart_macro_auc:+.4f} | "
            f"{row.delta_pixart_auc:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This is a pre-frozen seed-42 pilot. Protected unseen and "
                "WildDeepfake were not used to compute method weights. The "
                "weights must not be revised in response to these results."
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
        "weighted_manifest": args.weighted_manifest.resolve(),
        "weighted_manifest_summary": args.weighted_manifest_summary.resolve(),
        "baseline_manifest": args.baseline_manifest.resolve(),
        "weighted_runs_root": args.weighted_runs_root.resolve(),
        "baseline_runs_root": args.baseline_runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
        "baseline_config": args.baseline_config.resolve(),
        "conditions_config": args.conditions_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    baseline_evaluation_root = (
        args.baseline_evaluation_root.resolve()
        if args.baseline_evaluation_root is not None
        else None
    )
    if baseline_evaluation_root is not None and not baseline_evaluation_root.is_dir():
        raise FileNotFoundError(baseline_evaluation_root)

    config = load_json(paths["protocol_config"])
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected weighted protocol")
    manifest_summary = load_json(paths["weighted_manifest_summary"])
    if manifest_summary.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected weighted manifest summary")
    if manifest_summary.get("manifest_sha256") != sha256_file(
        paths["weighted_manifest"]
    ):
        raise ValueError("Weighted manifest hash mismatch")
    baseline_config = load_json(paths["baseline_config"])
    if baseline_config.get("protocol") != config["baseline"]["protocol"]:
        raise ValueError("Unexpected baseline protocol")

    conditions = load_conditions(paths["conditions_config"])
    m7_condition = conditions["M7"]
    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Unexpected test image count: {len(test_frame)}")
    test_hash = sha256_file(paths["test_manifest"])
    if test_hash != config["test_manifest_sha256"]:
        raise ValueError("Test manifest hash mismatch")
    test_frame["source_reference_path"] = test_frame["source_path"]
    test_frame["resolved_path"] = test_frame["source_path"]
    missing = [
        value for value in test_frame["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError("Test archive incomplete: " + ", ".join(missing[:5]))

    baseline_hash = sha256_file(paths["baseline_manifest"])
    if baseline_hash != config["baseline"]["training_manifest_sha256"]:
        raise ValueError("Uniform M7 baseline manifest hash mismatch")
    weighted_development = read_manifest(paths["weighted_manifest"])
    baseline_development = read_manifest(paths["baseline_manifest"])
    audits = {
        "uniform_m7": cross_split_audit(baseline_development, test_frame),
        "transfer_weighted_m7": cross_split_audit(weighted_development, test_frame),
    }

    baseline_checkpoint_path = (
        paths["baseline_runs_root"]
        / config["baseline"]["run_name"]
        / "best.pt"
    )
    weighted_checkpoint_path = (
        paths["weighted_runs_root"] / WEIGHTED_RUN_NAME / "best.pt"
    )
    for checkpoint_path in (baseline_checkpoint_path, weighted_checkpoint_path):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    baseline_checkpoint = torch.load(
        baseline_checkpoint_path, map_location="cpu", weights_only=False
    )
    baseline_info = validate_baseline_checkpoint(
        baseline_checkpoint,
        baseline_checkpoint_path,
        paths["baseline_manifest"],
        "M7",
        m7_condition,
        42,
        baseline_config,
    )
    weighted_checkpoint = torch.load(
        weighted_checkpoint_path, map_location="cpu", weights_only=False
    )
    weighted_info = validate_weighted_checkpoint(
        weighted_checkpoint,
        weighted_checkpoint_path,
        paths["weighted_manifest"],
        config,
    )
    if baseline_checkpoint["config"]["data_config"] != weighted_info["data_config"]:
        raise ValueError("Baseline and proposed checkpoints use different inputs")
    data_config = weighted_info["data_config"]
    del baseline_checkpoint, weighted_checkpoint

    identity = {
        "protocol": PROTOCOL,
        "protocol_config_sha256": sha256_file(paths["protocol_config"]),
        "test_manifest_sha256": test_hash,
        "weighted_manifest_sha256": sha256_file(paths["weighted_manifest"]),
        "baseline_manifest_sha256": baseline_hash,
        "checkpoints": {
            "uniform_m7": baseline_info["sha256"],
            "transfer_weighted_m7": weighted_info["sha256"],
        },
        "evaluation_conditions": [
            item["name"] for item in config["evaluation_conditions"]
        ],
    }
    output_root = paths["output_root"]
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file() or load_json(identity_path) != identity:
            raise RuntimeError("Existing evaluation belongs to another run")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required")
    loaders = {}
    for condition in config["evaluation_conditions"]:
        transform, preprocessing_name = build_transform(
            data_config, condition, config
        )
        dataset = EvaluationDataset(
            test_frame,
            paths["data_root"],
            transform,
            preprocessing_name,
            data_config["input_size"][1],
        )
        loaders[condition["name"]] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    models = {
        "uniform_m7": (baseline_checkpoint_path, baseline_info),
        "transfer_weighted_m7": (weighted_checkpoint_path, weighted_info),
    }
    model_rows = []
    method_rows = []
    for model_label, (checkpoint_path, checkpoint_info) in models.items():
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
        saved_weights = checkpoint["config"].get("class_weights")
        weight_tensor = None
        if saved_weights is not None:
            weight_tensor = torch.tensor(
                [saved_weights["real"], saved_weights["fake"]],
                dtype=torch.float32,
                device=device,
            )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        for condition in config["evaluation_conditions"]:
            condition_name = condition["name"]
            result_dir = output_root / model_label / condition_name
            metrics_path = result_dir / "metrics.json"
            predictions_path = result_dir / "predictions.csv"
            if (
                model_label == "uniform_m7"
                and not metrics_path.exists()
                and not predictions_path.exists()
                and maybe_reuse_baseline(
                    baseline_evaluation_root,
                    condition_name,
                    result_dir,
                    checkpoint_info["sha256"],
                    test_hash,
                )
            ):
                print("Reused uniform M7:", condition_name, flush=True)

            if metrics_path.is_file() and predictions_path.is_file():
                metrics_payload = load_json(metrics_path)
                if metrics_payload.get("checkpoint_sha256") != checkpoint_info["sha256"]:
                    raise RuntimeError(f"Checkpoint mismatch: {result_dir}")
                if metrics_payload.get("test_manifest_sha256") != test_hash:
                    raise RuntimeError(f"Test mismatch: {result_dir}")
                prediction = pd.read_csv(predictions_path)
                validate_prediction(prediction, test_frame)
                print("Skipping completed:", model_label, condition_name, flush=True)
            elif result_dir.exists() and any(result_dir.iterdir()):
                raise FileExistsError(f"Incomplete evaluation: {result_dir}")
            else:
                print(f"\n=== {model_label}: {condition_name} ===", flush=True)
                metrics, labels, predictions, probabilities = evaluate_with_predictions(
                    model,
                    loaders[condition_name],
                    test_frame,
                    criterion,
                    device,
                    progress_desc=f"{model_label} {condition_name}",
                    persistent_progress=True,
                )
                prediction = prediction_frame(
                    test_frame, labels, predictions, probabilities
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                temporary = predictions_path.with_suffix(".csv.tmp")
                prediction.to_csv(temporary, index=False)
                temporary.replace(predictions_path)
                write_json(
                    metrics_path,
                    {
                        "model": model_label,
                        "training_seed": 42,
                        "evaluation_condition": condition,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_info["sha256"],
                        "test_manifest_sha256": test_hash,
                        "cross_split_audit": audits[model_label],
                        "metrics": metrics,
                        "decision_rule": "argmax logits (0.5 equivalent)",
                        "test_used_for_tuning": False,
                        "versions": versions(),
                    },
                )

            model_row, current_method_rows = summarize_prediction(
                model_label, m7_condition, 42, condition, prediction
            )
            model_row["checkpoint_epoch"] = checkpoint_info["epoch"]
            model_row["best_validation_auc"] = checkpoint_info[
                "best_validation_auc"
            ]
            model_rows.append(model_row)
            method_rows.extend(current_method_rows)

        del model, checkpoint, criterion, weight_tensor
        torch.cuda.empty_cache()

    model_summary = pd.DataFrame(model_rows)
    method_summary = pd.DataFrame(method_rows)
    metric_names = [
        "real_fpr", "fake_recall", "all_methods_macro_auc",
        "protected_unseen_macro_auc",
        "protected_unseen_excluding_pixart_macro_auc", "pixart_auc",
    ]
    comparisons = []
    for condition_name, group in model_summary.groupby("evaluation_condition"):
        by_model = group.set_index("model")
        row = {"evaluation_condition": condition_name}
        for metric in metric_names:
            uniform = float(by_model.loc["uniform_m7", metric])
            weighted = float(by_model.loc["transfer_weighted_m7", metric])
            row[f"uniform_{metric}"] = uniform
            row[f"weighted_{metric}"] = weighted
            row[f"delta_{metric}"] = weighted - uniform
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons).sort_values("evaluation_condition")

    model_summary.to_csv(output_root / "model_summary.csv", index=False)
    method_summary.to_csv(output_root / "method_summary.csv", index=False)
    comparison.to_csv(output_root / "weighted_vs_uniform_comparison.csv", index=False)
    (output_root / "REPORT.md").write_text(
        build_report(model_summary, comparison), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "models": list(models),
            "training_seed": 42,
            "evaluation_conditions": [
                item["name"] for item in config["evaluation_conditions"]
            ],
            "test_images": len(test_frame),
            "protected_unseen_used_for_weighting": False,
            "wilddeepfake_used_for_weighting": False,
            "test_used_for_tuning": False,
            "cross_split_audits": audits,
            "versions": versions(),
        },
    )
    print("\n=== Weighted minus uniform M7 ===", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("Saved:", output_root, flush=True)


if __name__ == "__main__":
    main()
