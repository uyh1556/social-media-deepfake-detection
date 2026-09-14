#!/usr/bin/env python3
"""Evaluate M1-M7 checkpoints on one frozen balanced all-method test set."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from evaluate_checkpoint import (
    EvaluationDataset,
    evaluate_with_predictions,
    write_predictions,
)
from train_baseline import LABEL_MAP, sha256_file, write_json
from xception_preprocessing import (
    LETTERBOX_NAME,
    evaluation_transform_from_checkpoint,
)


MODEL_IDS = [f"M{index}" for index in range(1, 8)]
EXPECTED_TEST_IMAGES = 30_000
EXPECTED_IMAGES_PER_METHOD = 2_000
FFPP_METHODS = {"Deepfakes", "Face2Face"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/family_coverage_v1/conditions.json"
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=MODEL_IDS,
        default=MODEL_IDS,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def split_tokens(values: pd.Series) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(
            token.strip()
            for token in str(value).replace(",", "|").split("|")
            if token.strip()
        )
    return result


def nonempty(values: pd.Series) -> set[str]:
    return {str(value).strip() for value in values if str(value).strip()}


def method_video_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row.method), str(row.video_id))
        for row in frame.itertuples(index=False)
        if str(row.video_id).strip()
    }


def cross_split_audit(
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, int]:
    audit = {
        "sample_id_overlap": len(
            nonempty(development["sample_id"]) & nonempty(test["sample_id"])
        ),
        "source_path_overlap": len(
            nonempty(development["source_path"])
            & nonempty(test["source_path"])
        ),
        "content_hash_overlap": len(
            nonempty(development["content_sha256"])
            & nonempty(test["content_sha256"])
        ),
        "group_id_overlap": len(
            nonempty(development["group_id"])
            & nonempty(test["group_id"])
        ),
        "method_video_overlap": len(
            method_video_keys(development) & method_video_keys(test)
        ),
        "source_id_overlap": len(
            split_tokens(development["source_ids"])
            & split_tokens(test["source_ids"])
        ),
        "driver_id_overlap": len(
            split_tokens(development["driver_id"])
            & split_tokens(test["driver_id"])
        ),
    }
    if any(audit.values()):
        raise RuntimeError(f"Development/test leakage detected: {audit}")
    return audit


def validate_test_manifest(frame: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "split",
        "label",
        "method",
        "family",
        "role",
        "group_id",
        "video_id",
        "source_ids",
        "driver_id",
        "content_sha256",
        "source_path",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test manifest columns missing: {sorted(missing)}")
    if set(frame["split"]) != {"test"}:
        raise ValueError("The balanced evaluation manifest must contain only test rows.")
    if len(frame) != EXPECTED_TEST_IMAGES:
        raise ValueError(f"Expected {EXPECTED_TEST_IMAGES} test images, found {len(frame)}")
    counts = frame.groupby("method").size()
    if len(counts) != 15 or set(counts.tolist()) != {EXPECTED_IMAGES_PER_METHOD}:
        raise ValueError(f"Unexpected per-method test counts: {counts.to_dict()}")
    if set(frame.loc[frame["label"] == "real", "method"]) != {"original"}:
        raise ValueError("The real test class must contain only original images.")
    for column in ("sample_id", "source_path", "content_sha256"):
        if frame[column].duplicated().any():
            raise RuntimeError(f"Duplicate test {column}")


def checkpoint_path(
    runs_root: Path,
    model_id: str,
    condition: dict,
    seed: int,
) -> Path:
    run_name = (
        f"xception_{model_id.lower()}_{condition['name']}_"
        f"letterbox299_family_coverage_v1_seed{seed}"
    )
    return runs_root / run_name / "best.pt"


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_file: Path,
    training_manifest: Path,
    model_id: str,
    condition: dict,
) -> dict[str, int]:
    config = checkpoint["config"]
    if checkpoint_file.name != "best.pt":
        raise ValueError(f"Evaluation requires best.pt: {checkpoint_file}")
    if checkpoint["model_name"] not in {"xception", "legacy_xception"}:
        raise ValueError(f"Unexpected model in {checkpoint_file}")
    if config.get("preprocessing_name") != LETTERBOX_NAME:
        raise ValueError(f"Unexpected preprocessing in {checkpoint_file}")
    expected_name = f"{model_id.lower()}_{condition['name']}"
    if config.get("condition_name") != expected_name:
        raise ValueError(
            f"Checkpoint condition mismatch: expected={expected_name}, "
            f"actual={config.get('condition_name')}"
        )
    training_hash = sha256_file(training_manifest)
    if config.get("manifest_sha256") != training_hash:
        raise ValueError(
            f"Checkpoint/training manifest mismatch for {model_id}: "
            f"checkpoint={config.get('manifest_sha256')}, manifest={training_hash}"
        )
    return {
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
    }


def metric_mean(
    method_rows: list[dict],
    metric: str,
    *,
    predicate=lambda row: True,
) -> float | None:
    values = [
        row[metric]
        for row in method_rows
        if predicate(row) and row.get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def make_method_rows(
    model_id: str,
    condition: dict,
    test_frame: pd.DataFrame,
    metrics: dict,
) -> list[dict]:
    trained = set(condition["seen_fake_methods"])
    rows = []
    fake = test_frame[test_frame["label"] == "fake"]
    for method, method_frame in fake.groupby("method", sort=True):
        key = f"original_vs_{method}"
        pair = metrics["by_manipulation"][key]
        confusion = pair["confusion_matrix"]
        true_negative, false_positive = confusion[0]
        false_negative, true_positive = confusion[1]
        rows.append(
            {
                "model": model_id,
                "condition": condition["name"],
                "method": method,
                "family": method_frame["family"].iloc[0],
                "dataset_role": method_frame["role"].iloc[0],
                "relative_status": (
                    "trained_method" if method in trained else "held_out_method"
                ),
                "images_fake": int(len(method_frame)),
                "images_real": EXPECTED_IMAGES_PER_METHOD,
                "accuracy": float(pair["accuracy"]),
                "precision": float(pair["precision"]),
                "fake_recall": float(pair["recall"]),
                "real_recall": (
                    float(true_negative / (true_negative + false_positive))
                    if true_negative + false_positive
                    else 0.0
                ),
                "f1": float(pair["f1"]),
                "roc_auc": float(pair["roc_auc"]),
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
            }
        )
    return rows


def make_model_row(
    model_id: str,
    condition: dict,
    checkpoint_info: dict,
    metrics: dict,
    method_rows: list[dict],
) -> dict:
    confusion = metrics["confusion_matrix"]
    true_negative, false_positive = confusion[0]
    return {
        "model": model_id,
        "condition": condition["name"],
        "families": "+".join(condition["families"]),
        "trained_fake_methods": "|".join(condition["seen_fake_methods"]),
        "checkpoint_epoch": checkpoint_info["epoch"],
        "best_validation_auc": checkpoint_info["best_validation_auc"],
        "test_images": EXPECTED_TEST_IMAGES,
        "overall_accuracy_micro": float(metrics["accuracy"]),
        "overall_precision_micro": float(metrics["precision"]),
        "overall_fake_recall_micro": float(metrics["recall"]),
        "overall_real_recall": float(
            true_negative / (true_negative + false_positive)
        ),
        "overall_f1_micro": float(metrics["f1"]),
        "overall_roc_auc_micro": float(metrics["roc_auc"]),
        "all_methods_macro_auc": metric_mean(method_rows, "roc_auc"),
        "all_methods_macro_f1": metric_mean(method_rows, "f1"),
        "trained_methods_macro_auc": metric_mean(
            method_rows,
            "roc_auc",
            predicate=lambda row: row["relative_status"] == "trained_method",
        ),
        "held_out_methods_macro_auc": metric_mean(
            method_rows,
            "roc_auc",
            predicate=lambda row: row["relative_status"] == "held_out_method",
        ),
        "protected_unseen_macro_auc": metric_mean(
            method_rows,
            "roc_auc",
            predicate=lambda row: row["dataset_role"] == "protected_unseen",
        ),
        "ffpp_methods_macro_auc": metric_mean(
            method_rows,
            "roc_auc",
            predicate=lambda row: row["method"] in FFPP_METHODS,
        ),
    }


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "torchvision": torchvision.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    test_manifest = args.test_manifest.resolve()
    training_manifest_dir = args.training_manifest_dir.resolve()
    runs_root = args.runs_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.conditions_config.resolve()
    for path in (
        data_root,
        test_manifest,
        training_manifest_dir,
        runs_root,
        config_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = protocol["conditions"]

    test_frame = read_manifest(test_manifest)
    validate_test_manifest(test_frame)
    test_frame["source_reference_path"] = test_frame["source_path"]
    test_frame["resolved_path"] = test_frame["source_path"]
    missing_images = [
        value
        for value in test_frame["source_path"]
        if not (data_root / value).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Test archive is not fully extracted. First missing paths: "
            + ", ".join(missing_images[:5])
        )

    checkpoints = {}
    training_manifests = {}
    audits = {}
    reference_data_config = None
    for model_id in args.conditions:
        condition = conditions[model_id]
        training_manifest = (
            training_manifest_dir / f"{model_id.lower()}_seed42.csv"
        )
        checkpoint_file = checkpoint_path(
            runs_root, model_id, condition, args.seed
        )
        for path in (training_manifest, checkpoint_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_info = validate_checkpoint(
            checkpoint,
            checkpoint_file,
            training_manifest,
            model_id,
            condition,
        )
        development = read_manifest(training_manifest)
        audits[model_id] = cross_split_audit(development, test_frame)
        data_config = checkpoint["config"]["data_config"]
        if reference_data_config is None:
            reference_data_config = data_config
        elif data_config != reference_data_config:
            raise ValueError("M1-M7 checkpoints do not share one input configuration.")
        checkpoints[model_id] = {
            "path": checkpoint_file,
            **checkpoint_info,
        }
        training_manifests[model_id] = training_manifest
        del checkpoint

    first_checkpoint = torch.load(
        checkpoints[args.conditions[0]]["path"],
        map_location="cpu",
        weights_only=False,
    )
    transform = evaluation_transform_from_checkpoint(first_checkpoint)
    input_size = first_checkpoint["config"]["data_config"]["input_size"][1]
    del first_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required.")
    dataset = EvaluationDataset(
        test_frame,
        data_root,
        transform,
        LETTERBOX_NAME,
        input_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    model_rows = []
    all_method_rows = []
    test_hash = sha256_file(test_manifest)
    for model_id in args.conditions:
        condition = conditions[model_id]
        model_dir = output_root / model_id.lower()
        metrics_path = model_dir / "metrics.json"
        predictions_path = model_dir / "predictions.csv"
        if metrics_path.is_file() and predictions_path.is_file():
            print(f"Skipping completed evaluation: {model_id}", flush=True)
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result["test_manifest"]["sha256"] != test_hash:
                raise RuntimeError(
                    f"Existing {model_id} result used a different test manifest."
                )
            if (
                result["training_manifest"]["sha256"]
                != sha256_file(training_manifests[model_id])
            ):
                raise RuntimeError(
                    f"Existing {model_id} result used a different training manifest."
                )
            metrics = result["metrics"]
            checkpoint_info = result["checkpoint"]
        elif model_dir.exists() and any(model_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete output directory exists: {model_dir}. "
                "Inspect it before retrying."
            )
        else:
            print(f"\n=== Evaluating {model_id} on 30,000 images ===", flush=True)
            checkpoint = torch.load(
                checkpoints[model_id]["path"],
                map_location=device,
                weights_only=False,
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
            metrics, labels, predictions, probabilities = evaluate_with_predictions(
                model,
                loader,
                test_frame,
                criterion,
                device,
                progress_desc=f"{model_id} test",
                persistent_progress=True,
            )
            checkpoint_info = {
                "path": str(checkpoints[model_id]["path"]),
                "sha256": sha256_file(checkpoints[model_id]["path"]),
                "epoch": int(checkpoint["epoch"]),
                "best_validation_auc": float(checkpoint["best_val_auc"]),
            }
            result = {
                "condition": model_id,
                "condition_name": condition["name"],
                "trained_fake_methods": condition["seen_fake_methods"],
                "checkpoint": checkpoint_info,
                "training_manifest": {
                    "path": str(training_manifests[model_id]),
                    "sha256": sha256_file(training_manifests[model_id]),
                },
                "test_manifest": {
                    "path": str(test_manifest),
                    "sha256": test_hash,
                    "images": len(test_frame),
                },
                "cross_split_audit": audits[model_id],
                "checkpoint_selection": "best validation ROC-AUC",
                "decision_rule": "argmax logits (binary threshold equivalent: 0.5)",
                "test_set_used_for_threshold_selection": False,
                "metrics": metrics,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "versions": versions(),
            }
            model_dir.mkdir(parents=True, exist_ok=True)
            write_predictions(
                predictions_path,
                test_frame,
                labels,
                predictions,
                probabilities,
            )
            write_json(metrics_path, result)
            del model, checkpoint, criterion, weight_tensor
            torch.cuda.empty_cache()

        method_rows = make_method_rows(
            model_id,
            condition,
            test_frame,
            metrics,
        )
        model_rows.append(
            make_model_row(
                model_id,
                condition,
                {
                    "epoch": int(checkpoint_info["epoch"]),
                    "best_validation_auc": float(
                        checkpoint_info["best_validation_auc"]
                    ),
                },
                metrics,
                method_rows,
            )
        )
        all_method_rows.extend(method_rows)

    model_summary = pd.DataFrame(model_rows)
    method_summary = pd.DataFrame(all_method_rows)
    model_summary.to_csv(output_root / "model_summary.csv", index=False)
    method_summary.to_csv(output_root / "method_summary.csv", index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "family_coverage_balanced_all_methods_evaluation_v1",
        "conditions": args.conditions,
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": test_hash,
        "test_images": len(test_frame),
        "images_per_method": EXPECTED_IMAGES_PER_METHOD,
        "fake_methods": 14,
        "real_methods": 1,
        "checkpoint_selection": "best validation ROC-AUC",
        "decision_rule": "argmax logits (binary threshold equivalent: 0.5)",
        "test_set_used_for_threshold_selection": False,
        "cross_split_audits": audits,
        "outputs": {
            "model_summary": str(output_root / "model_summary.csv"),
            "method_summary": str(output_root / "method_summary.csv"),
        },
        "versions": versions(),
    }
    write_json(output_root / "evaluation_summary.json", summary)
    print("\n=== Model summary ===")
    print(model_summary.to_string(index=False))
    print(f"\nSaved to: {output_root}")


if __name__ == "__main__":
    main()
