#!/usr/bin/env python3
"""Evaluate the frozen M7 JPEG-consistency pilot at Q95/Q90/Q75."""

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
from PIL import features
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
    aggregate_jpeg_condition_deltas,
    compare_jpeg_conditions,
    prediction_frame,
    summarize_prediction,
)
from train_baseline import LABEL_MAP, sha256_file, write_json
from train_xception_jpeg_consistency import (
    CONDITION_ID,
    EXPERIMENT_FAMILY,
    PREPROCESSING_NAME,
    load_protocol,
    run_name,
)
from xception_preprocessing import canonical_reencode_transform


EXPECTED_TEST_IMAGES = 30_000


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=(
            project_root
            / "configs/family_coverage_jpeg_consistency_pilot_v1/"
            "protocol.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "pillow": PIL.__version__,
        "libjpeg": features.version_codec("jpg"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


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
        "condition_name": "m7_fs_fr_efs_jpeg_consistency",
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
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
    }


def build_report(model_summary: pd.DataFrame) -> str:
    lines = [
        "# M7 Paired-JPEG Consistency Pilot",
        "",
        (
            "> One seed-42 M7 pilot. Training uses two distinct JPEG "
            "qualities per source image and a Jensen–Shannon consistency "
            "term with weight 0.1. This is diagnostic pilot evidence, not "
            "a final multi-seed result."
        ),
        "",
        (
            "| Test condition | Real FPR | All-method AUC | Protected "
            "unseen AUC | Protected unseen without PixArt | PixArt AUC |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.sort_values("evaluation_condition").itertuples(
        index=False
    ):
        lines.append(
            f"| {row.evaluation_condition} | {row.real_fpr:.4f} | "
            f"{row.all_methods_macro_auc:.4f} | "
            f"{row.protected_unseen_macro_auc:.4f} | "
            f"{row.protected_unseen_excluding_pixart_macro_auc:.4f} | "
            f"{row.pixart_auc:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            (
                "Compare this pilot only with the existing M7 Mixed-JPEG "
                "seed-42 checkpoint on the same Q95/Q90/Q75 test protocol. "
                "WildDeepfake must not be used to choose the consistency "
                "weight or other hyperparameters."
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
        "runs_root": args.runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "pilot_config": args.pilot_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    protocol = load_protocol(paths["pilot_config"])
    if sha256_file(paths["training_manifest"]) != protocol[
        "manifest_sha256"
    ][CONDITION_ID]:
        raise ValueError("M7 training manifest hash mismatch.")
    if sha256_file(paths["test_manifest"]) != protocol[
        "test_manifest_sha256"
    ]:
        raise ValueError("Balanced test manifest hash mismatch.")

    conditions = load_conditions(paths["conditions_config"])
    training_condition = conditions[CONDITION_ID]
    development = read_manifest(paths["training_manifest"])
    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise ValueError("Unexpected test image count.")
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
            "Test archive is incomplete. First missing paths: "
            + ", ".join(missing[:5])
        )

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
        protocol,
    )
    data_config = checkpoint["config"]["data_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required.")
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    weights = checkpoint["config"]["class_weights"]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [weights["real"], weights["fake"]],
            dtype=torch.float32,
            device=device,
        )
    )

    output_root = paths["output_root"]
    identity = {
        "protocol": EXPERIMENT_FAMILY,
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "training_manifest_sha256": sha256_file(
            paths["training_manifest"]
        ),
        "test_manifest_sha256": sha256_file(paths["test_manifest"]),
        "checkpoint_sha256": checkpoint_info["sha256"],
        "evaluation_conditions": [
            item["name"] for item in protocol["evaluation_conditions"]
        ],
    }
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(
                f"Non-empty output has no run identity: {output_root}"
            )
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing evaluation belongs to another run.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    model_rows = []
    method_rows_all = []
    preprocessing = protocol["preprocessing"]
    for evaluation_condition in protocol["evaluation_conditions"]:
        condition_name = evaluation_condition["name"]
        result_dir = output_root / condition_name
        metrics_path = result_dir / "metrics.json"
        predictions_path = result_dir / "predictions.csv"
        if metrics_path.is_file() and predictions_path.is_file():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result["checkpoint_sha256"] != checkpoint_info["sha256"]:
                raise RuntimeError("Existing result used another checkpoint.")
            prediction = pd.read_csv(predictions_path)
            if prediction["sample_id"].astype(str).tolist() != test_frame[
                "sample_id"
            ].astype(str).tolist():
                raise RuntimeError("Prediction order changed.")
            print(f"Skipping completed: {condition_name}", flush=True)
        elif result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete evaluation directory: {result_dir}"
            )
        else:
            transform = canonical_reencode_transform(
                data_config,
                canonical_size=evaluation_condition["canonical_size"],
                jpeg_quality=evaluation_condition["jpeg_quality"],
                jpeg_subsampling=preprocessing["jpeg_subsampling"],
                jpeg_optimize=preprocessing["jpeg_optimize"],
                jpeg_progressive=preprocessing["jpeg_progressive"],
            )
            dataset = EvaluationDataset(
                test_frame,
                paths["data_root"],
                transform,
                PREPROCESSING_NAME,
                data_config["input_size"][1],
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=args.workers > 0,
            )
            print(f"\n=== M7 consistency: {condition_name} ===", flush=True)
            metrics, labels, predictions, probabilities = (
                evaluate_with_predictions(
                    model,
                    loader,
                    test_frame,
                    criterion,
                    device,
                    progress_desc=f"M7 consistency {condition_name}",
                    persistent_progress=True,
                )
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
                    "model": CONDITION_ID,
                    "training_seed": 42,
                    "evaluation_condition": evaluation_condition,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "training_manifest": str(paths["training_manifest"]),
                    "training_manifest_sha256": identity[
                        "training_manifest_sha256"
                    ],
                    "test_manifest": str(paths["test_manifest"]),
                    "test_manifest_sha256": identity[
                        "test_manifest_sha256"
                    ],
                    "cross_split_audit": audit,
                    "metrics": metrics,
                    "decision_rule": "argmax logits (0.5 equivalent)",
                    "test_used_for_tuning": False,
                    "versions": versions(),
                },
            )
        model_row, method_rows = summarize_prediction(
            CONDITION_ID,
            training_condition,
            42,
            evaluation_condition,
            prediction,
        )
        model_row.update(
            {
                "checkpoint_epoch": checkpoint_info["epoch"],
                "best_validation_auc": checkpoint_info[
                    "best_validation_auc"
                ],
            }
        )
        model_rows.append(model_row)
        method_rows_all.extend(method_rows)

    model_summary = pd.DataFrame(model_rows)
    method_summary = pd.DataFrame(method_rows_all)
    jpeg_deltas = compare_jpeg_conditions(model_summary)
    jpeg_delta_aggregate = aggregate_jpeg_condition_deltas(jpeg_deltas)
    model_summary.to_csv(output_root / "model_summary.csv", index=False)
    method_summary.to_csv(output_root / "method_summary.csv", index=False)
    jpeg_deltas.to_csv(
        output_root / "jpeg_condition_delta_summary.csv", index=False
    )
    jpeg_delta_aggregate.to_csv(
        output_root / "jpeg_condition_delta_aggregate.csv", index=False
    )
    (output_root / "REPORT.md").write_text(
        build_report(model_summary), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": EXPERIMENT_FAMILY,
            "model": CONDITION_ID,
            "training_seed": 42,
            "evaluation_conditions": identity["evaluation_conditions"],
            "test_images": len(test_frame),
            "cross_split_audit": audit,
            "source_images_modified": False,
            "test_used_for_training_or_tuning": False,
            "versions": versions(),
        },
    )
    print("\nSaved to:", output_root, flush=True)


if __name__ == "__main__":
    main()
