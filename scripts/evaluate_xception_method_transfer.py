#!/usr/bin/env python3
"""Build Q95/Q90/Q75 cross-method transfer matrices for six seen methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
from evaluate_xception_family_coverage import (
    cross_split_audit,
    read_manifest,
)
from train_baseline import sha256_file, write_json
from xception_preprocessing import (
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
)


PROTOCOL = "method_transfer_matrix_v1"
EXPECTED_IMAGES_PER_METHOD = 2_000


def validate_transfer_test_manifest(
    frame: pd.DataFrame,
    selected_methods: set[str],
    expected_per_method: int,
) -> None:
    required = {
        "sample_id", "split", "label", "method", "family", "role",
        "group_id", "video_id", "source_ids", "driver_id",
        "content_sha256", "source_path",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test manifest columns missing: {sorted(missing)}")
    if set(frame["split"]) != {"test"}:
        raise ValueError("Transfer evaluation requires test rows only")
    selected = frame[
        (frame["method"] == "original")
        | frame["method"].isin(selected_methods)
    ]
    counts = selected.groupby("method").size().to_dict()
    expected = {"original", *selected_methods}
    if set(counts) != expected or set(counts.values()) != {expected_per_method}:
        raise RuntimeError(f"Unexpected transfer test counts: {counts}")
    for column in ("sample_id", "source_path", "content_sha256"):
        if selected[column].duplicated().any():
            raise RuntimeError(f"Duplicate selected test {column}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=project_root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument(
        "--evaluation-conditions",
        nargs="+",
        default=[
            "canonical_256_jpeg_q95",
            "canonical_256_jpeg_q90",
            "canonical_256_jpeg_q75",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_protocol(path: Path, selected_conditions: list[str]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not protocol.get("protocol"):
        raise ValueError(f"Missing protocol name in {path}")
    configured = [item["name"] for item in protocol["evaluation_conditions"]]
    if not selected_conditions or len(selected_conditions) != len(
        set(selected_conditions)
    ):
        raise ValueError("Evaluation conditions must be non-empty and unique")
    unknown = set(selected_conditions) - set(configured)
    if unknown:
        raise ValueError(f"Unknown evaluation conditions: {sorted(unknown)}")
    training_methods = list(protocol["methods"])
    evaluation_methods = protocol["evaluation_methods"]
    if len(evaluation_methods) != len(set(evaluation_methods)):
        raise ValueError("Evaluation methods must be unique")
    if set(training_methods) != set(evaluation_methods):
        missing_from_evaluation = sorted(
            set(training_methods) - set(evaluation_methods)
        )
        missing_from_training = sorted(
            set(evaluation_methods) - set(training_methods)
        )
        raise ValueError(
            "Training/evaluation method membership differs: "
            f"missing_from_evaluation={missing_from_evaluation}, "
            f"missing_from_training={missing_from_training}"
        )
    return protocol


def run_name(method: str, definition: dict, seed: int, protocol: dict) -> str:
    template = protocol.get("run_name_template")
    if template:
        return template.format(method=method, slug=definition["slug"], seed=seed)
    return (
        f"xception_method_transfer_{definition['slug']}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
        f"seed{seed}"
    )


def condition_name(method: str, definition: dict, protocol: dict) -> str:
    template = protocol.get("condition_name_template")
    if template:
        return template.format(method=method, slug=definition["slug"])
    return f"method_transfer_{definition['slug']}_jpeg_mixed"


def metrics_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "fake_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "real_recall": float(
            true_negative / (true_negative + false_positive)
        ),
        "real_fpr": float(
            false_positive / (true_negative + false_positive)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "real_mean_fake_score": float(probabilities[labels == 0].mean()),
        "fake_mean_fake_score": float(probabilities[labels == 1].mean()),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def prediction_frame(
    test_frame: pd.DataFrame,
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
) -> pd.DataFrame:
    frame = test_frame[
        [
            "sample_id",
            "source_path",
            "label",
            "method",
            "family",
            "role",
            "group_id",
            "video_id",
        ]
    ].copy()
    frame["true_class"] = labels
    frame["predicted_class"] = predictions
    frame["fake_probability"] = probabilities
    frame["correct"] = (
        frame["true_class"].to_numpy()
        == frame["predicted_class"].to_numpy()
    )
    return frame


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    training_manifest: Path,
    method: str,
    definition: dict,
    protocol: dict,
) -> dict:
    config = checkpoint["config"]
    if checkpoint_path.name != "best.pt":
        raise ValueError(f"Evaluation requires best.pt: {checkpoint_path}")
    if checkpoint["model_name"] not in {"xception", "legacy_xception"}:
        raise ValueError(f"Unexpected model in {checkpoint_path}")
    protocol_name = protocol["protocol"]
    if config.get("experiment_family") != protocol_name:
        raise ValueError(f"Unexpected experiment family in {checkpoint_path}")
    if config.get("split_protocol") != protocol_name:
        raise ValueError(f"Unexpected split protocol in {checkpoint_path}")
    if config.get("condition_name") != condition_name(method, definition, protocol):
        raise ValueError(f"Checkpoint condition mismatch: {checkpoint_path}")
    if config.get("preprocessing_name") != MIXED_JPEG_REENCODE_NAME:
        raise ValueError(f"Unexpected preprocessing in {checkpoint_path}")
    manifest_hash = sha256_file(training_manifest)
    if manifest_hash != protocol["manifest_sha256"][method]:
        raise ValueError(f"Training manifest protocol mismatch for {method}")
    if config.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Checkpoint/manifest mismatch for {method}")
    training_seed = int(protocol.get("training_seeds", [42])[0])
    if int(config.get("seed")) != training_seed:
        raise ValueError(f"Unexpected training seed for {method}")
    saved = config.get("preprocessing", {})
    expected = protocol["preprocessing"]
    for key in [
        "canonical_size",
        "train_jpeg_qualities",
        "train_jpeg_sampling",
        "validation_jpeg_quality",
        "jpeg_subsampling",
        "jpeg_optimize",
        "jpeg_progressive",
    ]:
        if saved.get(key) != expected.get(key):
            raise ValueError(
                f"Checkpoint preprocessing mismatch for {method}/{key}"
            )
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
        "data_config": config["data_config"],
    }


def summarize_pairs(
    prediction: pd.DataFrame,
    *,
    training_method: str,
    training_family: str,
    evaluation_condition: str,
    protocol: dict,
) -> list[dict]:
    real = prediction[prediction["label"] == "real"]
    rows = []
    for target_method in protocol["evaluation_methods"]:
        fake = prediction[
            (prediction["label"] == "fake")
            & (prediction["method"] == target_method)
        ]
        pair = pd.concat([real, fake], ignore_index=True)
        metrics = metrics_from_arrays(
            pair["true_class"].to_numpy(int),
            pair["predicted_class"].to_numpy(int),
            pair["fake_probability"].to_numpy(float),
        )
        target_family = protocol["methods"][target_method]["family"]
        rows.append(
            {
                "training_method": training_method,
                "training_family": training_family,
                "target_method": target_method,
                "target_family": target_family,
                "evaluation_condition": evaluation_condition,
                "in_domain": training_method == target_method,
                "same_family": training_family == target_family,
                "images_real": int(len(real)),
                "images_fake": int(len(fake)),
                **metrics,
            }
        )
    return rows


def build_report(method_summary: pd.DataFrame) -> str:
    cross = method_summary[~method_summary["in_domain"]]
    aggregate = (
        cross.groupby(["evaluation_condition", "training_method"])["roc_auc"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values(["evaluation_condition", "mean"], ascending=[True, False])
    )
    lines = [
        "# Single-method Cross-method Transfer Matrix",
        "",
        (
            "> Six single-method Xception models use identical Real/Fake "
            "budgets and the frozen Mixed-JPEG training path. Protected "
            "unseen methods and WildDeepfake are not used here."
        ),
        "",
        "## Cross-method summary",
        "",
        "| Condition | Training method | Mean AUC | Worst AUC | Best AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.training_method} | "
            f"{row.mean:.4f} | {row.min:.4f} | {row.max:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "These matrices measure training-side pairwise transfer. "
                "They do not by themselves define or validate a subset-selection "
                "algorithm. Any selection rule must be frozen and checked by an "
                "outer method holdout before protected unseen evaluation."
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
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "runs_root": args.runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    protocol = load_protocol(
        paths["protocol_config"], args.evaluation_conditions
    )
    full_test = read_manifest(paths["test_manifest"])
    test_hash = sha256_file(paths["test_manifest"])
    if test_hash != protocol["test_manifest_sha256"]:
        raise ValueError(
            f"Test manifest hash mismatch: expected="
            f"{protocol['test_manifest_sha256']}, actual={test_hash}"
        )
    selected_methods = set(protocol["evaluation_methods"])
    expected_per_method = int(
        protocol.get("expected_images_per_method", EXPECTED_IMAGES_PER_METHOD)
    )
    validate_transfer_test_manifest(
        full_test, selected_methods, expected_per_method
    )
    test_frame = full_test[
        (full_test["method"] == "original")
        | (full_test["method"].isin(selected_methods))
    ].copy().reset_index(drop=True)
    expected_rows = expected_per_method * (len(selected_methods) + 1)
    counts = test_frame.groupby("method").size().to_dict()
    if len(test_frame) != expected_rows or set(counts.values()) != {
        expected_per_method
    }:
        raise RuntimeError(f"Unexpected transfer test counts: {counts}")
    test_frame["source_reference_path"] = test_frame["source_path"]
    test_frame["resolved_path"] = test_frame["source_path"]
    missing_images = [
        value
        for value in test_frame["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Test archive is incomplete. First missing paths: "
            + ", ".join(missing_images[:5])
        )

    checkpoints = {}
    audits = {}
    reference_data_config = None
    identity = {
        "protocol": protocol["protocol"],
        "protocol_config_sha256": sha256_file(paths["protocol_config"]),
        "test_manifest_sha256": test_hash,
        "evaluation_methods": protocol["evaluation_methods"],
        "evaluation_conditions": args.evaluation_conditions,
        "checkpoints": {},
    }
    for method, definition in protocol["methods"].items():
        training_manifest = (
            paths["training_manifest_dir"]
            / f"{definition['slug']}_seed42.csv"
        )
        if not training_manifest.is_file():
            raise FileNotFoundError(training_manifest)
        development = read_manifest(training_manifest)
        audits[method] = cross_split_audit(development, full_test)
        checkpoint_path = (
            paths["runs_root"]
            / run_name(method, definition, 42, protocol)
            / "best.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        info = validate_checkpoint(
            checkpoint,
            checkpoint_path,
            training_manifest,
            method,
            definition,
            protocol,
        )
        if reference_data_config is None:
            reference_data_config = info["data_config"]
        elif info["data_config"] != reference_data_config:
            raise ValueError("Single-method checkpoints use different inputs")
        checkpoints[method] = info
        identity["checkpoints"][method] = {
            "sha256": info["sha256"],
            "epoch": info["epoch"],
            "training_manifest_sha256": sha256_file(training_manifest),
        }

    output_root = paths["output_root"]
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(
                f"Non-empty output has no run identity: {output_root}"
            )
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing evaluation belongs to another run")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    conditions = [
        item
        for item in protocol["evaluation_conditions"]
        if item["name"] in args.evaluation_conditions
    ]
    preprocessing = protocol["preprocessing"]
    loaders = {}
    for condition in conditions:
        transform = canonical_reencode_transform(
            reference_data_config,
            canonical_size=preprocessing["canonical_size"],
            jpeg_quality=condition["jpeg_quality"],
            jpeg_subsampling=preprocessing["jpeg_subsampling"],
            jpeg_optimize=preprocessing["jpeg_optimize"],
            jpeg_progressive=preprocessing["jpeg_progressive"],
        )
        dataset = EvaluationDataset(
            test_frame,
            paths["data_root"],
            transform,
            MIXED_JPEG_REENCODE_NAME,
            reference_data_config["input_size"][1],
        )
        loaders[condition["name"]] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required")
    all_rows = []
    model_rows = []
    for training_method, definition in protocol["methods"].items():
        checkpoint_path = Path(checkpoints[training_method]["path"])
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
        weights = None
        if saved_weights is not None:
            weights = torch.tensor(
                [saved_weights["real"], saved_weights["fake"]],
                dtype=torch.float32,
                device=device,
            )
        criterion = nn.CrossEntropyLoss(weight=weights)
        for condition in conditions:
            condition_name = condition["name"]
            result_dir = output_root / definition["slug"] / condition_name
            predictions_path = result_dir / "predictions.csv"
            metrics_path = result_dir / "metrics.json"
            if predictions_path.is_file() and metrics_path.is_file():
                prediction = pd.read_csv(predictions_path)
                if prediction["sample_id"].tolist() != test_frame[
                    "sample_id"
                ].tolist():
                    raise RuntimeError(
                        f"Prediction order changed: {predictions_path}"
                    )
                print(
                    f"Skipping completed: {training_method}/{condition_name}",
                    flush=True,
                )
            elif result_dir.exists() and any(result_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete evaluation directory: {result_dir}"
                )
            else:
                print(
                    f"\n=== Train {training_method} -> all seen methods: "
                    f"{condition_name} ===",
                    flush=True,
                )
                metrics, labels, predictions, probabilities = (
                    evaluate_with_predictions(
                        model,
                        loaders[condition_name],
                        test_frame,
                        criterion,
                        device,
                        progress_desc=f"{training_method} {condition_name}",
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
                        "training_method": training_method,
                        "training_family": definition["family"],
                        "evaluation_condition": condition,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": checkpoints[training_method][
                            "sha256"
                        ],
                        "test_manifest_sha256": test_hash,
                        "cross_split_audit": audits[training_method],
                        "metrics": metrics,
                        "test_methods": protocol["evaluation_methods"],
                        "protected_unseen_used": False,
                        "wilddeepfake_used": False,
                    },
                )
            rows = summarize_pairs(
                prediction,
                training_method=training_method,
                training_family=definition["family"],
                evaluation_condition=condition_name,
                protocol=protocol,
            )
            all_rows.extend(rows)
            cross_values = [row["roc_auc"] for row in rows if not row["in_domain"]]
            diagonal = next(row for row in rows if row["in_domain"])
            model_rows.append(
                {
                    "training_method": training_method,
                    "training_family": definition["family"],
                    "evaluation_condition": condition_name,
                    "in_domain_auc": diagonal["roc_auc"],
                    "cross_method_mean_auc": float(np.mean(cross_values)),
                    "cross_method_min_auc": float(np.min(cross_values)),
                    "cross_method_max_auc": float(np.max(cross_values)),
                    "checkpoint_epoch": checkpoints[training_method]["epoch"],
                    "best_validation_auc": checkpoints[training_method][
                        "best_validation_auc"
                    ],
                }
            )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    method_summary = pd.DataFrame(all_rows).sort_values(
        ["evaluation_condition", "training_method", "target_method"]
    )
    model_summary = pd.DataFrame(model_rows).sort_values(
        ["evaluation_condition", "training_method"]
    )
    method_summary.to_csv(output_root / "transfer_matrix_long.csv", index=False)
    model_summary.to_csv(output_root / "model_summary.csv", index=False)
    for condition in conditions:
        name = condition["name"]
        matrix = method_summary[
            method_summary["evaluation_condition"] == name
        ].pivot(
            index="training_method",
            columns="target_method",
            values="roc_auc",
        )
        matrix = matrix.loc[
            protocol["evaluation_methods"], protocol["evaluation_methods"]
        ]
        matrix.to_csv(output_root / f"transfer_matrix_{name}.csv")

    q95 = "canonical_256_jpeg_q95"
    if q95 in set(method_summary["evaluation_condition"]):
        reference = method_summary[
            method_summary["evaluation_condition"] == q95
        ][["training_method", "target_method", "roc_auc", "real_fpr"]].rename(
            columns={"roc_auc": "q95_roc_auc", "real_fpr": "q95_real_fpr"}
        )
        deltas = method_summary.merge(
            reference, on=["training_method", "target_method"], how="left"
        )
        deltas = deltas[deltas["evaluation_condition"] != q95].copy()
        deltas["delta_roc_auc_from_q95"] = (
            deltas["roc_auc"] - deltas["q95_roc_auc"]
        )
        deltas["delta_real_fpr_from_q95"] = (
            deltas["real_fpr"] - deltas["q95_real_fpr"]
        )
        deltas.to_csv(output_root / "jpeg_condition_deltas.csv", index=False)

    (output_root / "REPORT.md").write_text(
        build_report(method_summary), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "protocol": protocol["protocol"],
            "training_methods": protocol["evaluation_methods"],
            "evaluation_methods": protocol["evaluation_methods"],
            "evaluation_conditions": args.evaluation_conditions,
            "test_images": int(len(test_frame)),
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
            "outputs": {
                "transfer_matrix_long": "transfer_matrix_long.csv",
                "model_summary": "model_summary.csv",
                "report": "REPORT.md",
            },
        },
    )
    print("Saved:", output_root, flush=True)


if __name__ == "__main__":
    main()
