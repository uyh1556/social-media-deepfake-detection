#!/usr/bin/env python3
"""Evaluate basic, fixed-Q95, and Mixed-JPEG M0-M7 on WildDeepfake."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
    checkpoint_path as basic_checkpoint_path,
    load_conditions,
    read_manifest,
    validate_checkpoint as validate_basic_checkpoint,
)
from evaluate_xception_reencoding_control import (
    checkpoint_path as controlled_checkpoint_path,
    validate_checkpoint as validate_controlled_checkpoint,
    versions,
)
from train_baseline import sha256_file, write_json
from xception_preprocessing import (
    LETTERBOX_NAME,
    letterbox_transforms,
)


MODEL_IDS = [f"M{index}" for index in range(8)]
PROTOCOL_NAMES = ["basic", "fixed_q95", "mixed_jpeg"]
EXPECTED_COLUMNS = {
    "sample_id",
    "dataset",
    "split",
    "label",
    "target",
    "sequence_uid",
    "archive_id",
    "sequence_id",
    "frame_number",
    "selection_order",
    "relative_path",
    "width",
    "height",
    "format",
    "mode",
    "sha256",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--basic-runs-root", type=Path, required=True)
    parser.add_argument("--q95-runs-root", type=Path, required=True)
    parser.add_argument("--mixed-runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=project_root / "configs/wilddeepfake_evaluation_v1/protocol.json",
    )
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--q95-config",
        type=Path,
        default=(
            project_root
            / "configs/family_coverage_reencoding_control_v1/m0_m7_protocol.json"
        ),
    )
    parser.add_argument(
        "--mixed-config",
        type=Path,
        default=(
            project_root
            / "configs/family_coverage_jpeg_mixed_v1/m0_m7_3seed_protocol.json"
        ),
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=PROTOCOL_NAMES,
        default=PROTOCOL_NAMES,
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_IDS, default=MODEL_IDS
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol_config(config: dict) -> None:
    if config.get("protocol") != (
        "wilddeepfake_native_letterbox299_sequence_evaluation_v1"
    ):
        raise ValueError("Unexpected WildDeepfake evaluation protocol.")
    if config.get("models") != MODEL_IDS:
        raise ValueError("WildDeepfake protocol must define M0-M7 in order.")
    if set(config.get("model_protocols", {})) != set(PROTOCOL_NAMES):
        raise ValueError("WildDeepfake model protocol definitions are incomplete.")


def validate_manifest(
    path: Path, data_root: Path, protocol: dict
) -> pd.DataFrame:
    expected = protocol["manifest"]
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(
            "WildDeepfake manifest hash mismatch: "
            f"expected={expected['sha256']}, actual={actual_hash}"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"WildDeepfake manifest columns missing: {sorted(missing)}")
    if len(frame) != expected["images"]:
        raise ValueError(f"Expected {expected['images']} images, found {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("WildDeepfake sample_id values must be unique.")
    if set(frame["dataset"]) != {"WildDeepfake"}:
        raise ValueError("Unexpected dataset name in WildDeepfake manifest.")
    if set(frame["split"]) != {"test"}:
        raise ValueError("WildDeepfake manifest must contain only test rows.")
    if set(frame["label"]) != {"real", "fake"}:
        raise ValueError("WildDeepfake labels must be real/fake.")
    targets = frame["target"].astype(int)
    expected_targets = frame["label"].map({"real": 0, "fake": 1})
    if not targets.equals(expected_targets):
        raise ValueError("WildDeepfake target and label columns disagree.")
    image_counts = frame.groupby("label").size().to_dict()
    if image_counts != expected["class_counts_images"]:
        raise ValueError(f"Unexpected image class counts: {image_counts}")
    sequence_counts = (
        frame[["sequence_uid", "label"]]
        .drop_duplicates()
        .groupby("label")
        .size()
        .to_dict()
    )
    if sequence_counts != expected["class_counts_sequences"]:
        raise ValueError(f"Unexpected sequence class counts: {sequence_counts}")
    frames_per_sequence = frame.groupby("sequence_uid").size()
    if set(frames_per_sequence) != {expected["frames_per_sequence"]}:
        raise ValueError("Every sequence must contain exactly 16 selected frames.")
    if frames_per_sequence.size != expected["sequences"]:
        raise ValueError("Unexpected WildDeepfake sequence count.")
    missing_paths = [
        value
        for value in frame["relative_path"]
        if not (data_root / value).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "WildDeepfake archive is incomplete. First missing paths: "
            + ", ".join(missing_paths[:5])
        )
    frame["method"] = frame["label"].map(
        {"real": "original", "fake": "WildDeepfake"}
    )
    frame["source_path"] = frame["relative_path"]
    frame["source_reference_path"] = frame["relative_path"]
    frame["resolved_path"] = frame["relative_path"]
    return frame


def metrics_from_arrays(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "fake_recall": float(recall_score(labels, predictions, zero_division=0)),
        "real_recall": float(true_negative / (true_negative + false_positive)),
        "real_fpr": float(false_positive / (true_negative + false_positive)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "real_mean_fake_score": float(scores[labels == 0].mean()),
        "fake_mean_fake_score": float(scores[labels == 1].mean()),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def make_frame_predictions(
    frame: pd.DataFrame,
    labels: list[int],
    probabilities: list[float],
) -> pd.DataFrame:
    result = frame[
        [
            "sample_id",
            "sequence_uid",
            "archive_id",
            "sequence_id",
            "frame_number",
            "selection_order",
            "relative_path",
            "label",
        ]
    ].copy()
    result["true_class"] = labels
    result["fake_probability"] = probabilities
    result["predicted_class"] = (
        result["fake_probability"].to_numpy(float) >= 0.5
    ).astype(int)
    result["correct"] = result["true_class"] == result["predicted_class"]
    return result


def make_sequence_predictions(frame_predictions: pd.DataFrame) -> pd.DataFrame:
    label_counts = frame_predictions.groupby("sequence_uid")["true_class"].nunique()
    if label_counts.max() != 1:
        raise RuntimeError("A WildDeepfake sequence contains mixed labels.")
    rows = []
    for sequence_uid, group in frame_predictions.groupby(
        "sequence_uid", sort=True
    ):
        score = float(group["fake_probability"].mean())
        label = int(group["true_class"].iloc[0])
        rows.append(
            {
                "sequence_uid": sequence_uid,
                "archive_id": group["archive_id"].iloc[0],
                "sequence_id": group["sequence_id"].iloc[0],
                "label": group["label"].iloc[0],
                "true_class": label,
                "frames": int(len(group)),
                "mean_fake_probability": score,
                "median_fake_probability": float(
                    group["fake_probability"].median()
                ),
                "min_fake_probability": float(group["fake_probability"].min()),
                "max_fake_probability": float(group["fake_probability"].max()),
                "predicted_class": int(score >= 0.5),
                "correct": int(score >= 0.5) == label,
            }
        )
    return pd.DataFrame(rows)


def training_overlap_audit(
    wild: pd.DataFrame, training: pd.DataFrame
) -> dict[str, int]:
    audit = {
        "sample_id_overlap": len(
            set(wild["sample_id"]) & set(training.get("sample_id", []))
        ),
        "path_overlap": len(
            set(wild["relative_path"]) & set(training.get("source_path", []))
        ),
        "content_hash_overlap": len(
            set(wild["sha256"]) & set(training.get("content_sha256", []))
        ),
    }
    if any(audit.values()):
        raise RuntimeError(f"Training/WildDeepfake overlap detected: {audit}")
    return audit


def protocol_entries(
    args: argparse.Namespace,
    protocol: dict,
    q95_control: dict,
    mixed_control: dict,
) -> dict[str, dict]:
    entries = {
        "basic": {
            "runs_root": args.basic_runs_root.resolve(),
            "seeds": protocol["model_protocols"]["basic"]["training_seeds"],
            "control": None,
            "preprocessing": "native_letterbox299",
        },
        "fixed_q95": {
            "runs_root": args.q95_runs_root.resolve(),
            "seeds": protocol["model_protocols"]["fixed_q95"]["training_seeds"],
            "control": q95_control,
            "preprocessing": "native_letterbox299",
        },
        "mixed_jpeg": {
            "runs_root": args.mixed_runs_root.resolve(),
            "seeds": protocol["model_protocols"]["mixed_jpeg"]["training_seeds"],
            "control": mixed_control,
            "preprocessing": "native_letterbox299",
        },
    }
    return {name: entries[name] for name in args.protocols}


def validate_controls(q95: dict, mixed: dict) -> None:
    expected = {
        "family_coverage_reencoding_control_v1": q95,
        "family_coverage_jpeg_mixed_v1": mixed,
    }
    for protocol_name, control in expected.items():
        if control.get("protocol") != protocol_name:
            raise ValueError(f"Unexpected control config for {protocol_name}.")
        if control.get("models") != MODEL_IDS:
            raise ValueError(f"{protocol_name} must define all M0-M7 models.")
        if control.get("training_seeds") != [42, 43, 44]:
            raise ValueError(f"{protocol_name} must use seeds 42, 43, 44.")


def load_and_validate_checkpoints(
    entries: dict[str, dict],
    conditions: dict[str, dict],
    models: list[str],
    training_manifest_dir: Path,
    wild_frame: pd.DataFrame,
) -> tuple[dict, dict, dict]:
    checkpoint_info = {}
    audits = {}
    reference_data_config = None
    for model_id in models:
        training_manifest = training_manifest_dir / f"{model_id.lower()}_seed42.csv"
        if not training_manifest.is_file():
            raise FileNotFoundError(training_manifest)
        training_frame = read_manifest(training_manifest)
        audits[model_id] = training_overlap_audit(wild_frame, training_frame)
        condition = conditions[model_id]
        for protocol_name, entry in entries.items():
            control = entry["control"]
            if control is not None:
                expected_hash = control["manifest_sha256"][model_id]
                actual_hash = sha256_file(training_manifest)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"{protocol_name}/{model_id} training manifest mismatch."
                    )
            for seed in entry["seeds"]:
                if protocol_name == "basic":
                    checkpoint_file = basic_checkpoint_path(
                        entry["runs_root"], model_id, condition, seed
                    )
                else:
                    checkpoint_file = controlled_checkpoint_path(
                        entry["runs_root"],
                        model_id,
                        condition,
                        seed,
                        control["protocol"],
                    )
                if not checkpoint_file.is_file():
                    raise FileNotFoundError(checkpoint_file)
                checkpoint = torch.load(
                    checkpoint_file, map_location="cpu", weights_only=False
                )
                if protocol_name == "basic":
                    info = validate_basic_checkpoint(
                        checkpoint,
                        checkpoint_file,
                        training_manifest,
                        model_id,
                        condition,
                    )
                    if int(checkpoint["config"].get("seed")) != seed:
                        raise ValueError(f"Basic checkpoint seed mismatch: {checkpoint_file}")
                    info["path"] = str(checkpoint_file)
                    info["sha256"] = sha256_file(checkpoint_file)
                else:
                    info = validate_controlled_checkpoint(
                        checkpoint,
                        checkpoint_file,
                        training_manifest,
                        model_id,
                        condition,
                        seed,
                        control,
                    )
                data_config = checkpoint["config"]["data_config"]
                if reference_data_config is None:
                    reference_data_config = data_config
                elif data_config != reference_data_config:
                    raise ValueError("Selected checkpoints use different Xception inputs.")
                checkpoint_info[(protocol_name, model_id, seed)] = info
                del checkpoint
    if reference_data_config is None:
        raise RuntimeError("No checkpoints were selected.")
    return checkpoint_info, audits, reference_data_config


def build_loaders(
    frame: pd.DataFrame,
    data_root: Path,
    data_config: dict,
    batch_size: int,
    workers: int,
) -> dict[str, DataLoader]:
    _, transform = letterbox_transforms(data_config)
    transforms = {
        "native_letterbox299": (
            transform,
            LETTERBOX_NAME,
        )
    }
    loaders = {}
    for name, (transform, preprocessing_name) in transforms.items():
        dataset = EvaluationDataset(
            frame,
            data_root,
            transform,
            preprocessing_name,
            data_config["input_size"][1],
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    return loaders


def result_is_complete(
    result_dir: Path,
    expected_samples: list[str],
    expected_sequences: int,
    checkpoint_hash: str,
    manifest_hash: str,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    metrics_path = result_dir / "metrics.json"
    frame_path = result_dir / "frame_predictions.csv"
    sequence_path = result_dir / "sequence_predictions.csv"
    existing = [path.exists() for path in (metrics_path, frame_path, sequence_path)]
    if not any(existing):
        return False, pd.DataFrame(), pd.DataFrame()
    if not all(existing):
        raise RuntimeError(f"Incomplete WildDeepfake result: {result_dir}")
    metrics = load_json(metrics_path)
    if metrics["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError(f"Checkpoint changed for existing result: {result_dir}")
    if metrics["test_manifest_sha256"] != manifest_hash:
        raise RuntimeError(f"Manifest changed for existing result: {result_dir}")
    frame = pd.read_csv(frame_path)
    sequence = pd.read_csv(sequence_path)
    if frame["sample_id"].tolist() != expected_samples:
        raise RuntimeError(f"Frame prediction order changed: {frame_path}")
    if len(sequence) != expected_sequences:
        raise RuntimeError(f"Incomplete sequence predictions: {sequence_path}")
    return True, frame, sequence


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_one(
    protocol_name: str,
    model_id: str,
    seed: int,
    entry: dict,
    checkpoint: dict,
    checkpoint_meta: dict,
    loader: DataLoader,
    test_frame: pd.DataFrame,
    manifest_path: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir = output_root / protocol_name / model_id.lower() / f"seed{seed}"
    complete, frame_prediction, sequence_prediction = result_is_complete(
        result_dir,
        test_frame["sample_id"].tolist(),
        test_frame["sequence_uid"].nunique(),
        checkpoint_meta["sha256"],
        sha256_file(manifest_path),
    )
    if complete:
        print(f"Skipping completed: {protocol_name}/{model_id}/seed{seed}", flush=True)
        return frame_prediction, sequence_prediction

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
    _, labels, _, probabilities = evaluate_with_predictions(
        model,
        loader,
        test_frame,
        criterion,
        device,
        progress_desc=f"{protocol_name} {model_id} seed{seed}",
        persistent_progress=True,
    )
    frame_prediction = make_frame_predictions(test_frame, labels, probabilities)
    sequence_prediction = make_sequence_predictions(frame_prediction)
    frame_metrics = metrics_from_arrays(
        frame_prediction["true_class"].to_numpy(int),
        frame_prediction["fake_probability"].to_numpy(float),
    )
    sequence_metrics = metrics_from_arrays(
        sequence_prediction["true_class"].to_numpy(int),
        sequence_prediction["mean_fake_probability"].to_numpy(float),
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(frame_prediction, result_dir / "frame_predictions.csv")
    atomic_csv(sequence_prediction, result_dir / "sequence_predictions.csv")
    write_json(
        result_dir / "metrics.json",
        {
            "dataset": "WildDeepfake",
            "protocol": protocol_name,
            "model": model_id,
            "training_seed": seed,
            "evaluation_preprocessing": entry["preprocessing"],
            "checkpoint": checkpoint_meta["path"],
            "checkpoint_sha256": checkpoint_meta["sha256"],
            "checkpoint_epoch": checkpoint_meta["epoch"],
            "best_validation_auc": checkpoint_meta["best_validation_auc"],
            "test_manifest": str(manifest_path),
            "test_manifest_sha256": sha256_file(manifest_path),
            "frame_metrics": frame_metrics,
            "sequence_metrics": sequence_metrics,
            "sequence_score": "mean fake probability over 16 selected frames",
            "decision_threshold": 0.5,
            "test_used_for_tuning": False,
            "versions": versions(),
        },
    )
    del model, criterion, weights
    torch.cuda.empty_cache()
    return frame_prediction, sequence_prediction


def summary_row(
    protocol_name: str,
    model_id: str,
    condition: dict,
    seed: int,
    preprocessing: str,
    frame_prediction: pd.DataFrame,
    sequence_prediction: pd.DataFrame,
) -> dict:
    frame_metrics = metrics_from_arrays(
        frame_prediction["true_class"].to_numpy(int),
        frame_prediction["fake_probability"].to_numpy(float),
    )
    sequence_metrics = metrics_from_arrays(
        sequence_prediction["true_class"].to_numpy(int),
        sequence_prediction["mean_fake_probability"].to_numpy(float),
    )
    row = {
        "protocol": protocol_name,
        "model": model_id,
        "training_condition": condition["name"],
        "training_seed": seed,
        "evaluation_preprocessing": preprocessing,
        "frame_images": int(len(frame_prediction)),
        "sequences": int(len(sequence_prediction)),
    }
    row.update({f"frame_{key}": value for key, value in frame_metrics.items()})
    row.update({f"sequence_{key}": value for key, value in sequence_metrics.items()})
    return row


def aggregate_seeds(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sequence_accuracy",
        "sequence_precision",
        "sequence_fake_recall",
        "sequence_real_recall",
        "sequence_real_fpr",
        "sequence_f1",
        "sequence_roc_auc",
        "sequence_auprc",
        "sequence_real_mean_fake_score",
        "sequence_fake_mean_fake_score",
        "frame_roc_auc",
        "frame_auprc",
        "frame_real_fpr",
    ]
    rows = []
    group_columns = [
        "protocol", "model", "training_condition", "evaluation_preprocessing"
    ]
    for keys, group in summary.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["training_seeds"] = "|".join(map(str, sorted(group["training_seed"])))
        row["seed_runs"] = int(len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_controlled(summary: pd.DataFrame) -> pd.DataFrame:
    controlled = summary[summary["protocol"].isin(["fixed_q95", "mixed_jpeg"])]
    rows = []
    metrics = [
        "sequence_roc_auc",
        "sequence_auprc",
        "sequence_real_fpr",
        "sequence_fake_recall",
        "sequence_f1",
        "frame_roc_auc",
    ]
    for (model_id, seed), group in controlled.groupby(
        ["model", "training_seed"], sort=True
    ):
        indexed = group.set_index("protocol")
        if not {"fixed_q95", "mixed_jpeg"}.issubset(indexed.index):
            continue
        row = {"model": model_id, "training_seed": int(seed)}
        for metric in metrics:
            fixed = float(indexed.loc["fixed_q95", metric])
            mixed = float(indexed.loc["mixed_jpeg", metric])
            row[f"fixed_q95_{metric}"] = fixed
            row[f"mixed_jpeg_{metric}"] = mixed
            row[f"mixed_minus_fixed_{metric}"] = mixed - fixed
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(seed_summary: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# WildDeepfake M0-M7 Evaluation",
        "",
        "> Primary results are sequence-level. Each source sequence contributes "
        "one score: the mean fake probability over its 16 fixed frames. The "
        "decision threshold is fixed at 0.5 and was not tuned on WildDeepfake.",
        "",
        "| Protocol | Model | Seeds | Sequence AUC | Sequence AUPRC | Real FPR | Fake recall | Frame AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary.sort_values(["protocol", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.model} | {row.training_seeds} | "
            f"{row.sequence_roc_auc_mean:.4f} ± {row.sequence_roc_auc_std:.4f} | "
            f"{row.sequence_auprc_mean:.4f} ± {row.sequence_auprc_std:.4f} | "
            f"{row.sequence_real_fpr_mean:.4f} ± {row.sequence_real_fpr_std:.4f} | "
            f"{row.sequence_fake_recall_mean:.4f} ± {row.sequence_fake_recall_std:.4f} | "
            f"{row.frame_roc_auc_mean:.4f} ± {row.frame_roc_auc_std:.4f} |"
        )
    if not comparison.empty:
        aggregate = comparison.groupby("model").agg(
            seeds=("training_seed", "count"),
            auc_delta_mean=("mixed_minus_fixed_sequence_roc_auc", "mean"),
            auc_delta_std=("mixed_minus_fixed_sequence_roc_auc", "std"),
            fpr_delta_mean=("mixed_minus_fixed_sequence_real_fpr", "mean"),
            fpr_delta_std=("mixed_minus_fixed_sequence_real_fpr", "std"),
        ).reset_index().fillna(0.0)
        lines.extend(
            [
                "",
                "## Mixed-JPEG minus fixed-Q95",
                "",
                "| Model | Seeds | Sequence AUC delta | Real FPR delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in aggregate.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.seeds} | "
                f"{row.auc_delta_mean:+.4f} ± {row.auc_delta_std:.4f} | "
                f"{row.fpr_delta_mean:+.4f} ± {row.fpr_delta_std:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Basic, Fixed-Q95, and Mixed-JPEG share the exact same native-image Letterbox-299 evaluation path. No additional JPEG re-encoding is applied to WildDeepfake.",
            "- The checkpoints were trained under different preprocessing protocols. This evaluation asks how those completed systems transfer to the same untouched external image distribution.",
            "- WildDeepfake is an external in-the-wild benchmark. It was not used for training, checkpoint selection, or threshold tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "manifest": args.manifest.resolve(),
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "basic_runs_root": args.basic_runs_root.resolve(),
        "q95_runs_root": args.q95_runs_root.resolve(),
        "mixed_runs_root": args.mixed_runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "q95_config": args.q95_config.resolve(),
        "mixed_config": args.mixed_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    if len(args.protocols) != len(set(args.protocols)):
        raise ValueError("Protocols must be unique.")
    if len(args.models) != len(set(args.models)):
        raise ValueError("Models must be unique.")

    protocol = load_json(paths["protocol_config"])
    validate_protocol_config(protocol)
    q95_control = load_json(paths["q95_config"])
    mixed_control = load_json(paths["mixed_config"])
    validate_controls(q95_control, mixed_control)
    conditions = load_conditions(paths["conditions_config"])
    test_frame = validate_manifest(paths["manifest"], paths["data_root"], protocol)
    entries = protocol_entries(args, protocol, q95_control, mixed_control)
    checkpoints, audits, data_config = load_and_validate_checkpoints(
        entries,
        conditions,
        args.models,
        paths["training_manifest_dir"],
        test_frame,
    )

    identity = {
        "protocol": protocol["protocol"],
        "protocol_config_sha256": sha256_file(paths["protocol_config"]),
        "conditions_config_sha256": sha256_file(paths["conditions_config"]),
        "q95_config_sha256": sha256_file(paths["q95_config"]),
        "mixed_config_sha256": sha256_file(paths["mixed_config"]),
        "test_manifest_sha256": sha256_file(paths["manifest"]),
        "evaluation_preprocessing": "native_letterbox299",
        "selected_protocols": args.protocols,
        "selected_models": args.models,
        "checkpoints": {
            f"{name}/{model}/seed{seed}": {
                "sha256": info["sha256"],
                "epoch": info["epoch"],
            }
            for (name, model, seed), info in checkpoints.items()
        },
    }
    output_root = paths["output_root"]
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Non-empty output has no identity: {output_root}")
        if load_json(identity_path) != identity:
            raise RuntimeError("Existing WildDeepfake output belongs to another run.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required.")
    device = torch.device("cuda")
    loaders = build_loaders(
        test_frame,
        paths["data_root"],
        data_config,
        args.batch_size,
        args.workers,
    )
    summary_rows = []
    for protocol_name, entry in entries.items():
        loader = loaders[entry["preprocessing"]]
        for model_id in args.models:
            for seed in entry["seeds"]:
                info = checkpoints[(protocol_name, model_id, seed)]
                checkpoint = torch.load(
                    info["path"], map_location=device, weights_only=False
                )
                print(
                    f"\n=== {protocol_name} / {model_id} / seed {seed} ===",
                    flush=True,
                )
                frame_prediction, sequence_prediction = evaluate_one(
                    protocol_name,
                    model_id,
                    seed,
                    entry,
                    checkpoint,
                    info,
                    loader,
                    test_frame,
                    paths["manifest"],
                    output_root,
                    device,
                )
                summary_rows.append(
                    summary_row(
                        protocol_name,
                        model_id,
                        conditions[model_id],
                        seed,
                        entry["preprocessing"],
                        frame_prediction,
                        sequence_prediction,
                    )
                )
                del checkpoint
                torch.cuda.empty_cache()

    run_summary = pd.DataFrame(summary_rows)
    seed_summary = aggregate_seeds(run_summary)
    comparison = compare_controlled(run_summary)
    atomic_csv(run_summary, output_root / "model_seed_summary.csv")
    atomic_csv(seed_summary, output_root / "seed_aggregate_summary.csv")
    atomic_csv(comparison, output_root / "controlled_protocol_comparison.csv")
    (output_root / "REPORT.md").write_text(
        build_report(seed_summary, comparison), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "WildDeepfake",
            "protocol": protocol["protocol"],
            "selected_model_protocols": args.protocols,
            "selected_models": args.models,
            "test_images": int(len(test_frame)),
            "test_sequences": int(test_frame["sequence_uid"].nunique()),
            "frames_per_sequence": 16,
            "primary_unit": "sequence",
            "sequence_score": "mean fake probability over 16 selected frames",
            "decision_threshold": 0.5,
            "threshold_tuned_on_wilddeepfake": False,
            "shared_test_preprocessing": (
                "RGB decode -> native-image Letterbox 299 -> Tensor -> "
                "Normalize; no JPEG re-encoding"
            ),
            "training_overlap_audits": audits,
            "source_images_modified": False,
            "versions": versions(),
        },
    )
    print("\n=== WildDeepfake sequence-level summary ===")
    display_columns = [
        "protocol",
        "model",
        "training_seeds",
        "sequence_roc_auc_mean",
        "sequence_auprc_mean",
        "sequence_real_fpr_mean",
        "sequence_fake_recall_mean",
    ]
    print(seed_summary[display_columns].to_string(index=False))
    print("\nSaved to:", output_root)


if __name__ == "__main__":
    main()
