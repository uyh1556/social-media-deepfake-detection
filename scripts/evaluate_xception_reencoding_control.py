#!/usr/bin/env python3
"""Evaluate controlled-reencoding M5/M7 checkpoints across fixed seeds."""

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

from evaluate_checkpoint import (
    EvaluationDataset,
    evaluate_with_predictions,
)
from evaluate_xception_family_coverage import (
    cross_split_audit,
    load_conditions,
    read_manifest,
    validate_test_manifest,
)
from train_baseline import LABEL_MAP, sha256_file, write_json
from xception_preprocessing import (
    CONTROLLED_REENCODE_NAME,
    LETTERBOX_NAME,
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
    letterbox_transforms,
)


FIXED_Q95_PROTOCOL = "family_coverage_reencoding_control_v1"
MIXED_JPEG_PROTOCOL = "family_coverage_jpeg_mixed_v1"
SUPPORTED_PROTOCOLS = {FIXED_Q95_PROTOCOL, MIXED_JPEG_PROTOCOL}
MODEL_IDS = ["M5", "M7"]
PROTECTED_UNSEEN = {
    "FaceDancer",
    "InSwapper",
    "SadTalker",
    "HyperReenact",
    "StyleGAN-XL",
    "PixArt-alpha",
}
FFPP_METHODS = {"Deepfakes", "Face2Face"}
EXPECTED_TEST_IMAGES = 30_000
EXPECTED_IMAGES_PER_METHOD = 2_000
JPEG_REFERENCE_CONDITION = "canonical_256_jpeg_q95"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=(
            project_root
            / "configs/family_coverage_reencoding_control_v1/protocol.json"
        ),
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--evaluation-conditions",
        nargs="+",
        default=["canonical_256_jpeg_q95"],
        help=(
            "Conditions from the control config to evaluate. The default is "
            "the primary train/validation/test-matched Q95 condition."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_control(
    path: Path,
    seeds: list[int],
    selected_conditions: list[str],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    control = json.loads(path.read_text(encoding="utf-8"))
    if control.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unexpected controlled re-encoding protocol.")
    if control.get("models") != MODEL_IDS:
        raise ValueError(f"Control models must be {MODEL_IDS}.")
    allowed_seeds = set(control.get("training_seeds", []))
    if not seeds or set(seeds) - allowed_seeds:
        raise ValueError(
            f"Training seeds must be selected from {sorted(allowed_seeds)}."
        )
    conditions = control.get("evaluation_conditions", [])
    names = [item.get("name") for item in conditions]
    expected = [
        "native_original",
        "canonical_256_png",
        "canonical_256_jpeg_q95",
        "canonical_256_jpeg_q90",
        "canonical_256_jpeg_q75",
    ]
    if names != expected:
        raise ValueError(f"Unexpected evaluation conditions: {names}")
    if not selected_conditions or len(selected_conditions) != len(
        set(selected_conditions)
    ):
        raise ValueError("Evaluation conditions must be non-empty and unique.")
    unknown = set(selected_conditions) - set(names)
    if unknown:
        raise ValueError(
            f"Unknown evaluation conditions: {sorted(unknown)}"
        )
    if sum(bool(item.get("primary")) for item in conditions) != 1:
        raise ValueError("Exactly one evaluation condition must be primary.")
    return control


def controlled_run_name(
    model_id: str,
    condition: dict,
    seed: int,
    protocol: str,
) -> str:
    if protocol == FIXED_Q95_PROTOCOL:
        suffix = "canonical256_jpegq95_letterbox299_control_v1"
    elif protocol == MIXED_JPEG_PROTOCOL:
        suffix = "canonical256_jpegmix75_80_85_90_95_letterbox299_v1"
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")
    return f"xception_{model_id.lower()}_{condition['name']}_{suffix}_seed{seed}"


def checkpoint_path(
    runs_root: Path,
    model_id: str,
    condition: dict,
    seed: int,
    protocol: str,
) -> Path:
    return (
        runs_root
        / controlled_run_name(model_id, condition, seed, protocol)
        / "best.pt"
    )


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_file: Path,
    training_manifest: Path,
    model_id: str,
    condition: dict,
    seed: int,
    control: dict,
) -> dict:
    config = checkpoint["config"]
    preprocessing = control["preprocessing"]
    if checkpoint_file.name != "best.pt":
        raise ValueError(f"Evaluation requires best.pt: {checkpoint_file}")
    if checkpoint["model_name"] not in {"xception", "legacy_xception"}:
        raise ValueError(f"Unexpected model: {checkpoint['model_name']}")
    protocol = control["protocol"]
    if config.get("experiment_family") != protocol:
        raise ValueError(f"Unexpected experiment family in {checkpoint_file}")
    if config.get("split_protocol") != protocol:
        raise ValueError(f"Unexpected split protocol in {checkpoint_file}")
    expected_preprocessing_name = (
        CONTROLLED_REENCODE_NAME
        if protocol == FIXED_Q95_PROTOCOL
        else MIXED_JPEG_REENCODE_NAME
    )
    if config.get("preprocessing_name") != expected_preprocessing_name:
        raise ValueError(f"Unexpected preprocessing in {checkpoint_file}")
    expected_condition_name = f"{model_id.lower()}_{condition['name']}"
    if protocol == MIXED_JPEG_PROTOCOL:
        expected_condition_name += "_jpeg_mixed"
    if config.get("condition_name") != expected_condition_name:
        raise ValueError(f"Condition mismatch in {checkpoint_file}")
    if int(config.get("seed")) != seed:
        raise ValueError(f"Training seed mismatch in {checkpoint_file}")
    if config.get("manifest_sha256") != sha256_file(training_manifest):
        raise ValueError(f"Training manifest mismatch in {checkpoint_file}")
    saved = config.get("preprocessing", {})
    keys = [
        "canonical_size",
        "jpeg_subsampling",
        "jpeg_optimize",
        "jpeg_progressive",
    ]
    if protocol == FIXED_Q95_PROTOCOL:
        keys.append("jpeg_quality")
    else:
        keys.extend(
            [
                "train_jpeg_qualities",
                "train_jpeg_sampling",
                "validation_jpeg_quality",
            ]
        )
    for key in keys:
        if saved.get(key) != preprocessing.get(key):
            raise ValueError(
                f"Checkpoint preprocessing mismatch for {key}: "
                f"expected={preprocessing.get(key)}, actual={saved.get(key)}"
            )
    return {
        "path": str(checkpoint_file),
        "sha256": sha256_file(checkpoint_file),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
    }


def build_transform(data_config: dict, condition: dict, control: dict):
    if condition["name"] == "native_original":
        _, evaluation = letterbox_transforms(data_config)
        return evaluation, LETTERBOX_NAME
    preprocessing = control["preprocessing"]
    return (
        canonical_reencode_transform(
            data_config,
            canonical_size=condition["canonical_size"],
            jpeg_quality=condition["jpeg_quality"],
            jpeg_subsampling=preprocessing["jpeg_subsampling"],
            jpeg_optimize=preprocessing["jpeg_optimize"],
            jpeg_progressive=preprocessing["jpeg_progressive"],
        ),
        control["preprocessing"]["name"],
    )


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


def summarize_prediction(
    model_id: str,
    training_condition: dict,
    seed: int,
    evaluation_condition: dict,
    prediction: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    labels = prediction["true_class"].to_numpy(int)
    predicted = prediction["predicted_class"].to_numpy(int)
    scores = prediction["fake_probability"].to_numpy(float)
    overall = metrics_from_arrays(labels, predicted, scores)
    real = prediction[prediction["label"] == "real"]
    trained = set(training_condition["seen_fake_methods"])
    method_rows = []
    for method, fake in prediction[prediction["label"] == "fake"].groupby(
        "method", sort=True
    ):
        pair = pd.concat([real, fake], ignore_index=True)
        metrics = metrics_from_arrays(
            pair["true_class"].to_numpy(int),
            pair["predicted_class"].to_numpy(int),
            pair["fake_probability"].to_numpy(float),
        )
        method_rows.append(
            {
                "model": model_id,
                "training_condition": training_condition["name"],
                "training_seed": seed,
                "evaluation_condition": evaluation_condition["name"],
                "primary_condition": bool(
                    evaluation_condition.get("primary")
                ),
                "method": method,
                "family": fake["family"].iloc[0],
                "dataset_role": fake["role"].iloc[0],
                "relative_status": (
                    "trained_method" if method in trained else "held_out_method"
                ),
                "images_fake": int(len(fake)),
                "images_real": int(len(real)),
                **metrics,
            }
        )
    method_frame = pd.DataFrame(method_rows)

    def macro(mask: pd.Series) -> float:
        values = method_frame.loc[mask, "roc_auc"]
        return float(values.mean()) if len(values) else float("nan")

    protected = method_frame["method"].isin(PROTECTED_UNSEEN)
    pixart = method_frame["method"] == "PixArt-alpha"
    model_row = {
        "model": model_id,
        "training_condition": training_condition["name"],
        "training_seed": seed,
        "evaluation_condition": evaluation_condition["name"],
        "primary_condition": bool(evaluation_condition.get("primary")),
        "test_images": int(len(prediction)),
        **overall,
        "all_methods_macro_auc": float(method_frame["roc_auc"].mean()),
        "trained_methods_macro_auc": macro(
            method_frame["relative_status"] == "trained_method"
        ),
        "protected_unseen_macro_auc": macro(protected),
        "protected_unseen_excluding_pixart_macro_auc": macro(
            protected & ~pixart
        ),
        "pixart_auc": macro(pixart),
        "ffpp_methods_macro_auc": macro(
            method_frame["method"].isin(FFPP_METHODS)
        ),
    }
    return model_row, method_rows


def aggregate_seeds(model_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "real_fpr",
        "fake_recall",
        "roc_auc",
        "auprc",
        "all_methods_macro_auc",
        "trained_methods_macro_auc",
        "protected_unseen_macro_auc",
        "protected_unseen_excluding_pixart_macro_auc",
        "pixart_auc",
        "ffpp_methods_macro_auc",
    ]
    rows = []
    for keys, group in model_summary.groupby(
        ["model", "training_condition", "evaluation_condition"],
        sort=True,
    ):
        row = {
            "model": keys[0],
            "training_condition": keys[1],
            "evaluation_condition": keys[2],
            "primary_condition": bool(group["primary_condition"].iloc[0]),
            "training_seeds": "|".join(map(str, sorted(group["training_seed"]))),
            "seed_runs": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_method_seeds(method_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "real_fpr",
        "fake_recall",
        "roc_auc",
        "auprc",
        "real_mean_fake_score",
        "fake_mean_fake_score",
    ]
    group_columns = [
        "model",
        "training_condition",
        "evaluation_condition",
        "primary_condition",
        "method",
        "family",
        "dataset_role",
        "relative_status",
        "images_fake",
        "images_real",
    ]
    rows = []
    for keys, group in method_summary.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["training_seeds"] = "|".join(
            map(str, sorted(group["training_seed"]))
        )
        row["seed_runs"] = int(len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = (
                float(group[metric].std(ddof=1))
                if len(group) > 1
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(model_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "protected_unseen_macro_auc",
        "protected_unseen_excluding_pixart_macro_auc",
        "pixart_auc",
        "all_methods_macro_auc",
        "real_fpr",
    ]
    rows = []
    for (seed, evaluation_condition), group in model_summary.groupby(
        ["training_seed", "evaluation_condition"], sort=True
    ):
        by_model = group.set_index("model")
        if set(by_model.index) != set(MODEL_IDS):
            raise RuntimeError(
                f"M5/M7 pairing is incomplete for seed={seed}, "
                f"condition={evaluation_condition}"
            )
        row = {
            "training_seed": int(seed),
            "evaluation_condition": evaluation_condition,
            "primary_condition": bool(group["primary_condition"].iloc[0]),
        }
        for metric in metrics:
            row[f"m5_{metric}"] = float(by_model.loc["M5", metric])
            row[f"m7_{metric}"] = float(by_model.loc["M7", metric])
            row[f"m5_minus_m7_{metric}"] = float(
                by_model.loc["M5", metric] - by_model.loc["M7", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_jpeg_conditions(model_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "real_fpr",
        "fake_recall",
        "roc_auc",
        "auprc",
        "all_methods_macro_auc",
        "protected_unseen_macro_auc",
        "protected_unseen_excluding_pixart_macro_auc",
        "pixart_auc",
        "real_mean_fake_score",
        "fake_mean_fake_score",
    ]
    if JPEG_REFERENCE_CONDITION not in set(
        model_summary["evaluation_condition"]
    ):
        return pd.DataFrame()
    rows = []
    for (model_id, seed), group in model_summary.groupby(
        ["model", "training_seed"], sort=True
    ):
        by_condition = group.set_index("evaluation_condition")
        reference = by_condition.loc[JPEG_REFERENCE_CONDITION]
        for target_name, target in by_condition.iterrows():
            if target_name == JPEG_REFERENCE_CONDITION:
                continue
            row = {
                "model": model_id,
                "training_seed": int(seed),
                "reference_condition": JPEG_REFERENCE_CONDITION,
                "target_condition": target_name,
            }
            for metric in metrics:
                reference_value = float(reference[metric])
                target_value = float(target[metric])
                row[f"reference_{metric}"] = reference_value
                row[f"target_{metric}"] = target_value
                row[f"delta_{metric}"] = target_value - reference_value
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_jpeg_condition_deltas(delta_summary: pd.DataFrame) -> pd.DataFrame:
    if delta_summary.empty:
        return pd.DataFrame()
    delta_columns = [
        column
        for column in delta_summary.columns
        if column.startswith("delta_")
    ]
    rows = []
    for (model_id, target_name), group in delta_summary.groupby(
        ["model", "target_condition"], sort=True
    ):
        row = {
            "model": model_id,
            "reference_condition": JPEG_REFERENCE_CONDITION,
            "target_condition": target_name,
            "training_seeds": "|".join(
                map(str, sorted(group["training_seed"]))
            ),
            "seed_runs": int(len(group)),
        }
        for column in delta_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1))
                if len(group) > 1
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    seed_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    jpeg_delta_aggregate: pd.DataFrame,
    control: dict,
) -> str:
    primary = seed_summary[seed_summary["primary_condition"]].sort_values(
        "model"
    )
    displayed = seed_summary.sort_values(
        ["evaluation_condition", "model"]
    )
    displayed_comparison = comparison.sort_values(
        ["evaluation_condition", "training_seed"]
    )
    if primary.empty:
        section_title = "Selected sensitivity conditions"
    elif seed_summary["evaluation_condition"].nunique() > 1:
        section_title = "Q95 baseline and JPEG sensitivity conditions"
    else:
        section_title = "Primary controlled condition"
    if control["protocol"] == MIXED_JPEG_PROTOCOL:
        title = "# M5/M7 Mixed-JPEG Augmentation Evaluation"
        training_note = (
            "> Checkpoints were trained with RGB decode, Letterbox 256, "
            "uniformly sampled JPEG Q75/Q80/Q85/Q90/Q95 4:2:0, "
            "Letterbox 299, and fixed normalization. Validation and model "
            "selection remained fixed at Q95; no test-time tuning was "
            "performed."
        )
    else:
        title = "# M5/M7 Controlled Re-encoding Evaluation"
        training_note = (
            "> Checkpoints were trained with RGB decode, Letterbox 256, "
            "JPEG Q95 4:2:0, Letterbox 299, and fixed normalization. The "
            "table names the evaluation-time condition; no test-time "
            "tuning was performed."
        )
    lines = [
        title,
        "",
        training_note,
        "",
        f"## {section_title}",
        "",
        (
            "| Condition | Model | Seeds | Real FPR | All-method AUC | "
            "Protected unseen AUC | Protected unseen without PixArt | "
            "PixArt AUC |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in displayed.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.model} | "
            f"{row.training_seeds} | "
            f"{row.real_fpr_mean:.4f} ± {row.real_fpr_std:.4f} | "
            f"{row.all_methods_macro_auc_mean:.4f} ± "
            f"{row.all_methods_macro_auc_std:.4f} | "
            f"{row.protected_unseen_macro_auc_mean:.4f} ± "
            f"{row.protected_unseen_macro_auc_std:.4f} | "
            f"{row.protected_unseen_excluding_pixart_macro_auc_mean:.4f} ± "
            f"{row.protected_unseen_excluding_pixart_macro_auc_std:.4f} | "
            f"{row.pixart_auc_mean:.4f} ± {row.pixart_auc_std:.4f} |"
        )
    lines.extend(
        [
            "",
            "## M5 minus M7 by training seed",
            "",
            (
                "| Condition | Seed | Protected unseen delta | "
                "Delta without PixArt | PixArt delta |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in displayed_comparison.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.training_seed} | "
            f"{row.m5_minus_m7_protected_unseen_macro_auc:+.4f} | "
            f"{row.m5_minus_m7_protected_unseen_excluding_pixart_macro_auc:+.4f} | "
            f"{row.m5_minus_m7_pixart_auc:+.4f} |"
        )
    if not jpeg_delta_aggregate.empty:
        lines.extend(
            [
                "",
                "## Change from matched JPEG Q95",
                "",
                (
                    "| Model | Target | Real fake-score delta | Real FPR "
                    "delta | All-method AUC delta | Protected unseen AUC "
                    "delta | Delta without PixArt | PixArt AUC delta |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in jpeg_delta_aggregate.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.target_condition} | "
                f"{row.delta_real_mean_fake_score_mean:+.4f} ± "
                f"{row.delta_real_mean_fake_score_std:.4f} | "
                f"{row.delta_real_fpr_mean:+.4f} ± "
                f"{row.delta_real_fpr_std:.4f} | "
                f"{row.delta_all_methods_macro_auc_mean:+.4f} ± "
                f"{row.delta_all_methods_macro_auc_std:.4f} | "
                f"{row.delta_protected_unseen_macro_auc_mean:+.4f} ± "
                f"{row.delta_protected_unseen_macro_auc_std:.4f} | "
                f"{row.delta_protected_unseen_excluding_pixart_macro_auc_mean:+.4f} ± "
                f"{row.delta_protected_unseen_excluding_pixart_macro_auc_std:.4f} | "
                f"{row.delta_pixart_auc_mean:+.4f} ± "
                f"{row.delta_pixart_auc_std:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The primary comparison is the fixed canonical-256/JPEG-Q95 "
                "condition. Native-resolution and non-Q95 rows are "
                "sensitivity analyses. Canonicalization controls the "
                "model-visible size, encoder, and subsequent preprocessing "
                "path, but it cannot erase every artifact inherited from "
                "each generator or upstream video codec."
            ),
            "",
        ]
    )
    return "\n".join(lines)


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


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "runs_root": args.runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "control_config": args.control_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    control = load_control(
        paths["control_config"],
        args.training_seeds,
        args.evaluation_conditions,
    )
    evaluation_conditions = [
        item
        for item in control["evaluation_conditions"]
        if item["name"] in args.evaluation_conditions
    ]
    conditions = load_conditions(paths["conditions_config"])
    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise ValueError("Unexpected test image count.")
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

    test_hash = sha256_file(paths["test_manifest"])
    if test_hash != control["test_manifest_sha256"]:
        raise ValueError(
            "Controlled test manifest hash mismatch: "
            f"expected={control['test_manifest_sha256']}, actual={test_hash}"
        )
    checkpoints = {}
    audits = {}
    reference_data_config = None
    identity = {
        "protocol": control["protocol"],
        "control_config_sha256": sha256_file(paths["control_config"]),
        "test_manifest_sha256": test_hash,
        "training_seeds": args.training_seeds,
        "evaluation_conditions": args.evaluation_conditions,
        "models": {},
    }
    for model_id in MODEL_IDS:
        condition = conditions[model_id]
        training_manifest = (
            paths["training_manifest_dir"] / f"{model_id.lower()}_seed42.csv"
        )
        if not training_manifest.is_file():
            raise FileNotFoundError(training_manifest)
        expected_training_hash = control["manifest_sha256"][model_id]
        actual_training_hash = sha256_file(training_manifest)
        if actual_training_hash != expected_training_hash:
            raise ValueError(
                f"Controlled {model_id} manifest hash mismatch: "
                f"expected={expected_training_hash}, "
                f"actual={actual_training_hash}"
            )
        development = read_manifest(training_manifest)
        audits[model_id] = cross_split_audit(development, test_frame)
        identity["models"][model_id] = {
            "training_manifest_sha256": sha256_file(training_manifest),
            "checkpoints": {},
        }
        for seed in args.training_seeds:
            checkpoint_file = checkpoint_path(
                paths["runs_root"],
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
            checkpoint_info = validate_checkpoint(
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
                raise ValueError("Controlled checkpoints use different inputs.")
            checkpoints[(model_id, seed)] = checkpoint_info
            identity["models"][model_id]["checkpoints"][str(seed)] = {
                "sha256": checkpoint_info["sha256"],
                "epoch": checkpoint_info["epoch"],
            }
            del checkpoint

    output_root = paths["output_root"]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required.")
    loaders = {}
    for evaluation_condition in evaluation_conditions:
        transform, preprocessing_name = build_transform(
            reference_data_config, evaluation_condition, control
        )
        dataset = EvaluationDataset(
            test_frame,
            paths["data_root"],
            transform,
            preprocessing_name,
            reference_data_config["input_size"][1],
        )
        loaders[evaluation_condition["name"]] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    model_rows = []
    all_method_rows = []
    for model_id in MODEL_IDS:
        training_condition = conditions[model_id]
        training_manifest = (
            paths["training_manifest_dir"] / f"{model_id.lower()}_seed42.csv"
        )
        for seed in args.training_seeds:
            checkpoint_file = Path(checkpoints[(model_id, seed)]["path"])
            checkpoint = torch.load(
                checkpoint_file, map_location=device, weights_only=False
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

            for evaluation_condition in evaluation_conditions:
                condition_name = evaluation_condition["name"]
                result_dir = (
                    output_root
                    / model_id.lower()
                    / f"seed{seed}"
                    / condition_name
                )
                metrics_path = result_dir / "metrics.json"
                predictions_path = result_dir / "predictions.csv"
                if metrics_path.is_file() and predictions_path.is_file():
                    result = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if result["test_manifest_sha256"] != test_hash:
                        raise RuntimeError(
                            f"Existing result used another test manifest: {result_dir}"
                        )
                    if result["checkpoint_sha256"] != checkpoints[
                        (model_id, seed)
                    ]["sha256"]:
                        raise RuntimeError(
                            f"Existing result used another checkpoint: {result_dir}"
                        )
                    prediction = pd.read_csv(predictions_path)
                    if len(prediction) != len(test_frame):
                        raise RuntimeError(
                            f"Incomplete predictions: {predictions_path}"
                        )
                    required_prediction_columns = {
                        "sample_id",
                        "source_path",
                        "label",
                        "method",
                        "family",
                        "role",
                        "group_id",
                        "video_id",
                        "true_class",
                        "predicted_class",
                        "fake_probability",
                        "correct",
                    }
                    missing_columns = (
                        required_prediction_columns - set(prediction.columns)
                    )
                    if missing_columns:
                        raise RuntimeError(
                            "Existing predictions are missing columns: "
                            f"{sorted(missing_columns)}"
                        )
                    if prediction["sample_id"].tolist() != test_frame[
                        "sample_id"
                    ].tolist():
                        raise RuntimeError(
                            f"Prediction order changed: {predictions_path}"
                        )
                    print(
                        f"Skipping completed: {model_id}/seed{seed}/"
                        f"{condition_name}",
                        flush=True,
                    )
                elif result_dir.exists() and any(result_dir.iterdir()):
                    raise FileExistsError(
                        f"Incomplete evaluation directory: {result_dir}"
                    )
                else:
                    print(
                        f"\n=== {model_id}/seed{seed}: {condition_name} ===",
                        flush=True,
                    )
                    metrics, labels, predictions, probabilities = (
                        evaluate_with_predictions(
                            model,
                            loaders[condition_name],
                            test_frame,
                            criterion,
                            device,
                            progress_desc=(
                                f"{model_id} seed{seed} {condition_name}"
                            ),
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
                            "model": model_id,
                            "training_seed": seed,
                            "evaluation_condition": evaluation_condition,
                            "checkpoint": str(checkpoint_file),
                            "checkpoint_sha256": checkpoints[
                                (model_id, seed)
                            ]["sha256"],
                            "training_manifest": str(training_manifest),
                            "training_manifest_sha256": sha256_file(
                                training_manifest
                            ),
                            "test_manifest": str(paths["test_manifest"]),
                            "test_manifest_sha256": test_hash,
                            "cross_split_audit": audits[model_id],
                            "metrics": metrics,
                            "decision_rule": "argmax logits (0.5 equivalent)",
                            "test_used_for_tuning": False,
                            "versions": versions(),
                        },
                    )
                model_row, method_rows = summarize_prediction(
                    model_id,
                    training_condition,
                    seed,
                    evaluation_condition,
                    prediction,
                )
                model_row.update(
                    {
                        "checkpoint_epoch": checkpoints[
                            (model_id, seed)
                        ]["epoch"],
                        "best_validation_auc": checkpoints[
                            (model_id, seed)
                        ]["best_validation_auc"],
                    }
                )
                model_rows.append(model_row)
                all_method_rows.extend(method_rows)

            del model, checkpoint, criterion, weight_tensor
            torch.cuda.empty_cache()

    model_summary = pd.DataFrame(model_rows)
    method_summary = pd.DataFrame(all_method_rows)
    seed_summary = aggregate_seeds(model_summary)
    method_seed_summary = aggregate_method_seeds(method_summary)
    comparison = compare_models(model_summary)
    jpeg_delta_summary = compare_jpeg_conditions(model_summary)
    jpeg_delta_aggregate = aggregate_jpeg_condition_deltas(
        jpeg_delta_summary
    )
    model_summary.to_csv(output_root / "model_seed_summary.csv", index=False)
    method_summary.to_csv(output_root / "method_seed_summary.csv", index=False)
    seed_summary.to_csv(output_root / "seed_aggregate_summary.csv", index=False)
    method_seed_summary.to_csv(
        output_root / "method_seed_aggregate_summary.csv", index=False
    )
    comparison.to_csv(output_root / "m5_m7_comparison.csv", index=False)
    if not jpeg_delta_summary.empty:
        jpeg_delta_summary.to_csv(
            output_root / "jpeg_condition_delta_summary.csv", index=False
        )
        jpeg_delta_aggregate.to_csv(
            output_root / "jpeg_condition_delta_seed_aggregate.csv",
            index=False,
        )
    (output_root / "REPORT.md").write_text(
        build_report(
            seed_summary, comparison, jpeg_delta_aggregate, control
        ),
        encoding="utf-8",
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": control["protocol"],
            "models": MODEL_IDS,
            "training_seeds": args.training_seeds,
            "evaluation_conditions": args.evaluation_conditions,
            "test_images": len(test_frame),
            "images_per_method": EXPECTED_IMAGES_PER_METHOD,
            "primary_evaluation_condition": (
                "canonical_256_jpeg_q95"
                if "canonical_256_jpeg_q95" in args.evaluation_conditions
                else None
            ),
            "protected_unseen_methods": sorted(PROTECTED_UNSEEN),
            "pixart_sensitivity_reporting": (
                "protected unseen Macro AUC is reported both with and "
                "without PixArt-alpha"
            ),
            "cross_split_audits": audits,
            "source_images_modified": False,
            "test_used_for_training_or_tuning": False,
            "versions": versions(),
        },
    )
    print("\n=== Primary controlled seed summary ===")
    print(seed_summary[seed_summary["primary_condition"]].to_string(index=False))
    print("\nSaved to:", output_root)


if __name__ == "__main__":
    main()
