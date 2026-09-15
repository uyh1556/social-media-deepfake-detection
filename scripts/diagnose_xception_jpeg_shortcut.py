#!/usr/bin/env python3
"""Evaluate frozen M0-M7 Xception models under matched JPEG round trips.

The canonical 30,000-image predictions are verified and reused from the frozen
all-method evaluation. JPEG transformations are applied in memory to every real
and fake image before the checkpoint's saved Letterbox preprocessing. Source
images, checkpoints, thresholds, and existing evaluation outputs are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from evaluate_xception_family_coverage import (
    MODEL_IDS,
    checkpoint_path,
    cross_split_audit,
    load_conditions,
    read_manifest,
    validate_checkpoint,
    validate_test_manifest,
)
from train_baseline import (
    LABEL_MAP,
    classification_metrics,
    manipulation_metrics,
    sha256_file,
    write_json,
)
from xception_preprocessing import (
    LETTERBOX_NAME,
    evaluation_transform_from_checkpoint,
)


EXPECTED_TEST_IMAGES = 30_000
EXPECTED_REAL_IMAGES = 2_000
EXPECTED_IMAGES_PER_FAKE_METHOD = 2_000


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--existing-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=project_root / "configs/jpeg_shortcut_diagnostic_v1/protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *map(str, parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def validate_protocol(config: dict) -> list[dict]:
    if config.get("protocol") != "jpeg_shortcut_all_models_v1":
        raise ValueError("Unexpected JPEG diagnostic protocol")
    if config.get("models") != MODEL_IDS:
        raise ValueError(f"Diagnostic models must be {MODEL_IDS}")
    if float(config.get("classification_threshold")) != 0.5:
        raise ValueError("The diagnostic is frozen to threshold 0.5")
    transformations = config.get("transformations", [])
    names = [item.get("name") for item in transformations]
    if names != ["original", "jpeg_q95", "jpeg_q90", "jpeg_q75"]:
        raise ValueError(f"Unexpected transformation order: {names}")
    if transformations[0].get("kind") != "identity":
        raise ValueError("The first condition must be the canonical original")
    for item in transformations[1:]:
        if item.get("kind") != "jpeg_round_trip":
            raise ValueError(f"Unexpected condition: {item}")
        if int(item["subsampling"]) != 2:
            raise ValueError("JPEG conditions must use frozen 4:2:0 subsampling")
    return transformations


def jpeg_round_trip(image: Image.Image, condition: dict) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=int(condition["quality"]),
        subsampling=int(condition["subsampling"]),
        optimize=bool(condition["optimize"]),
        progressive=bool(condition["progressive"]),
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


class JpegEvaluationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        condition: dict,
        model_transform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.condition = condition
        self.model_transform = model_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["source_path"]) as source:
            transformed = jpeg_round_trip(source.convert("RGB"), self.condition)
            tensor = self.model_transform(transformed)
        return {"image": tensor, "index": index}


@torch.no_grad()
def predict_condition(
    model,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    indices: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    for batch in tqdm(loader, desc=description, leave=True, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        indices.extend(batch["index"].tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities.extend(
            torch.softmax(logits, dim=1)[:, 1].float().cpu().tolist()
        )
    return indices, predictions, probabilities


def build_prediction_frame(
    test_frame: pd.DataFrame,
    condition_name: str,
    predictions: list[int],
    probabilities: list[float],
) -> pd.DataFrame:
    labels = test_frame["label"].map(LABEL_MAP).astype(int).to_numpy()
    predicted = np.asarray(predictions, dtype=int)
    return pd.DataFrame(
        {
            "sample_id": test_frame["sample_id"],
            "source_path": test_frame["source_path"],
            "label": test_frame["label"],
            "method": test_frame["method"],
            "family": test_frame["family"],
            "role": test_frame["role"],
            "group_id": test_frame["group_id"],
            "video_id": test_frame["video_id"],
            "condition": condition_name,
            "true_class": labels,
            "predicted_class": predicted,
            "fake_probability": probabilities,
            "correct": predicted == labels,
        }
    )


def validate_prediction_frame(
    frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    condition_name: str,
    path: Path,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "source_path",
        "label",
        "method",
        "group_id",
        "condition",
        "true_class",
        "predicted_class",
        "fake_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction columns missing in {path}: {sorted(missing)}")
    if len(frame) != len(test_frame):
        raise ValueError(f"Prediction row count mismatch in {path}")
    for column in ("sample_id", "source_path", "label", "method", "group_id"):
        if frame[column].astype(str).tolist() != test_frame[column].astype(str).tolist():
            raise RuntimeError(f"Prediction order mismatch for {column} in {path}")
    if set(frame["condition"].astype(str)) != {condition_name}:
        raise RuntimeError(f"Condition mismatch in {path}")
    for column in ("true_class", "predicted_class"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["fake_probability"] = pd.to_numeric(
        frame["fake_probability"], errors="raise"
    )
    if not frame["fake_probability"].between(0.0, 1.0).all():
        raise ValueError(f"Invalid probabilities in {path}")
    expected_labels = test_frame["label"].map(LABEL_MAP).astype(int).tolist()
    if frame["true_class"].tolist() != expected_labels:
        raise RuntimeError(f"Stored labels mismatch in {path}")
    if set(frame["predicted_class"]) - {0, 1}:
        raise ValueError(f"Invalid predicted classes in {path}")
    return frame


def load_canonical_predictions(
    path: Path,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    stored = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "source_path",
        "label",
        "method",
        "group_id",
        "video_id",
        "true_class",
        "predicted_class",
        "fake_probability",
    }
    missing = required - set(stored.columns)
    if missing:
        raise ValueError(f"Canonical prediction columns missing: {sorted(missing)}")
    if len(stored) != len(test_frame):
        raise ValueError(f"Canonical prediction row count mismatch in {path}")
    for column in ("source_path", "label", "method", "group_id", "video_id"):
        if stored[column].astype(str).tolist() != test_frame[column].astype(str).tolist():
            raise RuntimeError(f"Canonical order mismatch for {column} in {path}")
    result = test_frame[
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
    result["condition"] = "original"
    result["true_class"] = pd.to_numeric(stored["true_class"], errors="raise").astype(int)
    result["predicted_class"] = pd.to_numeric(
        stored["predicted_class"], errors="raise"
    ).astype(int)
    result["fake_probability"] = pd.to_numeric(
        stored["fake_probability"], errors="raise"
    )
    result["correct"] = result["true_class"] == result["predicted_class"]
    return validate_prediction_frame(result, test_frame, "original", path)


def metric_dict(frame: pd.DataFrame) -> dict:
    metrics = classification_metrics(
        frame["true_class"].to_numpy(int),
        frame["predicted_class"].to_numpy(int),
        frame["fake_probability"].to_numpy(float),
    )
    metrics["auprc"] = average_precision_score(
        frame["true_class"].to_numpy(int),
        frame["fake_probability"].to_numpy(float),
    )
    return metrics


def overall_row(model_id: str, condition: dict, frame: pd.DataFrame) -> dict:
    metrics = metric_dict(frame)
    confusion = metrics["confusion_matrix"]
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    return {
        "model": model_id,
        "condition": condition["name"],
        "jpeg_quality": condition.get("quality"),
        "severity": int(condition["severity"]),
        "images": len(frame),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "fake_recall": float(metrics["recall"]),
        "real_recall": float(true_negative / (true_negative + false_positive)),
        "real_fpr": float(false_positive / (true_negative + false_positive)),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "auprc": float(metrics["auprc"]),
        "real_mean_fake_score": float(
            frame.loc[frame["label"] == "real", "fake_probability"].mean()
        ),
        "real_median_fake_score": float(
            frame.loc[frame["label"] == "real", "fake_probability"].median()
        ),
        "fake_mean_fake_score": float(
            frame.loc[frame["label"] == "fake", "fake_probability"].mean()
        ),
        "fake_median_fake_score": float(
            frame.loc[frame["label"] == "fake", "fake_probability"].median()
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def method_rows(
    model_id: str,
    model_condition: dict,
    jpeg_condition: dict,
    frame: pd.DataFrame,
) -> list[dict]:
    trained = set(model_condition["seen_fake_methods"])
    metrics = manipulation_metrics(
        frame["true_class"].to_numpy(int),
        frame["predicted_class"].to_numpy(int),
        frame["fake_probability"].to_numpy(float),
        frame["method"].tolist(),
    )
    rows = []
    fake_frame = frame[frame["label"] == "fake"]
    for method, subset in fake_frame.groupby("method", sort=True):
        pair = metrics[f"original_vs_{method}"]
        pair_frame = frame[frame["method"].isin(["original", method])]
        pair_auprc = average_precision_score(
            pair_frame["true_class"].to_numpy(int),
            pair_frame["fake_probability"].to_numpy(float),
        )
        method_scores = subset["fake_probability"].to_numpy(float)
        confusion = pair["confusion_matrix"]
        true_negative, false_positive = confusion[0]
        false_negative, true_positive = confusion[1]
        rows.append(
            {
                "model": model_id,
                "training_condition": model_condition["name"],
                "jpeg_condition": jpeg_condition["name"],
                "jpeg_quality": jpeg_condition.get("quality"),
                "severity": int(jpeg_condition["severity"]),
                "method": method,
                "family": subset["family"].iloc[0],
                "dataset_role": subset["role"].iloc[0],
                "relative_status": (
                    "trained_method" if method in trained else "held_out_method"
                ),
                "images_fake": len(subset),
                "images_real": EXPECTED_REAL_IMAGES,
                "accuracy": float(pair["accuracy"]),
                "precision": float(pair["precision"]),
                "fake_recall": float(pair["recall"]),
                "real_recall": float(true_negative / (true_negative + false_positive)),
                "real_fpr": float(false_positive / (true_negative + false_positive)),
                "f1": float(pair["f1"]),
                "roc_auc": float(pair["roc_auc"]),
                "auprc": float(pair_auprc),
                "fake_mean_fake_score": float(method_scores.mean()),
                "fake_median_fake_score": float(np.median(method_scores)),
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
            }
        )
    return rows


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    grouped = frame.groupby("group_id", sort=True)
    clusters = np.asarray(
        [
            (
                float(group["score_delta"].sum()),
                float(group["positive_delta"].sum()),
                len(group),
            )
            for _, group in grouped
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    score_samples = np.empty(replicates, dtype=float)
    positive_samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        sample = clusters[selected]
        count = sample[:, 2].sum()
        score_samples[index] = sample[:, 0].sum() / count
        positive_samples[index] = sample[:, 1].sum() / count
    return {
        "mean_score_delta_ci_low": float(np.quantile(score_samples, 0.025)),
        "mean_score_delta_ci_high": float(np.quantile(score_samples, 0.975)),
        "positive_rate_delta_ci_low": float(
            np.quantile(positive_samples, 0.025)
        ),
        "positive_rate_delta_ci_high": float(
            np.quantile(positive_samples, 0.975)
        ),
    }


def paired_label_rows(
    model_id: str,
    condition: dict,
    original: pd.DataFrame,
    current: pd.DataFrame,
    replicates: int,
    seed: int,
) -> list[dict]:
    if original["sample_id"].tolist() != current["sample_id"].tolist():
        raise RuntimeError(f"Paired order mismatch for {model_id}/{condition['name']}")
    paired = original[
        ["sample_id", "label", "method", "group_id", "fake_probability", "predicted_class"]
    ].copy()
    paired = paired.rename(
        columns={
            "fake_probability": "original_fake_probability",
            "predicted_class": "original_predicted_class",
        }
    )
    paired["jpeg_fake_probability"] = current["fake_probability"].to_numpy(float)
    paired["jpeg_predicted_class"] = current["predicted_class"].to_numpy(int)
    paired["score_delta"] = (
        paired["jpeg_fake_probability"] - paired["original_fake_probability"]
    )
    paired["positive_delta"] = (
        paired["jpeg_predicted_class"] - paired["original_predicted_class"]
    )
    rows = []
    for label_name in ("real", "fake"):
        subset = paired[paired["label"] == label_name].copy()
        deltas = subset["score_delta"].to_numpy(float)
        intervals = cluster_bootstrap_interval(
            subset,
            replicates,
            stable_seed(seed, model_id, condition["name"], label_name),
        )
        metric_name = "false_positive_rate" if label_name == "real" else "recall"
        rows.append(
            {
                "model": model_id,
                "condition": condition["name"],
                "jpeg_quality": condition.get("quality"),
                "severity": int(condition["severity"]),
                "label_subset": label_name,
                "positive_rate_metric": metric_name,
                "images": len(subset),
                "groups": int(subset["group_id"].nunique()),
                "original_mean_fake_score": float(
                    subset["original_fake_probability"].mean()
                ),
                "original_median_fake_score": float(
                    subset["original_fake_probability"].median()
                ),
                "jpeg_mean_fake_score": float(
                    subset["jpeg_fake_probability"].mean()
                ),
                "jpeg_median_fake_score": float(
                    subset["jpeg_fake_probability"].median()
                ),
                "mean_score_delta": float(deltas.mean()),
                "median_score_delta": float(np.median(deltas)),
                "score_increase_rate": float((deltas > 0).mean()),
                "original_positive_rate": float(
                    subset["original_predicted_class"].mean()
                ),
                "jpeg_positive_rate": float(
                    subset["jpeg_predicted_class"].mean()
                ),
                "positive_rate_delta": float(subset["positive_delta"].mean()),
                **intervals,
            }
        )
    return rows


def paired_method_rows(
    model_id: str,
    model_condition: dict,
    condition: dict,
    original: pd.DataFrame,
    current: pd.DataFrame,
    replicates: int,
    seed: int,
) -> list[dict]:
    if original["sample_id"].tolist() != current["sample_id"].tolist():
        raise RuntimeError(f"Method pairing mismatch for {model_id}/{condition['name']}")
    paired = original[
        [
            "sample_id",
            "label",
            "method",
            "family",
            "role",
            "group_id",
            "fake_probability",
            "predicted_class",
        ]
    ].copy()
    paired = paired.rename(
        columns={
            "fake_probability": "original_fake_probability",
            "predicted_class": "original_predicted_class",
        }
    )
    paired["jpeg_fake_probability"] = current["fake_probability"].to_numpy(float)
    paired["jpeg_predicted_class"] = current["predicted_class"].to_numpy(int)
    paired["score_delta"] = (
        paired["jpeg_fake_probability"] - paired["original_fake_probability"]
    )
    paired["positive_delta"] = (
        paired["jpeg_predicted_class"] - paired["original_predicted_class"]
    )
    trained = set(model_condition["seen_fake_methods"])
    rows = []
    for method, subset in paired.groupby("method", sort=True):
        deltas = subset["score_delta"].to_numpy(float)
        intervals = cluster_bootstrap_interval(
            subset,
            replicates,
            stable_seed(seed, model_id, condition["name"], method),
        )
        label_name = subset["label"].iloc[0]
        rows.append(
            {
                "model": model_id,
                "training_condition": model_condition["name"],
                "jpeg_condition": condition["name"],
                "jpeg_quality": condition.get("quality"),
                "severity": int(condition["severity"]),
                "method": method,
                "family": subset["family"].iloc[0],
                "dataset_role": subset["role"].iloc[0],
                "relative_status": (
                    "real"
                    if label_name == "real"
                    else "trained_method"
                    if method in trained
                    else "held_out_method"
                ),
                "label": label_name,
                "images": len(subset),
                "groups": int(subset["group_id"].nunique()),
                "original_mean_fake_score": float(
                    subset["original_fake_probability"].mean()
                ),
                "original_median_fake_score": float(
                    subset["original_fake_probability"].median()
                ),
                "jpeg_mean_fake_score": float(
                    subset["jpeg_fake_probability"].mean()
                ),
                "jpeg_median_fake_score": float(
                    subset["jpeg_fake_probability"].median()
                ),
                "mean_score_delta": float(deltas.mean()),
                "median_score_delta": float(np.median(deltas)),
                "score_increase_rate": float((deltas > 0).mean()),
                "original_positive_rate": float(
                    subset["original_predicted_class"].mean()
                ),
                "jpeg_positive_rate": float(
                    subset["jpeg_predicted_class"].mean()
                ),
                "positive_rate_delta": float(subset["positive_delta"].mean()),
                **intervals,
            }
        )
    return rows


def add_overall_deltas(rows: list[dict]) -> None:
    by_model = {}
    for row in rows:
        if row["condition"] == "original":
            by_model[row["model"]] = row
    for row in rows:
        original = by_model[row["model"]]
        for metric in (
            "accuracy",
            "precision",
            "fake_recall",
            "real_recall",
            "real_fpr",
            "f1",
            "roc_auc",
            "auprc",
            "real_mean_fake_score",
            "real_median_fake_score",
            "fake_mean_fake_score",
            "fake_median_fake_score",
        ):
            row[f"{metric}_delta"] = float(row[metric] - original[metric])


def add_method_deltas(rows: list[dict]) -> None:
    originals = {
        (row["model"], row["method"]): row
        for row in rows
        if row["jpeg_condition"] == "original"
    }
    for row in rows:
        original = originals[(row["model"], row["method"])]
        for metric in (
            "fake_recall",
            "real_fpr",
            "f1",
            "roc_auc",
            "auprc",
            "fake_mean_fake_score",
            "fake_median_fake_score",
        ):
            row[f"{metric}_delta"] = float(row[metric] - original[metric])


def save_plots(condition_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    order = ["original", "jpeg_q95", "jpeg_q90", "jpeg_q75"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for model_id in MODEL_IDS:
        frame = (
            condition_summary[condition_summary["model"] == model_id]
            .set_index("condition")
            .loc[order]
        )
        axes[0].plot(order, frame["real_fpr"], marker="o", label=model_id)
        axes[1].plot(order, frame["roc_auc"], marker="o", label=model_id)
    axes[0].set_ylabel("Real false-positive rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(ncol=4)
    axes[1].set_ylabel("Overall ROC-AUC")
    axes[1].set_xlabel("Input condition")
    axes[1].set_ylim(-0.02, 1.02)
    figure.suptitle("M0-M7 response to matched JPEG round trips")
    figure.tight_layout()
    figure.savefig(plot_dir / "jpeg_fpr_and_auc.png", dpi=180)
    plt.close(figure)


def build_report(
    condition_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    lines = [
        "# M0–M7 JPEG Shortcut Diagnostic",
        "",
        "> Frozen-checkpoint diagnostic on the same 30,000-image canonical test set. JPEG was applied in memory to every real and fake image. No model was retrained, no source file was rewritten, and the decision rule remained logits argmax (0.5 equivalent).",
        "",
        "## Overall results",
        "",
        "| Model | Condition | Real FPR | Fake recall | ROC-AUC | AUPRC | AUC delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in condition_summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.condition} | {row.real_fpr:.4f} | "
            f"{row.fake_recall:.4f} | {row.roc_auc:.4f} | {row.auprc:.4f} | "
            f"{row.roc_auc_delta:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired real-image response at JPEG Q90",
            "",
            "| Model | Original FPR | Q90 FPR | FPR delta | Mean fake-score delta | 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    q90_real = paired_summary[
        (paired_summary["condition"] == "jpeg_q90")
        & (paired_summary["label_subset"] == "real")
    ]
    for row in q90_real.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.original_positive_rate:.4f} | "
            f"{row.jpeg_positive_rate:.4f} | {row.positive_rate_delta:+.4f} | "
            f"{row.mean_score_delta:+.4f} | "
            f"[{row.mean_score_delta_ci_low:+.4f}, {row.mean_score_delta_ci_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A large Real FPR increase shows JPEG sensitivity at the frozen decision rule. A simultaneous ROC-AUC collapse after applying the same JPEG transform to both classes is stronger evidence that discriminative ordering is lost, whereas stable AUC with high FPR indicates primarily a score-calibration shift. This diagnostic does not identify the training-data cause by itself and must not be reported as mitigation or retraining.",
            "",
        ]
    )
    return "\n".join(lines)


def versions() -> dict[str, str]:
    import matplotlib

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "pillow": PIL.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "runs_root": args.runs_root.resolve(),
        "existing_evaluation_root": args.existing_evaluation_root.resolve(),
        "output_dir": args.output_dir.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "diagnostic_config": args.diagnostic_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)

    diagnostic = json.loads(
        paths["diagnostic_config"].read_text(encoding="utf-8")
    )
    transformations = validate_protocol(diagnostic)
    jpeg_conditions = transformations[1:]
    model_conditions = load_conditions(paths["conditions_config"])

    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    if len(test_frame) != EXPECTED_TEST_IMAGES:
        raise ValueError("Unexpected canonical test size")
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
    if (test_frame["label"] == "real").sum() != EXPECTED_REAL_IMAGES:
        raise ValueError("Unexpected real test count")
    fake_counts = test_frame[test_frame["label"] == "fake"].groupby("method").size()
    if set(fake_counts.tolist()) != {EXPECTED_IMAGES_PER_FAKE_METHOD}:
        raise ValueError(f"Unexpected fake method counts: {fake_counts.to_dict()}")

    checkpoint_files = {}
    canonical_files = {}
    audits = {}
    identity = {
        "protocol": diagnostic["protocol"],
        "test_manifest_sha256": sha256_file(paths["test_manifest"]),
        "diagnostic_config_sha256": sha256_file(paths["diagnostic_config"]),
        "conditions_config_sha256": sha256_file(paths["conditions_config"]),
        "m0_config_sha256": sha256_file(paths["conditions_config"].with_name("m0.json")),
        "models": {},
    }
    for model_id in MODEL_IDS:
        model_condition = model_conditions[model_id]
        training_manifest = (
            paths["training_manifest_dir"] / f"{model_id.lower()}_seed42.csv"
        )
        checkpoint_file = checkpoint_path(
            paths["runs_root"], model_id, model_condition, args.seed
        )
        metrics_file = (
            paths["existing_evaluation_root"] / model_id.lower() / "metrics.json"
        )
        canonical_file = (
            paths["existing_evaluation_root"] / model_id.lower() / "predictions.csv"
        )
        for path in (training_manifest, checkpoint_file, metrics_file, canonical_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        validate_checkpoint(
            checkpoint,
            checkpoint_file,
            training_manifest,
            model_id,
            model_condition,
        )
        if checkpoint["config"].get("preprocessing_name") != LETTERBOX_NAME:
            raise ValueError(f"{model_id} is not a Letterbox checkpoint")
        existing_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if existing_metrics["test_manifest"]["sha256"] != identity["test_manifest_sha256"]:
            raise RuntimeError(f"Existing {model_id} evaluation used another test manifest")
        if existing_metrics["training_manifest"]["sha256"] != sha256_file(training_manifest):
            raise RuntimeError(f"Existing {model_id} evaluation used another training manifest")
        development = read_manifest(training_manifest)
        audits[model_id] = cross_split_audit(development, test_frame)
        checkpoint_files[model_id] = checkpoint_file
        canonical_files[model_id] = canonical_file
        identity["models"][model_id] = {
            "training_manifest_sha256": sha256_file(training_manifest),
            "checkpoint_sha256": sha256_file(checkpoint_file),
            "canonical_predictions_sha256": sha256_file(canonical_file),
        }
        del checkpoint

    output_dir = paths["output_dir"]
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(
                f"Non-empty diagnostic directory has no run identity: {output_dir}"
            )
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != identity:
            raise RuntimeError("Existing diagnostic cache belongs to another protocol")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required")

    condition_rows: list[dict] = []
    all_method_rows: list[dict] = []
    paired_rows: list[dict] = []
    all_method_paired_rows: list[dict] = []
    for model_id in MODEL_IDS:
        print(f"\n=== {model_id}: loading canonical predictions ===", flush=True)
        model_condition = model_conditions[model_id]
        original = load_canonical_predictions(
            canonical_files[model_id], test_frame
        )
        condition_rows.append(overall_row(model_id, transformations[0], original))
        all_method_rows.extend(
            method_rows(
                model_id,
                model_condition,
                transformations[0],
                original,
            )
        )

        checkpoint = torch.load(
            checkpoint_files[model_id], map_location=device, weights_only=False
        )
        model = timm.create_model(
            checkpoint["model_name"],
            pretrained=False,
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model_transform = evaluation_transform_from_checkpoint(checkpoint)

        for jpeg_condition in jpeg_conditions:
            prediction_path = (
                output_dir
                / "predictions"
                / model_id.lower()
                / f"{jpeg_condition['name']}.csv"
            )
            if prediction_path.is_file():
                current = pd.read_csv(
                    prediction_path, dtype=str, keep_default_na=False
                )
                current = validate_prediction_frame(
                    current,
                    test_frame,
                    jpeg_condition["name"],
                    prediction_path,
                )
                print(
                    f"Skipping completed {model_id}/{jpeg_condition['name']}",
                    flush=True,
                )
            else:
                dataset = JpegEvaluationDataset(
                    test_frame,
                    paths["data_root"],
                    jpeg_condition,
                    model_transform,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=args.workers > 0,
                )
                indices, predictions, probabilities = predict_condition(
                    model,
                    loader,
                    device,
                    f"{model_id} {jpeg_condition['name']}",
                )
                if indices != list(range(len(test_frame))):
                    raise RuntimeError("JPEG inference order changed unexpectedly")
                current = build_prediction_frame(
                    test_frame,
                    jpeg_condition["name"],
                    predictions,
                    probabilities,
                )
                atomic_to_csv(current, prediction_path)

            condition_rows.append(overall_row(model_id, jpeg_condition, current))
            all_method_rows.extend(
                method_rows(
                    model_id,
                    model_condition,
                    jpeg_condition,
                    current,
                )
            )
            paired_rows.extend(
                paired_label_rows(
                    model_id,
                    jpeg_condition,
                    original,
                    current,
                    args.bootstrap_replicates,
                    args.seed,
                )
            )
            all_method_paired_rows.extend(
                paired_method_rows(
                    model_id,
                    model_condition,
                    jpeg_condition,
                    original,
                    current,
                    args.bootstrap_replicates,
                    args.seed,
                )
            )

        del model, checkpoint, model_transform
        torch.cuda.empty_cache()

    add_overall_deltas(condition_rows)
    add_method_deltas(all_method_rows)
    condition_summary = pd.DataFrame(condition_rows)
    method_summary = pd.DataFrame(all_method_rows)
    paired_summary = pd.DataFrame(paired_rows)
    method_paired_summary = pd.DataFrame(all_method_paired_rows)
    atomic_to_csv(condition_summary, output_dir / "condition_summary.csv")
    atomic_to_csv(method_summary, output_dir / "method_condition_summary.csv")
    atomic_to_csv(paired_summary, output_dir / "paired_label_summary.csv")
    atomic_to_csv(
        method_paired_summary,
        output_dir / "method_paired_score_summary.csv",
    )
    save_plots(condition_summary, output_dir)
    report = build_report(condition_summary, paired_summary)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    final_summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": diagnostic["protocol"],
        "models": MODEL_IDS,
        "conditions": [item["name"] for item in transformations],
        "test_images": len(test_frame),
        "real_images": int((test_frame["label"] == "real").sum()),
        "fake_images": int((test_frame["label"] == "fake").sum()),
        "fake_methods": int(fake_counts.size),
        "canonical_predictions_reused": True,
        "source_images_modified": False,
        "checkpoints_modified": False,
        "threshold": 0.5,
        "test_used_for_training_or_tuning": False,
        "jpeg_encoder": {
            "library": "Pillow",
            "subsampling": "4:2:0 (Pillow value 2)",
            "optimize": False,
            "progressive": False,
        },
        "bootstrap": {
            "unit": "group_id cluster",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "cross_split_audits": audits,
        "outputs": {
            "report": "REPORT.md",
            "condition_summary": "condition_summary.csv",
            "method_condition_summary": "method_condition_summary.csv",
            "paired_label_summary": "paired_label_summary.csv",
            "method_paired_score_summary": "method_paired_score_summary.csv",
            "plot": "plots/jpeg_fpr_and_auc.png",
            "predictions": "predictions/{model}/{jpeg_condition}.csv",
        },
        "versions": versions(),
    }
    write_json(output_dir / "diagnostic_summary.json", final_summary)
    print("\n=== JPEG diagnostic summary ===")
    print(
        condition_summary[
            [
                "model",
                "condition",
                "real_fpr",
                "fake_recall",
                "roc_auc",
                "auprc",
                "roc_auc_delta",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
