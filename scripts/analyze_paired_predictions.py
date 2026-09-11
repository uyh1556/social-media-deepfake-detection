import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "source_reference_path",
    "split",
    "label",
    "method",
    "group_id",
    "video_id",
    "true_class",
    "predicted_class",
    "fake_probability",
    "correct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Perform paired raw/target score, flip, and group-bootstrap "
            "analysis without changing either model evaluation."
        )
    )
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--target-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc_score(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = pd.Series(probabilities).rank(method="average").to_numpy()
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    positive_rank_sum = float(probabilities[positives].sum())
    return float(
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / float(n_positive * n_negative)


def classification_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    fake_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    real_recall = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    f1 = (
        2.0 * precision * fake_recall / (precision + fake_recall)
        if precision + fake_recall
        else 0.0
    )
    return {
        "accuracy": float((labels == predictions).mean()),
        "precision": precision,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "f1": f1,
        "roc_auc": auc_score(labels, probabilities),
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def numeric_distribution(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def score_summary(frame):
    return {
        "images": int(len(frame)),
        "probability_difference": numeric_distribution(
            frame["probability_difference"]
        ),
        "absolute_probability_difference": numeric_distribution(
            frame["absolute_probability_difference"]
        ),
        "true_class_confidence_difference": numeric_distribution(
            frame["true_class_confidence_difference"]
        ),
        "prediction_flip_count": int(frame["prediction_flip"].sum()),
        "prediction_flip_rate": float(frame["prediction_flip"].mean()),
    }


def bootstrap_metric_deltas(frame, replicates, seed):
    metric_names = [
        "accuracy", "precision", "fake_recall", "real_recall", "f1", "roc_auc"
    ]
    groups = frame["group_id"].astype(str).unique()
    group_indices = {
        group: np.flatnonzero(frame["group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    labels = frame["true_class"].to_numpy(dtype=np.int64)
    raw_predictions = frame["raw_predicted_class"].to_numpy(dtype=np.int64)
    target_predictions = frame["target_predicted_class"].to_numpy(dtype=np.int64)
    raw_probabilities = frame["raw_fake_probability"].to_numpy(dtype=np.float64)
    target_probabilities = frame["target_fake_probability"].to_numpy(
        dtype=np.float64
    )
    observed_raw = classification_metrics(
        labels, raw_predictions, raw_probabilities
    )
    observed_target = classification_metrics(
        labels, target_predictions, target_probabilities
    )
    samples = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        raw = classification_metrics(
            labels[indices], raw_predictions[indices], raw_probabilities[indices]
        )
        target = classification_metrics(
            labels[indices],
            target_predictions[indices],
            target_probabilities[indices],
        )
        for metric in metric_names:
            if raw[metric] is not None and target[metric] is not None:
                samples[metric].append(target[metric] - raw[metric])

    result = {}
    for metric in metric_names:
        values = np.asarray(samples[metric], dtype=np.float64)
        observed = observed_target[metric] - observed_raw[metric]
        result[metric] = {
            "observed_delta": float(observed),
            "bootstrap_mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "replicates": int(len(values)),
            "resampling_unit": "source-aware group_id",
        }
    return observed_raw, observed_target, result


def main():
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")
    raw_path = args.raw_predictions.resolve()
    target_path = args.target_predictions.resolve()
    output_dir = args.output_dir.resolve()
    pairs_path = output_dir / "paired_predictions.csv"
    summary_path = output_dir / "paired_prediction_summary.json"
    for path in [raw_path, target_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.overwrite and (pairs_path.exists() or summary_path.exists()):
        raise FileExistsError("Prediction-analysis outputs exist; use --overwrite.")

    raw = pd.read_csv(
        raw_path, dtype={"group_id": str, "video_id": str}
    )
    target = pd.read_csv(
        target_path, dtype={"group_id": str, "video_id": str}
    )
    for name, frame in [("raw", raw), ("target", target)]:
        if missing := REQUIRED_COLUMNS - set(frame.columns):
            raise ValueError(f"{name} predictions missing: {sorted(missing)}")
        if frame["source_reference_path"].duplicated().any():
            raise ValueError(f"{name} source references are not unique.")
    if set(raw.source_reference_path) != set(target.source_reference_path):
        raise ValueError("Raw and target source-reference sets differ.")

    keep = [
        "source_reference_path", "split", "label", "method", "group_id",
        "video_id", "true_class", "predicted_class", "fake_probability",
        "correct",
    ]
    frame = raw[keep].merge(
        target[keep],
        on="source_reference_path",
        suffixes=("_raw", "_target"),
        validate="one_to_one",
    )
    for column in [
        "split", "label", "method", "group_id", "video_id", "true_class"
    ]:
        if not (frame[f"{column}_raw"] == frame[f"{column}_target"]).all():
            raise ValueError(f"Paired metadata differs: {column}")
        frame[column] = frame[f"{column}_raw"]
    frame = frame.rename(
        columns={
            "predicted_class_raw": "raw_predicted_class",
            "predicted_class_target": "target_predicted_class",
            "fake_probability_raw": "raw_fake_probability",
            "fake_probability_target": "target_fake_probability",
            "correct_raw": "raw_correct",
            "correct_target": "target_correct",
        }
    )
    frame["probability_difference"] = (
        frame.target_fake_probability - frame.raw_fake_probability
    )
    frame["absolute_probability_difference"] = frame[
        "probability_difference"
    ].abs()
    frame["raw_confidence"] = np.maximum(
        frame.raw_fake_probability, 1.0 - frame.raw_fake_probability
    )
    frame["target_confidence"] = np.maximum(
        frame.target_fake_probability, 1.0 - frame.target_fake_probability
    )
    labels = frame.true_class.to_numpy(dtype=np.int64)
    frame["raw_true_class_confidence"] = np.where(
        labels == 1,
        frame.raw_fake_probability,
        1.0 - frame.raw_fake_probability,
    )
    frame["target_true_class_confidence"] = np.where(
        labels == 1,
        frame.target_fake_probability,
        1.0 - frame.target_fake_probability,
    )
    frame["true_class_confidence_difference"] = (
        frame.target_true_class_confidence
        - frame.raw_true_class_confidence
    )
    frame["prediction_flip"] = (
        frame.raw_predicted_class != frame.target_predicted_class
    )
    frame["raw_correct"] = (
        frame.raw_predicted_class == frame.true_class
    )
    frame["target_correct"] = (
        frame.target_predicted_class == frame.true_class
    )
    frame["raw_correct_target_wrong"] = (
        frame.raw_correct.astype(bool) & ~frame.target_correct.astype(bool)
    )
    frame["raw_wrong_target_correct"] = (
        ~frame.raw_correct.astype(bool) & frame.target_correct.astype(bool)
    )

    raw_metrics, target_metrics, confidence_intervals = (
        bootstrap_metric_deltas(
            frame, args.bootstrap_replicates, args.seed
        )
    )
    metric_delta = {
        key: (
            target_metrics[key] - raw_metrics[key]
            if isinstance(raw_metrics[key], (int, float))
            and raw_metrics[key] is not None
            else None
        )
        for key in raw_metrics
        if key != "confusion_matrix"
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_name": args.target_name,
        "pairs": int(len(frame)),
        "groups": int(frame.group_id.nunique()),
        "inputs": {
            "raw_predictions": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
            },
            "target_predictions": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
            },
        },
        "raw_metrics": raw_metrics,
        "target_metrics": target_metrics,
        "metric_delta_target_minus_raw": metric_delta,
        "paired_score_changes": score_summary(frame),
        "correctness_transitions": {
            "raw_correct_target_wrong": int(
                frame.raw_correct_target_wrong.sum()
            ),
            "raw_wrong_target_correct": int(
                frame.raw_wrong_target_correct.sum()
            ),
            "both_correct": int(
                (frame.raw_correct.astype(bool) & frame.target_correct.astype(bool)).sum()
            ),
            "both_wrong": int(
                (~frame.raw_correct.astype(bool) & ~frame.target_correct.astype(bool)).sum()
            ),
        },
        "by_label": {
            label: score_summary(group)
            for label, group in frame.groupby("label", sort=True)
        },
        "by_method": {
            method: score_summary(group)
            for method, group in frame.groupby("method", sort=True)
        },
        "group_bootstrap_delta_ci95": confidence_intervals,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(pairs_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metric_delta_target_minus_raw"], indent=2))
    print(f"Paired predictions: {pairs_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
