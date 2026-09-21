#!/usr/bin/env python3
"""Nested soft-voting test of transfer-derived expert weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "method_transfer_ensemble_v1"
REQUIRED_PREDICTION_COLUMNS = {
    "sample_id",
    "source_path",
    "label",
    "method",
    "family",
    "true_class",
    "fake_probability",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC requires both classes")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        raise ValueError("Average precision requires positive samples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
        len(sorted_scores) - 1,
    ]
    cumulative_true = np.cumsum(sorted_labels == 1)[group_ends]
    cumulative_total = group_ends + 1
    precision = cumulative_true / cumulative_total
    recall = cumulative_true / positive_count
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def metrics_from_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    total = len(labels)
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    actual_negative = true_negative + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    fake_recall = true_positive / actual_positive
    real_recall = true_negative / actual_negative
    f1 = (
        2 * precision * fake_recall / (precision + fake_recall)
        if precision + fake_recall
        else 0.0
    )
    return {
        "accuracy": float((true_positive + true_negative) / total),
        "precision": float(precision),
        "fake_recall": float(fake_recall),
        "real_recall": float(real_recall),
        "real_fpr": float(false_positive / actual_negative),
        "f1": float(f1),
        "roc_auc": float(binary_auc(labels, probabilities)),
        "auprc": float(average_precision(labels, probabilities)),
        "real_mean_fake_score": float(probabilities[labels == 0].mean()),
        "fake_mean_fake_score": float(probabilities[labels == 1].mean()),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-evaluation-dir",
        type=Path,
        required=True,
        help="Completed method_transfer_matrix_v1 evaluation directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {path}")
    if config["selection_boundary"]["protected_unseen_used"]:
        raise ValueError("Protected unseen data must remain disabled")
    if config["selection_boundary"]["wilddeepfake_used"]:
        raise ValueError("WildDeepfake must remain disabled")
    return config


def load_transfer_summary(root: Path, config: dict) -> pd.DataFrame:
    path = root / "transfer_matrix_long.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "training_method",
        "target_method",
        "evaluation_condition",
        "roc_auc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing transfer columns: {sorted(missing)}")
    methods = list(config["methods"])
    conditions = config["evaluation_conditions"]
    expected = {
        (source, target, condition)
        for source in methods
        for target in methods
        for condition in conditions
    }
    actual = set(
        zip(
            frame["training_method"],
            frame["target_method"],
            frame["evaluation_condition"],
        )
    )
    if actual != expected or len(frame) != len(expected):
        missing_rows = sorted(expected - actual)
        extra_rows = sorted(actual - expected)
        raise RuntimeError(
            "Transfer matrix is incomplete or incompatible. "
            f"Missing={missing_rows[:3]}, extra={extra_rows[:3]}"
        )
    if not frame["roc_auc"].between(0.0, 1.0).all():
        raise ValueError("Transfer AUC must be in [0, 1]")
    return frame


def transfer_weights(
    transfer: pd.DataFrame,
    methods: list[str],
    conditions: list[str],
    outer_method: str,
    chance_auc: float,
) -> tuple[dict[str, float], dict[str, float]]:
    candidates = [method for method in methods if method != outer_method]
    utilities = {}
    for expert in candidates:
        targets = [method for method in candidates if method != expert]
        rows = transfer[
            (transfer["training_method"] == expert)
            & (transfer["target_method"].isin(targets))
            & (transfer["evaluation_condition"].isin(conditions))
        ]
        expected_count = len(targets) * len(conditions)
        if len(rows) != expected_count:
            raise RuntimeError(
                f"Unexpected utility rows for {outer_method}/{expert}: "
                f"{len(rows)} != {expected_count}"
            )
        if outer_method in set(rows["training_method"]) | set(
            rows["target_method"]
        ):
            raise RuntimeError("Outer method leaked into transfer weighting")
        excess = np.maximum(rows["roc_auc"].to_numpy(float) - chance_auc, 0.0)
        utilities[expert] = float(excess.mean())
    total = float(sum(utilities.values()))
    if total <= 0.0:
        raise RuntimeError(f"No positive transfer utility for {outer_method}")
    weights = {method: value / total for method, value in utilities.items()}
    return utilities, weights


def load_condition_predictions(
    root: Path,
    config: dict,
    condition: str,
) -> dict[str, pd.DataFrame]:
    result = {}
    reference = None
    metadata_columns = [
        "sample_id",
        "source_path",
        "label",
        "method",
        "family",
        "true_class",
    ]
    for method, definition in config["methods"].items():
        path = root / definition["slug"] / condition / "predictions.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample_id in {path}")
        if not frame["fake_probability"].between(0.0, 1.0).all():
            raise ValueError(f"Invalid fake probability in {path}")
        current = frame[metadata_columns].reset_index(drop=True)
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise RuntimeError(
                f"Expert predictions are not aligned for {condition}: {path}"
            )
        result[method] = frame.reset_index(drop=True)
    return result


def evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> dict:
    predictions = (scores >= 0.5).astype(int)
    return metrics_from_arrays(labels, predictions, scores)


def build_report(
    strategy_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: dict,
) -> str:
    lines = [
        "# Nested Transfer-Weighted Expert Ensemble",
        "",
        (
            "> No new model was trained. For each outer holdout, its expert and "
            "its transfer-matrix row/column were removed before combining the "
            "other five experts. Protected unseen methods and WildDeepfake were "
            "not used."
        ),
        "",
        "## Strategy summary",
        "",
        "| Condition | Strategy | Mean AUC | Worst AUC | Mean AUPRC | Mean Real FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in strategy_summary.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.strategy} | "
            f"{row.mean_auc:.4f} | {row.worst_auc:.4f} | "
            f"{row.mean_auprc:.4f} | {row.mean_real_fpr:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Transfer-weighted minus uniform",
            "",
            "| Condition | Mean AUC delta | Worst AUC delta | Wins / 6 | Mean FPR delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.mean_auc_delta:+.4f} | "
            f"{row.worst_auc_delta:+.4f} | {row.weighted_wins} / "
            f"{row.outer_holdouts} | {row.mean_real_fpr_delta:+.4f} |"
        )
    criterion = config["advance_criterion"]
    lines.extend(
        [
            "",
            "## Pre-declared decision boundary",
            "",
            criterion["description"],
            "",
            (
                "A positive result only justifies a later training-side "
                "distillation pilot. It is not itself a new trained detector "
                "and is not evidence from protected unseen data."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    transfer_root = args.transfer_evaluation_dir.resolve()
    output_root = args.output_dir.resolve()
    config_path = args.protocol_config.resolve()
    if not transfer_root.is_dir():
        raise FileNotFoundError(transfer_root)
    config = load_config(config_path)
    methods = list(config["methods"])
    conditions = config["evaluation_conditions"]
    transfer = load_transfer_summary(transfer_root, config)

    source_summary = transfer_root / "evaluation_summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    source_identity = json.loads(source_summary.read_text(encoding="utf-8"))
    if source_identity.get("protocol") != config["parent_transfer_protocol"]:
        raise ValueError("Unexpected parent transfer evaluation protocol")
    if source_identity.get("protected_unseen_used"):
        raise ValueError("Parent evaluation used protected unseen data")
    if source_identity.get("wilddeepfake_used"):
        raise ValueError("Parent evaluation used WildDeepfake")

    prediction_hashes = {}
    for method, definition in config["methods"].items():
        prediction_hashes[method] = {}
        for condition in conditions:
            path = (
                transfer_root
                / definition["slug"]
                / condition
                / "predictions.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            prediction_hashes[method][condition] = sha256_file(path)

    identity = {
        "protocol": PROTOCOL,
        "protocol_config_sha256": sha256_file(config_path),
        "parent_evaluation_summary_sha256": sha256_file(source_summary),
        "parent_transfer_matrix_sha256": sha256_file(
            transfer_root / "transfer_matrix_long.csv"
        ),
        "methods": methods,
        "evaluation_conditions": conditions,
        "prediction_sha256": prediction_hashes,
        "protected_unseen_used": False,
        "wilddeepfake_used": False,
    }
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(
                f"Non-empty output has no run identity: {output_root}"
            )
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing output belongs to a different run")
        required_outputs = [
            "nested_ensemble_results.csv",
            "strategy_summary.csv",
            "transfer_weighted_comparisons.csv",
            "ensemble_weights.csv",
            "nested_ensemble_predictions.csv",
            "REPORT.md",
            "evaluation_summary.json",
        ]
        if all((output_root / name).is_file() for name in required_outputs):
            print("Already complete:", output_root, flush=True)
            return
        raise FileExistsError(f"Incomplete output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, identity)

    weight_rows = []
    outer_weights = {}
    chance_auc = float(config["transfer_weighting"]["chance_auc"])
    for outer in methods:
        utilities, weights = transfer_weights(
            transfer, methods, conditions, outer, chance_auc
        )
        outer_weights[outer] = weights
        for expert in methods:
            if expert == outer:
                continue
            weight_rows.append(
                {
                    "outer_holdout": outer,
                    "expert_method": expert,
                    "expert_family": config["methods"][expert]["family"],
                    "transfer_utility": utilities[expert],
                    "uniform_weight": 1.0 / (len(methods) - 1),
                    "transfer_weight": weights[expert],
                }
            )

    result_rows = []
    prediction_rows = []
    for condition in conditions:
        print(f"\n=== {condition} ===", flush=True)
        predictions = load_condition_predictions(
            transfer_root, config, condition
        )
        reference = predictions[methods[0]]
        for outer in methods:
            candidates = [method for method in methods if method != outer]
            pair_mask = (reference["label"] == "real") | (
                (reference["label"] == "fake")
                & (reference["method"] == outer)
            )
            pair = reference.loc[pair_mask].reset_index(drop=True)
            labels = pair["true_class"].to_numpy(int)
            expected_per_class = int(
                config["evaluation"]["expected_images_per_class"]
            )
            class_counts = {
                int(label): int(count)
                for label, count in zip(*np.unique(labels, return_counts=True))
            }
            if class_counts != {0: expected_per_class, 1: expected_per_class}:
                raise RuntimeError(
                    f"Unexpected pair counts for {condition}/{outer}: "
                    f"{class_counts}"
                )
            candidate_scores = np.column_stack(
                [
                    predictions[expert]
                    .loc[pair_mask, "fake_probability"]
                    .to_numpy(float)
                    for expert in candidates
                ]
            )
            strategies = {
                "uniform5": candidate_scores.mean(axis=1),
                "transfer_weighted5": candidate_scores
                @ np.asarray(
                    [outer_weights[outer][expert] for expert in candidates],
                    dtype=float,
                ),
            }
            for strategy, scores in strategies.items():
                metrics = evaluate_scores(labels, scores)
                result_rows.append(
                    {
                        "outer_holdout": outer,
                        "outer_family": config["methods"][outer]["family"],
                        "evaluation_condition": condition,
                        "strategy": strategy,
                        "experts": "|".join(candidates),
                        "images_real": int((labels == 0).sum()),
                        "images_fake": int((labels == 1).sum()),
                        **metrics,
                    }
                )
                for metadata, score in zip(
                    pair[
                        ["sample_id", "source_path", "label", "method", "family"]
                    ].itertuples(index=False),
                    scores,
                ):
                    predicted = int(score >= 0.5)
                    true_class = 0 if metadata.label == "real" else 1
                    prediction_rows.append(
                        {
                            "outer_holdout": outer,
                            "evaluation_condition": condition,
                            "strategy": strategy,
                            "sample_id": metadata.sample_id,
                            "source_path": metadata.source_path,
                            "label": metadata.label,
                            "method": metadata.method,
                            "family": metadata.family,
                            "true_class": true_class,
                            "predicted_class": predicted,
                            "fake_probability": float(score),
                            "correct": predicted == true_class,
                        }
                    )
                print(
                    f"{outer:10s} {strategy:18s} "
                    f"AUC={metrics['roc_auc']:.4f} "
                    f"FPR={metrics['real_fpr']:.4f}",
                    flush=True,
                )

    results = pd.DataFrame(result_rows).sort_values(
        ["evaluation_condition", "outer_holdout", "strategy"]
    )
    summary = (
        results.groupby(["evaluation_condition", "strategy"], as_index=False)
        .agg(
            mean_auc=("roc_auc", "mean"),
            worst_auc=("roc_auc", "min"),
            mean_auprc=("auprc", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            mean_real_fpr=("real_fpr", "mean"),
            mean_fake_recall=("fake_recall", "mean"),
        )
        .sort_values(["evaluation_condition", "strategy"])
    )
    comparison_rows = []
    for condition in conditions:
        subset = results[results["evaluation_condition"] == condition]
        pivot_auc = subset.pivot(
            index="outer_holdout", columns="strategy", values="roc_auc"
        )
        pivot_fpr = subset.pivot(
            index="outer_holdout", columns="strategy", values="real_fpr"
        )
        uniform_summary = summary[
            (summary["evaluation_condition"] == condition)
            & (summary["strategy"] == "uniform5")
        ].iloc[0]
        weighted_summary = summary[
            (summary["evaluation_condition"] == condition)
            & (summary["strategy"] == "transfer_weighted5")
        ].iloc[0]
        deltas = pivot_auc["transfer_weighted5"] - pivot_auc["uniform5"]
        comparison_rows.append(
            {
                "evaluation_condition": condition,
                "mean_auc_delta": float(deltas.mean()),
                "worst_auc_delta": float(
                    weighted_summary["worst_auc"]
                    - uniform_summary["worst_auc"]
                ),
                "weighted_wins": int((deltas > 0).sum()),
                "ties": int(np.isclose(deltas, 0.0, atol=1e-12).sum()),
                "outer_holdouts": int(len(deltas)),
                "mean_real_fpr_delta": float(
                    (
                        pivot_fpr["transfer_weighted5"]
                        - pivot_fpr["uniform5"]
                    ).mean()
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    weights = pd.DataFrame(weight_rows).sort_values(
        ["outer_holdout", "expert_method"]
    )
    all_predictions = pd.DataFrame(prediction_rows).sort_values(
        ["evaluation_condition", "outer_holdout", "strategy", "sample_id"]
    )

    results.to_csv(output_root / "nested_ensemble_results.csv", index=False)
    summary.to_csv(output_root / "strategy_summary.csv", index=False)
    comparisons.to_csv(
        output_root / "transfer_weighted_comparisons.csv", index=False
    )
    weights.to_csv(output_root / "ensemble_weights.csv", index=False)
    all_predictions.to_csv(
        output_root / "nested_ensemble_predictions.csv", index=False
    )
    (output_root / "REPORT.md").write_text(
        build_report(summary, comparisons, config), encoding="utf-8"
    )
    q95 = config["evaluation"]["primary_condition"]
    q95_comparison = comparisons[
        comparisons["evaluation_condition"] == q95
    ].iloc[0]
    criterion = config["advance_criterion"]
    advance = bool(
        q95_comparison["mean_auc_delta"] > 0.0
        and int(q95_comparison["weighted_wins"])
        >= int(criterion["minimum_q95_wins"])
        and q95_comparison["worst_auc_delta"]
        >= -float(criterion["maximum_worst_auc_regression"])
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "protocol": PROTOCOL,
            "parent_transfer_protocol": config["parent_transfer_protocol"],
            "outer_holdouts": methods,
            "evaluation_conditions": conditions,
            "strategies": list(config["strategies"]),
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
            "new_model_training": False,
            "advance_to_distillation_pilot": advance,
            "advance_criterion": criterion,
            "outputs": {
                "nested_results": "nested_ensemble_results.csv",
                "strategy_summary": "strategy_summary.csv",
                "comparisons": "transfer_weighted_comparisons.csv",
                "weights": "ensemble_weights.csv",
                "predictions": "nested_ensemble_predictions.csv",
                "report": "REPORT.md",
            },
        },
    )
    print("\nSaved:", output_root, flush=True)
    print("Advance to distillation pilot:", advance, flush=True)


if __name__ == "__main__":
    main()
