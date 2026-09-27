#!/usr/bin/env python3
"""Nested test of transfer-matrix complementarity on existing expert scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from analyze_method_transfer_ensemble import evaluate_scores
from create_family_rotation_method_archives import METHOD_SLUGS


CONDITIONS = [
    "canonical_256_jpeg_q95",
    "canonical_256_jpeg_q90",
    "canonical_256_jpeg_q75",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selections", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument(
        "--uniform-floor",
        type=float,
        default=0.5,
        help="Fraction of the final weights reserved for uniform weighting.",
    )
    return parser.parse_args()


def locate_selection(root: Path, selection: str) -> Path:
    candidates = [
        root / f"{selection.lower()}_corrected_seed42_q95_q90_q75",
        root / f"{selection.lower()}_seed42_q95_q90_q75",
    ]
    matches = [path for path in candidates if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one evaluation directory for {selection}: {candidates}")
    return matches[0]


def load_transfer(root: Path) -> tuple[pd.DataFrame, list[str]]:
    path = root / "transfer_matrix_long.csv"
    frame = pd.read_csv(path)
    required = {"training_method", "target_method", "evaluation_condition", "roc_auc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Invalid transfer matrix: {path}")
    methods = list(dict.fromkeys(frame["training_method"].tolist()))
    if len(methods) != 6 or set(methods) != set(frame["target_method"]):
        raise RuntimeError(f"Expected a complete six-method matrix: {path}")
    return frame, methods


def complementarity_weights(
    transfer: pd.DataFrame,
    candidates: list[str],
    conditions: list[str],
    uniform_floor: float,
) -> tuple[dict[str, float], float]:
    """Maximise worst cross-method coverage; diagonal/self transfer is excluded."""
    utility = np.zeros((len(candidates), len(candidates)), dtype=float)
    for i, source in enumerate(candidates):
        for j, target in enumerate(candidates):
            if source == target:
                continue
            rows = transfer[
                (transfer.training_method == source)
                & (transfer.target_method == target)
                & (transfer.evaluation_condition.isin(conditions))
            ]
            if len(rows) != len(conditions):
                raise RuntimeError(f"Incomplete inner matrix: {source} -> {target}")
            utility[i, j] = max(float(rows.roc_auc.mean()) - 0.5, 0.0)

    # Variables are five weights followed by the minimum covered utility z.
    objective = np.r_[np.zeros(len(candidates)), -1.0]
    inequalities = np.column_stack([-utility.T, np.ones(len(candidates))])
    solution = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(len(candidates)),
        A_eq=np.asarray([np.r_[np.ones(len(candidates)), 0.0]]),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * len(candidates) + [(0.0, None)],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    optimized = solution.x[:-1]
    weights = uniform_floor / len(candidates) + (1.0 - uniform_floor) * optimized
    weights /= weights.sum()
    return dict(zip(candidates, weights.tolist())), float(solution.x[-1])


def load_predictions(root: Path, methods: list[str], condition: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    reference_ids = None
    for method in methods:
        path = root / METHOD_SLUGS[method] / condition / "predictions.csv"
        frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
        ids = frame.sample_id.tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError(f"Prediction rows do not align: {path}")
        result[method] = frame
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.uniform_floor <= 1.0:
        raise ValueError("--uniform-floor must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_rows, weight_rows = [], []

    for selection in [value.upper() for value in args.selections]:
        selection_root = locate_selection(args.transfer_root, selection)
        transfer, methods = load_transfer(selection_root)
        for outer in methods:
            candidates = [method for method in methods if method != outer]
            weights, worst_inner_coverage = complementarity_weights(
                transfer, candidates, CONDITIONS, args.uniform_floor
            )
            for expert in candidates:
                weight_rows.append({
                    "selection": selection,
                    "outer_holdout": outer,
                    "expert_method": expert,
                    "uniform_weight": 1.0 / len(candidates),
                    "complementarity_weight": weights[expert],
                    "optimized_worst_inner_coverage": worst_inner_coverage,
                })

            for condition in CONDITIONS:
                predictions = load_predictions(selection_root, methods, condition)
                reference = predictions[methods[0]]
                mask = (reference.label == "real") | (
                    (reference.label == "fake") & (reference.method == outer)
                )
                pair = reference.loc[mask]
                labels = pair.true_class.to_numpy(int)
                scores = np.column_stack([
                    predictions[expert].loc[mask, "fake_probability"].to_numpy(float)
                    for expert in candidates
                ])
                strategies = {
                    "uniform5": scores.mean(axis=1),
                    "complementarity5": scores @ np.asarray([weights[m] for m in candidates]),
                }
                for strategy, values in strategies.items():
                    result_rows.append({
                        "selection": selection,
                        "outer_holdout": outer,
                        "evaluation_condition": condition,
                        "strategy": strategy,
                        **evaluate_scores(labels, values),
                    })
                print(selection, outer, condition, "done", flush=True)

    results = pd.DataFrame(result_rows)
    weights = pd.DataFrame(weight_rows)
    pivot = results.pivot_table(
        index=["selection", "outer_holdout", "evaluation_condition"],
        columns="strategy", values=["roc_auc", "real_fpr"],
    )
    comparisons = pivot.reset_index()
    comparisons.columns = ["_".join(filter(None, map(str, col))).rstrip("_") for col in comparisons.columns]
    comparisons["auc_delta"] = comparisons["roc_auc_complementarity5"] - comparisons["roc_auc_uniform5"]
    comparisons["real_fpr_delta"] = comparisons["real_fpr_complementarity5"] - comparisons["real_fpr_uniform5"]
    summary = comparisons.groupby(["selection", "evaluation_condition"], as_index=False).agg(
        mean_auc_delta=("auc_delta", "mean"),
        worst_auc_delta=("auc_delta", "min"),
        wins=("auc_delta", lambda values: int((values > 0).sum())),
        mean_real_fpr_delta=("real_fpr_delta", "mean"),
        holdouts=("outer_holdout", "nunique"),
    )
    overall = comparisons.groupby("evaluation_condition", as_index=False).agg(
        mean_auc_delta=("auc_delta", "mean"),
        worst_auc_delta=("auc_delta", "min"),
        wins=("auc_delta", lambda values: int((values > 0).sum())),
        mean_real_fpr_delta=("real_fpr_delta", "mean"),
        holdouts=("outer_holdout", "size"),
    )
    results.to_csv(args.output_dir / "nested_results.csv", index=False)
    weights.to_csv(args.output_dir / "complementarity_weights.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    summary.to_csv(args.output_dir / "selection_summary.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    (args.output_dir / "REPORT.md").write_text(
        "# S1-S3 Nested Complementarity\n\n"
        "No new model was trained. Each outer method and its transfer row/column were hidden.\n\n"
        + summary.to_markdown(index=False, floatfmt=".6f") + "\n\n## Overall\n\n"
        + overall.to_markdown(index=False, floatfmt=".6f") + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps({
        "protocol": "family_rotation_nested_complementarity_v1",
        "selections": [value.upper() for value in args.selections],
        "conditions": CONDITIONS,
        "uniform_floor": args.uniform_floor,
        "outer_holdout_used_for_weighting": False,
        "protected_unseen_used": False,
        "wilddeepfake_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print("Saved:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
