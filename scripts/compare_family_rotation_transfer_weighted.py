#!/usr/bin/env python3
"""Compare S1-S6 transfer-loss-weighted M7 against Mixed-JPEG M7."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "real_fpr", "df40_all_macro_auc", "df40_seen_macro_auc",
    "df40_unseen_macro_auc", "ffpp_reference_macro_auc",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--weighted-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def one_dir(root: Path, pattern: str) -> Path:
    found = sorted(root.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected one directory for {root / pattern}: {found}")
    return found[0]


def main():
    args = parse_args()
    comparisons, methods = [], []
    for index in range(1, 7):
        selection = f"S{index}"
        baseline_dir = one_dir(args.baseline_root, f"s{index}_seed42_21groups_q95")
        weighted_dir = one_dir(args.weighted_root, f"s{index}_seed42")
        baseline = pd.read_csv(baseline_dir / "model_summary.csv")
        baseline = baseline[(baseline.model == "M7") & (baseline.protocol == "mixed_jpeg")]
        weighted = pd.read_csv(weighted_dir / "model_summary.csv")
        weighted = weighted[(weighted.model == "M7") & (weighted.protocol == "transfer_weighted")]
        if len(baseline) != 1 or len(weighted) != 1:
            raise RuntimeError(f"Missing comparison rows for {selection}")
        row = {"selection": selection}
        for metric in METRICS:
            b, w = float(baseline.iloc[0][metric]), float(weighted.iloc[0][metric])
            row[f"baseline_{metric}"] = b
            row[f"weighted_{metric}"] = w
            row[f"delta_{metric}"] = w - b
        comparisons.append(row)

        base_methods = pd.read_csv(baseline_dir / "method_summary.csv")
        base_methods = base_methods[(base_methods.model == "M7") & (base_methods.protocol == "mixed_jpeg")]
        weighted_methods = pd.read_csv(weighted_dir / "method_summary.csv")
        weighted_methods = weighted_methods[(weighted_methods.model == "M7") & (weighted_methods.protocol == "transfer_weighted")]
        joined = base_methods.merge(weighted_methods, on=["selection", "model", "method", "family", "status"], suffixes=("_baseline", "_weighted"))
        for _, item in joined.iterrows():
            methods.append({
                "selection": selection, "method": item.method,
                "family": item.family, "status": item.status,
                "baseline_auc": item.roc_auc_baseline,
                "weighted_auc": item.roc_auc_weighted,
                "delta_auc": item.roc_auc_weighted - item.roc_auc_baseline,
            })

    comparison = pd.DataFrame(comparisons)
    method_delta = pd.DataFrame(methods)
    aggregate = []
    for metric in METRICS:
        delta = comparison[f"delta_{metric}"]
        higher_is_better = metric != "real_fpr"
        wins = int((delta > 0).sum()) if higher_is_better else int((delta < 0).sum())
        aggregate.append({
            "metric": metric, "mean_baseline": comparison[f"baseline_{metric}"].mean(),
            "mean_weighted": comparison[f"weighted_{metric}"].mean(),
            "mean_delta": delta.mean(), "median_delta": delta.median(),
            "weighted_wins_out_of_6": wins,
        })
    aggregate = pd.DataFrame(aggregate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "selection_comparison.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_comparison.csv", index=False)
    method_delta.to_csv(args.output_dir / "method_auc_deltas.csv", index=False)
    report = ["# Transfer-loss-weighted M7 comparison", "", "## Aggregate", "", aggregate.to_markdown(index=False, floatfmt=".6f"), "", "## By selection", "", comparison.to_markdown(index=False, floatfmt=".6f"), ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)
    print("\nSaved:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
