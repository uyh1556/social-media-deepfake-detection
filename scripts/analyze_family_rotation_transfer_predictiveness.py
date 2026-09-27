#!/usr/bin/env python3
"""Test whether rotation transfer scores predict baseline AUC or weighting gains."""

import argparse
import json
from pathlib import Path

import pandas as pd


CONDITIONS = (
    "canonical_256_jpeg_q95",
    "canonical_256_jpeg_q90",
    "canonical_256_jpeg_q75",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer-root", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--comparison-dir", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def unique_dir(root, pattern):
    found = sorted(root.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected one match for {root / pattern}: {found}")
    return found[0]


def main():
    args = parse_args()
    edges = []
    for n in range(1, 7):
        selection = f"S{n}"
        root = unique_dir(args.transfer_root, f"s{n}*_q95_q90_q75")
        for condition in CONDITIONS:
            matrix = pd.read_csv(root / f"transfer_matrix_{condition}.csv", index_col=0)
            for source in matrix.index:
                for target in matrix.columns:
                    if source != target:
                        edges.append({"selection_observed": selection, "condition": condition,
                                      "source": source, "target": target,
                                      "transfer_auc": float(matrix.loc[source, target])})
    edge_frame = pd.DataFrame(edges)
    # Average duplicate observations and JPEG conditions into one stable edge.
    edge_mean = edge_frame.groupby(["source", "target"], as_index=False).agg(
        transfer_auc=("transfer_auc", "mean"), observations=("transfer_auc", "size")
    )
    observed_delta = pd.read_csv(args.comparison_dir / "method_auc_deltas.csv")
    rows = []
    for n in range(1, 7):
        selection = f"S{n}"
        weights = json.loads((args.weights_dir / f"s{n}_weights.json").read_text())["weights"]
        baseline_dir = unique_dir(args.baseline_root, f"s{n}_seed42_21groups_q95")
        base = pd.read_csv(baseline_dir / "method_summary.csv")
        base = base[(base.model == "M7") & (base.protocol == "mixed_jpeg")]
        deltas = observed_delta[observed_delta.selection == selection]
        for _, target_row in base.iterrows():
            target = target_row.method
            if target_row.status != "unseen":
                continue
            available = edge_mean[
                edge_mean.source.isin(weights) & (edge_mean.target == target)
            ].copy()
            if available.empty:
                continue
            available["weight"] = available.source.map(weights)
            uniform_support = available.transfer_auc.mean()
            weighted_support = (
                (available.transfer_auc * available.weight).sum()
                / available.weight.sum()
            )
            delta_match = deltas[deltas.method == target]
            if len(delta_match) != 1:
                raise RuntimeError(f"Missing observed delta: {selection}/{target}")
            rows.append({
                "selection": selection, "target_method": target,
                "family": target_row.family, "edge_coverage": len(available),
                "uniform_transfer_support": uniform_support,
                "weighted_transfer_support": weighted_support,
                "predicted_support_delta": weighted_support - uniform_support,
                "baseline_auc": float(target_row.roc_auc),
                "observed_auc_delta": float(delta_match.iloc[0].delta_auc),
            })
    analysis = pd.DataFrame(rows)
    eligible = analysis[analysis.edge_coverage >= 2].copy()
    tests = [
        ("uniform_support_vs_baseline_auc", "uniform_transfer_support", "baseline_auc"),
        ("predicted_support_delta_vs_observed_auc_delta", "predicted_support_delta", "observed_auc_delta"),
    ]
    correlations = []
    for name, x, y in tests:
        correlations.append({
            "analysis": name, "n": len(eligible),
            "spearman_rho": eligible[x].corr(eligible[y], method="spearman"),
            "pearson_r": eligible[x].corr(eligible[y], method="pearson"),
        })
    correlations = pd.DataFrame(correlations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_frame.to_csv(args.output_dir / "observed_transfer_edges.csv", index=False)
    edge_mean.to_csv(args.output_dir / "aggregated_transfer_edges.csv", index=False)
    analysis.to_csv(args.output_dir / "target_support_analysis.csv", index=False)
    correlations.to_csv(args.output_dir / "correlations.csv", index=False)
    report = ["# Transfer predictiveness diagnostic", "", correlations.to_markdown(index=False, floatfmt=".6f"), "", "Only unseen targets with at least two observed source-target edges are included."]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(correlations.to_string(index=False), flush=True)
    print("Eligible target cases:", len(eligible), "/", len(analysis), flush=True)


if __name__ == "__main__":
    main()
