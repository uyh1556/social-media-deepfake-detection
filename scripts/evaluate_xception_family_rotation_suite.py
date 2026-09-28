#!/usr/bin/env python3
"""Run the rotation 3-seed canonical and native-Wild evaluation suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=[f"S{i}" for i in range(1, 7)])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--protocols", nargs="+", choices=["fixed_q95", "mixed_jpeg"], default=["fixed_q95", "mixed_jpeg"])
    p.add_argument("--models", nargs="+", choices=[f"M{i}" for i in range(1, 8)], default=[f"M{i}" for i in range(1, 8)])
    p.add_argument("--datasets", nargs="+", choices=["canonical", "wild"], default=["canonical", "wild"])
    for name in ("data-root", "df40-test-manifest", "ffpp-test-manifest", "rotation-manifest-root", "s1-manifest-root", "rotation-runs-root", "s1-fixed-runs-root", "s1-mixed-runs-root", "canonical-output-root", "wild-data-root", "wild-manifest", "wild-output-root", "summary-output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--reuse-s1-wild-root", type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--summaries-only", action="store_true")
    args = p.parse_args()
    for values in (args.selections, args.seeds, args.protocols, args.models, args.datasets):
        if len(values) != len(set(values)):
            p.error("Requested selections, seeds, protocols, models and datasets must be unique")
    return args


def aggregate(frame, group_columns, metrics):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["training_seeds"] = "|".join(map(str, sorted(group.training_seed)))
        row["seed_runs"] = len(group)
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame):
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in values
        ) + " |")
    return "\n".join(lines)


def summarize(args):
    args.summary_output_root.mkdir(parents=True, exist_ok=True)
    canonical, method_tables, wild = [], [], []
    expected = len(args.selections) * len(args.seeds) * len(args.protocols) * len(args.models)
    for selection in args.selections:
        if "canonical" in args.datasets:
            for seed in args.seeds:
                folder = args.canonical_output_root / f"{selection.lower()}_seed{seed}_21groups_q95"
                models = pd.read_csv(folder / "model_summary.csv")
                models = models[models.protocol.isin(args.protocols) & models.model.isin(args.models)].copy()
                if not (models.df40_unseen_method_count == 12).all():
                    raise RuntimeError(f"Canonical summaries are not common-unseen12: {folder}")
                models["training_seed"] = seed
                canonical.append(models)
                methods = pd.read_csv(folder / "method_summary.csv")
                methods = methods[methods.protocol.isin(args.protocols) & methods.model.isin(args.models)].copy()
                methods["training_seed"] = seed
                method_tables.append(methods)
        if "wild" in args.datasets:
            rows = pd.read_csv(args.wild_output_root / selection.lower() / "model_seed_summary.csv")
            rows = rows[rows.protocol.isin(args.protocols) & rows.model.isin(args.models) & rows.training_seed.isin(args.seeds)].copy()
            wild.append(rows)
    report = ["# Family-rotation evaluation", "", "Seed standard deviations describe training-seed variation, not test confidence intervals.", ""]
    for name, tables, metrics in (
        ("canonical", canonical, ["real_fpr", "df40_all_macro_auc", "df40_seen_macro_auc", "df40_unseen_macro_auc", "ffpp_reference_macro_auc"]),
        ("wild", wild, ["sequence_roc_auc", "sequence_auprc", "sequence_real_fpr", "sequence_fake_recall", "frame_roc_auc"]),
    ):
        if not tables:
            continue
        frame = pd.concat(tables, ignore_index=True)
        keys = ["selection", "protocol", "model", "training_seed"]
        if len(frame) != expected or frame.duplicated(keys).any():
            raise RuntimeError(f"Incomplete/duplicate {name} results: expected {expected}, found {len(frame)}")
        means = aggregate(frame, ["selection", "protocol", "model"], metrics)
        frame.to_csv(args.summary_output_root / f"{name}_model_seed_summary.csv", index=False)
        means.to_csv(args.summary_output_root / f"{name}_seed_aggregate_summary.csv", index=False)
        headline = "df40_unseen_macro_auc" if name == "canonical" else "sequence_roc_auc"
        report.extend([f"## {name}", "", markdown_table(means[["selection", "protocol", "model", "training_seeds", f"{headline}_mean", f"{headline}_std"]]), ""])
    if method_tables:
        methods = pd.concat(method_tables, ignore_index=True)
        methods.to_csv(args.summary_output_root / "canonical_method_seed_summary.csv", index=False)
        aggregate(methods, ["selection", "protocol", "model", "method", "family", "selection_status"], ["roc_auc", "fake_recall", "real_fpr"]).to_csv(args.summary_output_root / "canonical_method_seed_aggregate_summary.csv", index=False)
    (args.summary_output_root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (args.summary_output_root / "evaluation_summary.json").write_text(json.dumps({
        "selections": args.selections, "training_seeds": args.seeds,
        "models": args.models, "training_protocols": args.protocols,
        "datasets": args.datasets, "checkpoint_runs_per_dataset": expected,
        "canonical_preprocessing": "RGB -> Letterbox256 -> JPEG Q95 -> Letterbox299 -> Normalize",
        "df40_unseen_definition": "selection-common 12 DF40 methods",
        "ffpp_reference_separate": True,
        "wild_preprocessing": "native RGB -> Letterbox299 -> Normalize",
        "wild_primary_unit": "806 sequences; mean over 16 frames each",
        "wild_external_exploratory": True,
    }, indent=2) + "\n", encoding="utf-8")
    print("Combined summaries saved:", args.summary_output_root, flush=True)


def main():
    args = parse_args()
    scripts = Path(__file__).resolve().parent
    common = []
    for name in ("rotation-manifest-root", "s1-manifest-root", "rotation-runs-root", "s1-fixed-runs-root", "s1-mixed-runs-root"):
        common.extend([f"--{name}", str(getattr(args, name.replace("-", "_")))])
    common.extend(["--protocols", *args.protocols, "--models", *args.models, "--batch-size", str(args.batch_size), "--workers", str(args.workers)])
    if not args.summaries_only:
        for selection in args.selections:
            if "canonical" in args.datasets:
                for seed in args.seeds:
                    print(f"\n===== 21 groups: {selection} / seed {seed} =====", flush=True)
                    command = [sys.executable, "-u", str(scripts / "evaluate_xception_family_rotation.py"),
                        "--selection", selection, "--seed", str(seed),
                        "--data-root", str(args.data_root),
                        "--df40-test-manifest", str(args.df40_test_manifest),
                        "--ffpp-test-manifest", str(args.ffpp_test_manifest),
                        "--output-root", str(args.canonical_output_root / f"{selection.lower()}_seed{seed}_21groups_q95"), *common]
                    subprocess.run(command, check=True)
            if "wild" in args.datasets:
                print(f"\n===== WildDeepfake: {selection} / seeds {args.seeds} =====", flush=True)
                command = [sys.executable, "-u", str(scripts / "evaluate_xception_family_rotation_wilddeepfake.py"),
                    "--selection", selection, "--seeds", *map(str, args.seeds),
                    "--data-root", str(args.wild_data_root), "--manifest", str(args.wild_manifest),
                    "--output-root", str(args.wild_output_root / selection.lower()), *common]
                if args.reuse_s1_wild_root is not None:
                    command.extend(["--reuse-s1-root", str(args.reuse_s1_wild_root)])
                subprocess.run(command, check=True)
    summarize(args)


if __name__ == "__main__":
    main()
