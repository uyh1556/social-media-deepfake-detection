#!/usr/bin/env python3
"""Recalculate selection-common unseen AUC from saved per-method metrics only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


SUMMARY_POLICY = "selection_common_df40_unseen12"
OUTPUT_FOLDER = "common_unseen12"
FFPP_METHODS = {"Deepfakes", "Face2Face"}


def selection_scope(config: dict, selection: str) -> tuple[set[str], set[str]]:
    all_methods = {
        method for family in config["families"].values() for method in family.values()
    }
    selected = {
        config["families"][family][letter]
        for family in ("FS", "FR", "EFS")
        for letter in config["selections"][selection]
    }
    unseen = all_methods - selected
    if len(all_methods) != 18 or len(selected) != 6 or len(unseen) != 12:
        raise ValueError(f"Invalid selection scope: {selection}")
    return selected, unseen


def summarize_method_rows(rows: list[dict], config: dict, selection: str) -> dict:
    selected, unseen = selection_scope(config, selection)
    df40 = [row for row in rows if row["method"] in selected | unseen]
    if len(df40) != 18 or {row["method"] for row in df40} != selected | unseen:
        raise ValueError(f"Expected exactly one result for each DF40 method: {selection}")
    values = {row["method"]: float(row["roc_auc"]) for row in df40}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
        raise ValueError(f"Invalid method AUC: {selection}")
    for row in rows:
        row["selection_status"] = (
            "common_unseen" if row["method"] in unseen
            else "selection_candidate" if row["method"] in selected
            else "ffpp_reference"
        )

    def status_mean(status: str) -> float:
        group = [float(row["roc_auc"]) for row in rows if row["status"] == status]
        return mean(group) if group else float("nan")

    return {
        "df40_all_macro_auc": mean(values.values()),
        "df40_seen_macro_auc": status_mean("trained"),
        "df40_unseen_macro_auc": mean(values[method] for method in sorted(unseen)),
        "df40_unseen_method_count": len(unseen),
        "df40_unseen_methods": "|".join(sorted(unseen)),
        "df40_model_specific_unseen_macro_auc": status_mean("unseen"),
        "df40_model_specific_unseen_method_count": sum(
            row["status"] == "unseen" for row in df40
        ),
        "ffpp_reference_macro_auc": status_mean("ffpp_reference"),
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recalculate(directory: Path, config: dict) -> Path:
    methods = read_csv(directory / "method_summary.csv")
    models = read_csv(directory / "model_summary.csv")
    if not methods or not models:
        raise ValueError(f"Empty result tables: {directory}")
    keys = ["selection", "protocol", "model"]
    for optional in ("seed", "training_seed", "condition", "evaluation_condition"):
        if optional in methods[0] and optional in models[0]:
            keys.append(optional)
    grouped: dict[tuple, list[dict]] = {}
    for row in methods:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    used = set()
    for row in models:
        key = tuple(row[name] for name in keys)
        if key in used or key not in grouped:
            raise ValueError(f"Ambiguous or missing method group: {directory}: {key}")
        used.add(key)
        row.update(summarize_method_rows(grouped[key], config, row["selection"]))
    if used != set(grouped):
        raise ValueError(f"Method/model summary groups differ: {directory}")
    metadata_path = directory / "evaluation_summary.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metadata.update({
        "summary_policy": SUMMARY_POLICY,
        "summary_source": str(directory),
        "df40_unseen_definition": "18 DF40 methods minus the selection's six M7 methods",
        "ffpp_in_df40_unseen": False,
        "inference_rerun": False,
    })
    output = directory / OUTPUT_FOLDER
    output.mkdir(exist_ok=True)
    write_csv(output / "model_summary.csv", models)
    write_csv(output / "method_summary.csv", methods)
    (output / "evaluation_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/family_rotation_v1/selections.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directories = set()
    for root in args.results_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = [root / "method_summary.csv", *root.rglob("method_summary.csv")]
        for path in candidates:
            if path.is_file() and OUTPUT_FOLDER not in path.relative_to(root).parts:
                directories.add(path.parent)
    if not directories:
        raise FileNotFoundError("No method_summary.csv found in the result roots")
    for directory in sorted(directories):
        print("Recalculated:", recalculate(directory, config), flush=True)
    print(f"Completed {len(directories)} result directories; no inference required.", flush=True)


if __name__ == "__main__":
    main()
