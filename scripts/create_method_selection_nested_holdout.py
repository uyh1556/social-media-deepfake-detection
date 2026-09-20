#!/usr/bin/env python3
"""Freeze nested method-holdout selections and fixed-budget manifests."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "method_selection_nested_holdout_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--transfer-evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    return parser.parse_args()


def stable_rank(seed: int, *parts: str) -> str:
    value = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {path}")
    return config


def validate_source_manifests(directory: Path, config: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    required = {
        "sample_id", "split", "label", "method", "family", "source_path"
    }
    for condition, item in config["source_manifests"].items():
        path = directory / item["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(
                f"Source manifest hash mismatch for {path.name}: {actual}"
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        methods = set(frame.loc[frame["label"] == "fake", "method"])
        if methods != set(item["methods"]):
            raise RuntimeError(f"Unexpected fake methods in {path}: {methods}")
        frames[condition] = frame

    real_sets = [
        frozenset(frame.loc[frame["label"] == "real", "sample_id"])
        for frame in frames.values()
    ]
    if len(set(real_sets)) != 1:
        raise RuntimeError("M1-M3 do not share the same Real pool")
    return frames


def load_transfer_matrices(directory: Path, config: dict) -> tuple[dict, dict]:
    summary_path = directory / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != config["parent_transfer_protocol"]:
        raise ValueError(f"Unexpected parent evaluation: {summary_path}")
    if summary.get("protected_unseen_used") or summary.get("wilddeepfake_used"):
        raise RuntimeError("Selection input crossed the protected evaluation boundary")
    matrices = {}
    hashes = {"evaluation_summary": sha256_file(summary_path)}
    methods = list(config["methods"])
    for condition in config["transfer_selection"]["conditions"]:
        path = directory / f"transfer_matrix_{condition}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = pd.read_csv(path, index_col=0)
        if list(matrix.index) != methods or list(matrix.columns) != methods:
            raise RuntimeError(f"Method order mismatch in {path}")
        matrix = matrix.astype(float)
        if not np.isfinite(matrix.to_numpy()).all():
            raise RuntimeError(f"Non-finite transfer value in {path}")
        if ((matrix < 0) | (matrix > 1)).any().any():
            raise RuntimeError(f"AUC outside [0, 1] in {path}")
        matrices[condition] = matrix
        hashes[condition] = sha256_file(path)
    return matrices, hashes


def deterministic_random(candidates: list[str], outer: str, config: dict) -> tuple[str, ...]:
    seed = int(config["sampling_seed"])
    ranked = sorted(
        candidates,
        key=lambda method: stable_rank(seed, outer, "random3", method),
    )
    return tuple(sorted(ranked[:3]))


def family_balanced(candidates: list[str], outer: str, config: dict) -> tuple[str, ...]:
    seed = int(config["sampling_seed"])
    selected = []
    for family in ("FS", "FR", "EFS"):
        family_methods = [
            method
            for method in candidates
            if config["methods"][method]["family"] == family
        ]
        if not family_methods:
            raise RuntimeError(f"No {family} candidate after holding out {outer}")
        selected.append(
            min(
                family_methods,
                key=lambda method: stable_rank(
                    seed, outer, "family_balanced3", family, method
                ),
            )
        )
    return tuple(sorted(selected))


def score_transfer_subset(
    subset: tuple[str, ...],
    candidates: list[str],
    matrices: dict[str, pd.DataFrame],
    config: dict,
) -> dict:
    proxy_targets = sorted(set(candidates) - set(subset))
    if not proxy_targets:
        raise RuntimeError("Transfer selection requires unselected proxy targets")
    coverages = []
    for condition in config["transfer_selection"]["conditions"]:
        matrix = matrices[condition]
        for target in proxy_targets:
            coverage = max(float(matrix.loc[source, target]) for source in subset)
            coverages.append(
                {
                    "condition": condition,
                    "target": target,
                    "coverage": coverage,
                }
            )
    values = [item["coverage"] for item in coverages]
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "represented_families": len(
            {config["methods"][method]["family"] for method in subset}
        ),
        "proxy_targets": proxy_targets,
        "coverage_details": coverages,
    }


def transfer_selection(
    candidates: list[str],
    matrices: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[tuple[str, ...], dict, list[dict]]:
    scored = []
    for subset in itertools.combinations(sorted(candidates), 3):
        score = score_transfer_subset(subset, candidates, matrices, config)
        scored.append({"methods": subset, **score})
    ranked = sorted(
        scored,
        key=lambda item: (
            -item["minimum"],
            -item["mean"],
            -item["represented_families"],
            item["methods"],
        ),
    )
    winner = ranked[0]
    return tuple(winner["methods"]), winner, ranked


def select_rows(
    pool: pd.DataFrame,
    *,
    split: str,
    label: str,
    method: str,
    count: int,
    seed: int,
) -> pd.DataFrame:
    selected = pool[
        (pool["split"] == split)
        & (pool["label"] == label)
        & (pool["method"] == method)
    ].copy()
    if len(selected) < count:
        raise RuntimeError(
            f"Insufficient {split}/{method}: need={count}, have={len(selected)}"
        )
    selected["_nested_rank"] = selected["sample_id"].map(
        lambda value: stable_rank(seed, split, method, value)
    )
    return (
        selected.sort_values(["_nested_rank", "sample_id"])
        .head(count)
        .drop(columns="_nested_rank")
    )


def manifest_for_subset(
    selected_methods: tuple[str, ...],
    method_pools: dict[str, pd.DataFrame],
    real_pool: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    seed = int(config["sampling_seed"])
    pieces = []
    for split in ("train", "val"):
        budget = config["fixed_budgets"][split]
        fake_per_method = int(budget["fake"]) // len(selected_methods)
        if fake_per_method * len(selected_methods) != int(budget["fake"]):
            raise RuntimeError("Fake budget is not divisible by method count")
        pieces.append(
            select_rows(
                real_pool,
                split=split,
                label="real",
                method="original",
                count=int(budget["real"]),
                seed=seed,
            )
        )
        for method in selected_methods:
            pieces.append(
                select_rows(
                    method_pools[method],
                    split=split,
                    label="fake",
                    method=method,
                    count=fake_per_method,
                    seed=seed,
                )
            )
    manifest = pd.concat(pieces, ignore_index=True).sort_values(
        ["split", "label", "method", "video_id", "frame_index", "sample_id"]
    )
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in nested manifest")
    return manifest


def subset_key(methods: tuple[str, ...], config: dict) -> str:
    slugs = sorted(config["methods"][method]["slug"] for method in methods)
    return f"subset{len(methods)}_" + "_".join(slugs)


def main() -> None:
    args = parse_args()
    config_path = args.protocol_config.resolve()
    config = load_config(config_path)
    source_dir = args.source_manifest_dir.resolve()
    matrix_dir = args.transfer_evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()
    frames = validate_source_manifests(source_dir, config)
    matrices, matrix_hashes = load_transfer_matrices(matrix_dir, config)

    existing_summary_path = output_dir / "manifest_summary.json"
    if existing_summary_path.is_file():
        existing = json.loads(
            existing_summary_path.read_text(encoding="utf-8")
        )
        if existing.get("protocol") != PROTOCOL:
            raise RuntimeError(f"Unexpected existing output: {output_dir}")
        if existing.get("protocol_config_sha256") != sha256_file(config_path):
            raise RuntimeError("Existing nested output used another protocol config")
        if existing.get("parent_transfer_matrix_hashes") != matrix_hashes:
            raise RuntimeError("Existing nested output used another transfer matrix")

    method_pools = {
        method: frames[item["source_condition"]]
        for method, item in config["methods"].items()
    }
    real_pool = frames["M1"]
    all_methods = list(config["methods"])
    plan_rows = []
    selection_details = {}
    manifest_records = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for outer in all_methods:
        candidates = [method for method in all_methods if method != outer]
        transfer_methods, transfer_score, ranked = transfer_selection(
            candidates, matrices, config
        )
        choices = {
            "random3": deterministic_random(candidates, outer, config),
            "family_balanced3": family_balanced(candidates, outer, config),
            "transfer3": transfer_methods,
            "all5": tuple(sorted(candidates)),
        }
        selection_details[outer] = {
            "outer_holdout": outer,
            "candidate_methods": candidates,
            "outer_row_and_column_excluded": True,
            "transfer3_winner": transfer_score,
            "transfer3_ranked_subsets": ranked,
        }
        seen_subsets = {}
        for strategy, methods in choices.items():
            if outer in methods:
                raise RuntimeError(f"Outer method leaked into {strategy}/{outer}")
            key = subset_key(methods, config)
            manifest_name = f"holdout_{config['methods'][outer]['slug']}__{key}__seed42.csv"
            manifest_path = output_dir / manifest_name
            if key not in seen_subsets:
                manifest = manifest_for_subset(
                    methods, method_pools, real_pool, config
                )
                temporary = manifest_path.with_suffix(".csv.tmp")
                manifest.to_csv(temporary, index=False)
                generated_hash = sha256_file(temporary)
                if manifest_path.is_file():
                    if sha256_file(manifest_path) != generated_hash:
                        temporary.unlink()
                        raise FileExistsError(
                            f"Existing manifest differs: {manifest_path}"
                        )
                    temporary.unlink()
                else:
                    temporary.replace(manifest_path)
                seen_subsets[key] = (manifest_name, generated_hash)
                manifest_records[f"{outer}|{key}"] = {
                    "outer_holdout": outer,
                    "run_key": key,
                    "selected_methods": list(methods),
                    "manifest": manifest_name,
                    "manifest_sha256": generated_hash,
                }
            else:
                manifest_name, generated_hash = seen_subsets[key]

            score = (
                score_transfer_subset(methods, candidates, matrices, config)
                if len(methods) < len(candidates)
                else None
            )
            plan_rows.append(
                {
                    "outer_holdout": outer,
                    "outer_family": config["methods"][outer]["family"],
                    "strategy": strategy,
                    "strategy_label": config["strategies"][strategy]["label"],
                    "selected_methods": ";".join(methods),
                    "selected_families": ";".join(
                        config["methods"][method]["family"] for method in methods
                    ),
                    "method_count": len(methods),
                    "run_key": key,
                    "manifest": manifest_name,
                    "manifest_sha256": generated_hash,
                    "proxy_targets": (
                        ";".join(score["proxy_targets"]) if score else ""
                    ),
                    "selection_min_coverage": (
                        score["minimum"] if score else np.nan
                    ),
                    "selection_mean_coverage": (
                        score["mean"] if score else np.nan
                    ),
                }
            )

    plan = pd.DataFrame(plan_rows).sort_values(
        ["outer_holdout", "strategy"]
    )
    plan.to_csv(output_dir / "selection_plan.csv", index=False)
    write_json(output_dir / "selection_details.json", selection_details)
    write_json(
        output_dir / "manifest_summary.json",
        {
            "protocol": PROTOCOL,
            "protocol_config_sha256": sha256_file(config_path),
            "parent_transfer_matrix_hashes": matrix_hashes,
            "outer_holdouts": all_methods,
            "strategies": list(config["strategies"]),
            "fixed_budgets": config["fixed_budgets"],
            "unique_training_runs": len(manifest_records),
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
            "manifests": manifest_records,
        },
    )
    print(plan.to_string(index=False), flush=True)
    print("Unique training runs:", len(manifest_records), flush=True)
    print("Saved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
