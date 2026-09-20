#!/usr/bin/env python3
"""Create a six-method M7 manifest with training-side transfer weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "transfer_weighted_mixed_jpeg_pilot_v1"


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


def stable_rank(seed: int, *parts: str) -> str:
    value = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


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


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {path}")
    weighting = config["weighting"]
    if weighting.get("protected_unseen_used"):
        raise RuntimeError("Protected unseen data cannot determine weights")
    if weighting.get("wilddeepfake_used"):
        raise RuntimeError("WildDeepfake cannot determine weights")
    return config


def load_source_manifests(
    directory: Path, config: dict
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    required = {
        "sample_id", "split", "label", "method", "family", "source_path"
    }
    frames = {}
    for condition, item in config["source_manifests"].items():
        path = directory / item["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            raise ValueError(
                f"Source manifest hash mismatch for {path.name}: {actual_hash}"
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
    reference_item = config["reference_manifest"]
    reference_path = directory / reference_item["file"]
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if sha256_file(reference_path) != reference_item["sha256"]:
        raise ValueError("M7 reference manifest hash mismatch")
    reference = pd.read_csv(
        reference_path, dtype=str, keep_default_na=False
    )
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {reference_path}: {sorted(missing)}"
        )
    expected_methods = set(config["methods"])
    actual_methods = set(
        reference.loc[reference["label"] == "fake", "method"]
    )
    if actual_methods != expected_methods:
        raise RuntimeError("M7 reference does not contain the six methods")
    return frames, reference


def load_transfer_matrices(
    directory: Path, config: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    summary_path = directory / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != config["parent_transfer_protocol"]:
        raise ValueError(f"Unexpected transfer evaluation: {summary_path}")
    if summary.get("protected_unseen_used") or summary.get("wilddeepfake_used"):
        raise RuntimeError("Transfer evaluation crossed the protected boundary")

    methods = list(config["methods"])
    matrices = {}
    hashes = {"evaluation_summary": sha256_file(summary_path)}
    for condition in config["weighting"]["conditions"]:
        path = directory / f"transfer_matrix_{condition}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = pd.read_csv(path, index_col=0)
        if list(matrix.index) != methods or list(matrix.columns) != methods:
            raise RuntimeError(f"Method order mismatch in {path}")
        matrix = matrix.astype(float)
        values = matrix.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError(f"Invalid AUC matrix: {path}")
        matrices[condition] = matrix
        hashes[condition] = sha256_file(path)
    return matrices, hashes


def method_utilities(
    matrices: dict[str, pd.DataFrame], config: dict
) -> pd.DataFrame:
    methods = list(config["methods"])
    chance = float(config["weighting"]["chance_auc"])
    rows = []
    for source in methods:
        components = []
        for condition, matrix in matrices.items():
            for target in methods:
                if source == target:
                    continue
                auc = float(matrix.loc[source, target])
                components.append(max(auc - chance, 0.0))
        utility = float(np.mean(components))
        rows.append(
            {
                "method": source,
                "family": config["methods"][source]["family"],
                "utility": utility,
                "component_count": len(components),
            }
        )
    frame = pd.DataFrame(rows)
    if (frame["utility"] <= 0).any():
        failed = frame.loc[frame["utility"] <= 0, "method"].tolist()
        raise RuntimeError(
            "All methods must retain positive transfer utility: " + ", ".join(failed)
        )
    frame["weight"] = frame["utility"] / frame["utility"].sum()
    return frame


def largest_remainder(
    weights: pd.DataFrame, total: int
) -> dict[str, int]:
    exact = weights.set_index("method")["weight"] * total
    quota = np.floor(exact).astype(int)
    remainder = int(total - quota.sum())
    order = sorted(
        exact.index,
        key=lambda method: (-(exact[method] - quota[method]), method),
    )
    for method in order[:remainder]:
        quota[method] += 1
    result = {method: int(quota[method]) for method in exact.index}
    if sum(result.values()) != total or min(result.values()) <= 0:
        raise RuntimeError(f"Invalid quota allocation: {result}")
    return result


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
    selected["_weighted_rank"] = selected["sample_id"].map(
        lambda value: stable_rank(seed, split, method, value)
    )
    return (
        selected.sort_values(["_weighted_rank", "sample_id"])
        .head(count)
        .drop(columns="_weighted_rank")
    )


def build_manifest(
    frames: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    weights: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    seed = int(config["training_manifest_sampling_seed"])
    method_pools = {
        method: frames[item["source_condition"]]
        for method, item in config["methods"].items()
    }
    pieces = []
    quotas = {}
    train_budget = config["fixed_budgets"]["train"]
    train_real = reference[
        (reference["split"] == "train") & (reference["label"] == "real")
    ].copy()
    if len(train_real) != int(train_budget["real"]):
        raise RuntimeError("Unexpected M7 reference Real training count")
    pieces.append(train_real)
    quotas["train"] = largest_remainder(weights, int(train_budget["fake"]))
    for method, count in quotas["train"].items():
        pieces.append(
            select_rows(
                method_pools[method],
                split="train",
                label="fake",
                method=method,
                count=count,
                seed=seed,
            )
        )

    validation = reference[reference["split"] == "val"].copy()
    val_budget = config["fixed_budgets"]["val"]
    val_counts = validation.groupby("label").size().to_dict()
    if val_counts != val_budget:
        raise RuntimeError(
            f"Unexpected M7 reference validation budget: {val_counts}"
        )
    quotas["val"] = (
        validation[validation["label"] == "fake"]
        .groupby("method")
        .size()
        .astype(int)
        .to_dict()
    )
    if set(quotas["val"]) != set(config["methods"]):
        raise RuntimeError("M7 reference validation methods are incomplete")
    if len(set(quotas["val"].values())) != 1:
        raise RuntimeError("M7 reference validation is not method-balanced")
    pieces.append(validation)
    manifest = pd.concat(pieces, ignore_index=True).sort_values(
        ["split", "label", "method", "video_id", "frame_index", "sample_id"]
    )
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in weighted manifest")
    return manifest, quotas


def main() -> None:
    args = parse_args()
    config_path = args.protocol_config.resolve()
    config = load_config(config_path)
    source_dir = args.source_manifest_dir.resolve()
    transfer_dir = args.transfer_evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()

    frames, reference = load_source_manifests(source_dir, config)
    matrices, matrix_hashes = load_transfer_matrices(transfer_dir, config)
    weights = method_utilities(matrices, config)
    manifest, quotas = build_manifest(frames, reference, weights, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "m7_transfer_weighted_seed42.csv"
    temporary = manifest_path.with_suffix(".csv.tmp")
    manifest.to_csv(temporary, index=False)
    generated_hash = sha256_file(temporary)
    if manifest_path.is_file():
        if sha256_file(manifest_path) != generated_hash:
            temporary.unlink()
            raise FileExistsError(f"Existing manifest differs: {manifest_path}")
        temporary.unlink()
    else:
        temporary.replace(manifest_path)

    allocation = weights.copy()
    allocation["train_images"] = allocation["method"].map(quotas["train"])
    allocation["val_images"] = allocation["method"].map(quotas["val"])
    allocation.to_csv(output_dir / "method_allocation.csv", index=False)
    write_json(
        output_dir / "manifest_summary.json",
        {
            "protocol": PROTOCOL,
            "protocol_config_sha256": sha256_file(config_path),
            "parent_transfer_matrix_hashes": matrix_hashes,
            "manifest": manifest_path.name,
            "manifest_sha256": generated_hash,
            "fixed_budgets": config["fixed_budgets"],
            "method_quotas": quotas,
            "weighting": config["weighting"],
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
        },
    )
    print(allocation.to_string(index=False), flush=True)
    print("Manifest SHA-256:", generated_hash, flush=True)
    print("Saved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
