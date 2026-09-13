#!/usr/bin/env python3
"""Create a fixed-count manipulation-diversity training condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--fake-methods", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-reference-sha256",
        default=None,
        help="Fail if the reference manifest is not the frozen expected file.",
    )
    parser.add_argument(
        "--seen-eval-manifest",
        type=Path,
        default=None,
        help="Optional test manifest containing reference real plus seen methods.",
    )
    parser.add_argument(
        "--seen-eval-methods",
        nargs="+",
        default=None,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "sample_id",
        "split",
        "label",
        "method",
        "group_id",
        "video_id",
        "source_ids",
        "frame_index",
        "source_path",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    return frame


def stable_rank(sample_id: str, seed: int, split: str, method: str) -> str:
    payload = f"{seed}|{split}|{method}|{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def equal_quotas(total: int, methods: list[str]) -> dict[str, int]:
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("--fake-methods must be unique and non-empty")
    base, remainder = divmod(total, len(methods))
    return {
        method: base + int(index < remainder)
        for index, method in enumerate(methods)
    }


def deterministic_sample(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    split: str,
    method: str,
) -> pd.DataFrame:
    if len(frame) < count:
        raise RuntimeError(
            f"Not enough {split}/{method} rows: need {count}, have {len(frame)}"
        )
    ranked = frame.assign(
        _sample_rank=frame["sample_id"].map(
            lambda sample_id: stable_rank(sample_id, seed, split, method)
        )
    ).sort_values(["_sample_rank", "sample_id"])
    return ranked.head(count).drop(columns="_sample_rank")


def leakage_counts(frame: pd.DataFrame) -> dict[str, int]:
    source_splits: dict[str, set[str]] = defaultdict(set)
    driver_splits: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        for source_id in str(row.source_ids).replace(",", "|").split("|"):
            if source_id.strip():
                source_splits[source_id.strip()].add(row.split)
        driver_id = getattr(row, "driver_id", "")
        if str(driver_id).strip():
            driver_splits[str(driver_id).strip()].add(row.split)
    return {
        "source_id_overlap": sum(len(value) > 1 for value in source_splits.values()),
        "driver_id_overlap": sum(len(value) > 1 for value in driver_splits.values()),
        "group_overlap": int(
            frame.groupby("group_id")["split"].nunique().gt(1).sum()
        ),
        "method_video_overlap": int(
            frame.groupby(["method", "video_id"])["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
    }


def sorted_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["split", "label", "method", "video_id", "frame_index", "sample_id"]
    ).reset_index(drop=True)


def count_map(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"{split}:{method}": int(count)
        for (split, method), count in frame.groupby(["split", "method"]).size().items()
    }


def main() -> None:
    args = parse_args()
    pool_path = args.pool_manifest.resolve()
    reference_path = args.reference_manifest.resolve()
    output_path = args.output_manifest.resolve()
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(output_path)
    if args.seen_eval_manifest is not None:
        args.seen_eval_manifest = args.seen_eval_manifest.resolve()
        if args.seen_eval_manifest.exists():
            raise FileExistsError(args.seen_eval_manifest)
        if (args.seen_eval_manifest.parent / "config.json").exists():
            raise FileExistsError(args.seen_eval_manifest.parent / "config.json")

    pool = read_manifest(pool_path)
    reference = read_manifest(reference_path)
    pool_hash = sha256_file(pool_path)
    reference_hash = sha256_file(reference_path)
    if (
        args.expected_reference_sha256 is not None
        and reference_hash != args.expected_reference_sha256
    ):
        raise RuntimeError(
            "Reference manifest SHA-256 mismatch: "
            f"{reference_hash} != {args.expected_reference_sha256}"
        )

    expected_reference_methods = {"original", "Deepfakes", "Face2Face"}
    if set(reference["method"]) != expected_reference_methods:
        raise RuntimeError(
            f"Unexpected reference methods: {sorted(set(reference['method']))}"
        )
    if set(args.fake_methods) - set(pool["method"]):
        missing = set(args.fake_methods) - set(pool["method"])
        raise RuntimeError(f"Fake methods missing from pool: {sorted(missing)}")
    pool_fake_methods = set(pool[pool["label"] == "fake"]["method"])
    if pool_fake_methods != set(args.fake_methods):
        raise RuntimeError(
            "Pool fake methods do not exactly match --fake-methods: "
            f"{sorted(pool_fake_methods)}"
        )
    if set(pool[pool["label"] == "real"]["method"]) != {"original"}:
        raise RuntimeError("The pool must use only method=original for real rows")

    selected_parts: list[pd.DataFrame] = []
    quotas: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        reference_real = reference[
            (reference["split"] == split) & (reference["label"] == "real")
        ]
        pool_real_ids = set(
            pool[(pool["split"] == split) & (pool["label"] == "real")]["sample_id"]
        )
        if not set(reference_real["sample_id"]).issubset(pool_real_ids):
            raise RuntimeError(f"Reference real rows missing from pool for {split}")
        selected_parts.append(reference_real.copy())

        target_fake_total = int(
            ((reference["split"] == split) & (reference["label"] == "fake")).sum()
        )
        split_quotas = equal_quotas(target_fake_total, args.fake_methods)
        quotas[split] = split_quotas
        for method in args.fake_methods:
            candidates = pool[
                (pool["split"] == split)
                & (pool["label"] == "fake")
                & (pool["method"] == method)
            ]
            selected_parts.append(
                deterministic_sample(
                    candidates,
                    split_quotas[method],
                    args.seed,
                    split,
                    method,
                )
            )

    # All conditions use the exact same FF++ test rows as M0.
    reference_test = reference[reference["split"] == "test"].copy()
    selected_parts.append(reference_test)
    output = sorted_manifest(pd.concat(selected_parts, ignore_index=True))

    if output["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in output")
    checks = leakage_counts(output)
    if any(checks.values()):
        raise RuntimeError(f"Leakage detected: {checks}")
    for split in ("train", "val"):
        actual_real = set(
            output[(output["split"] == split) & (output["label"] == "real")]["sample_id"]
        )
        reference_real = set(
            reference[
                (reference["split"] == split) & (reference["label"] == "real")
            ]["sample_id"]
        )
        if actual_real != reference_real:
            raise RuntimeError(f"Real pool changed in {split}")
        actual_fake_total = int(
            ((output["split"] == split) & (output["label"] == "fake")).sum()
        )
        reference_fake_total = int(
            ((reference["split"] == split) & (reference["label"] == "fake")).sum()
        )
        if actual_fake_total != reference_fake_total:
            raise RuntimeError(f"Fake total changed in {split}")
    if set(output[output["split"] == "test"]["sample_id"]) != set(
        reference_test["sample_id"]
    ):
        raise RuntimeError("Common M0 FF++ test changed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    output_hash = sha256_file(output_path)
    config = {
        "protocol": "manipulation_diversity_fixed_count_global_v1",
        "condition_name": args.condition_name,
        "seed": args.seed,
        "sampling": "SHA-256 rank without replacement within split and method",
        "fake_methods": args.fake_methods,
        "fake_quotas": quotas,
        "real_policy": "exact M0 train/validation rows",
        "test_policy": "exact M0 FF++ test rows",
        "pool_manifest": str(pool_path),
        "pool_manifest_sha256": pool_hash,
        "reference_manifest": str(reference_path),
        "reference_manifest_sha256": reference_hash,
        "output_manifest": str(output_path),
        "output_manifest_sha256": output_hash,
        "images": len(output),
        "by_split_method": count_map(output),
        "checks": checks,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    seen_summary = None
    if args.seen_eval_manifest is not None:
        methods = args.seen_eval_methods or []
        if not methods:
            raise ValueError("--seen-eval-methods is required with --seen-eval-manifest")
        if set(methods) - set(args.fake_methods):
            raise ValueError("Seen evaluation methods must be training fake methods")
        seen_real = reference[
            (reference["split"] == "test") & (reference["label"] == "real")
        ]
        seen_fake = pool[
            (pool["split"] == "test")
            & (pool["label"] == "fake")
            & (pool["method"].isin(methods))
        ]
        seen = sorted_manifest(pd.concat([seen_real, seen_fake], ignore_index=True))
        seen_checks = leakage_counts(seen)
        if any(seen_checks.values()):
            raise RuntimeError(f"Seen-evaluation leakage detected: {seen_checks}")
        args.seen_eval_manifest.parent.mkdir(parents=True, exist_ok=True)
        seen.to_csv(args.seen_eval_manifest, index=False)
        seen_hash = sha256_file(args.seen_eval_manifest)
        seen_summary = {
            "protocol": "global_source_disjoint_seen_manipulation_test_v1",
            "source_manifest": str(output_path),
            "source_manifest_sha256": output_hash,
            "pool_manifest_sha256": pool_hash,
            "methods": methods,
            "manifest": str(args.seen_eval_manifest),
            "manifest_sha256": seen_hash,
            "images": len(seen),
            "by_split_method": count_map(seen),
            "checks": seen_checks,
        }
        (args.seen_eval_manifest.parent / "config.json").write_text(
            json.dumps(seen_summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"training": config, "seen_evaluation": seen_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
