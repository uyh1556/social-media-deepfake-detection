#!/usr/bin/env python3
"""Compose and validate an unsampled experiment pool from extracted modules."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--freeze-m0-ffpp",
        action="store_true",
        help="Freeze and verify the full real + FF++ fake M0 pool.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_root = data_root / "manifests" / "ffpp_df40_global_v1" / "modules"
    output = args.output_manifest.resolve()
    if output.exists():
        raise FileExistsError(output)
    frames = []
    input_hashes = {}
    for module in args.modules:
        path = manifest_root / f"{module}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Module manifest is missing; extract its TAR first: {path}"
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if set(frame["module"]) != {module}:
            raise RuntimeError(f"Module mismatch in {path}")
        frames.append(frame)
        input_hashes[module] = sha256(path)
    combined = pd.concat(frames, ignore_index=True)
    if args.freeze_m0_ffpp:
        if set(args.modules) != {"real", "ffpp_fake"} or len(args.modules) != 2:
            raise ValueError("M0 requires exactly: real ffpp_fake")
        expected_hashes = {
            "real": "e0acd835dbf999a87deeaeb46cebba8ad32a812ce870e70920edde161c2ee6dd",
            "ffpp_fake": "bc04aabc740ffdea51e5ddbdb01b6e45e91b9c14e4dd26823c154d921518f833",
        }
        if input_hashes != expected_hashes:
            raise RuntimeError(
                f"M0 module manifests changed: {input_hashes}"
            )
    if "real" not in set(combined["module"]):
        raise ValueError("Binary training/evaluation requires the real module")
    if combined["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs across selected modules")

    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in combined.itertuples(index=False):
        for source_id in row.source_ids.split("|"):
            source_splits[source_id].add(row.split)
    leakage = sum(len(splits) > 1 for splits in source_splits.values())
    group_leakage = int(combined.groupby("group_id")["split"].nunique().gt(1).sum())
    video_leakage = int(
        combined.groupby(["method", "video_id"])["split"].nunique().gt(1).sum()
    )
    if leakage or group_leakage or video_leakage:
        raise RuntimeError(
            f"Leakage detected: source={leakage}, group={group_leakage}, video={video_leakage}"
        )
    missing = [
        path for path in combined["source_path"]
        if not (data_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Selected module images are missing. First entries: " + ", ".join(missing[:10])
        )

    combined = combined.sort_values(
        ["split", "label", "method", "video_id", "frame_index"]
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    output_hash = sha256(output)
    if args.freeze_m0_ffpp:
        if len(combined) != 94589:
            raise RuntimeError(f"Unexpected M0 image count: {len(combined)}")
        expected_counts = {
            ("train", "original"): 22101,
            ("train", "Deepfakes"): 22037,
            ("train", "Face2Face"): 22098,
            ("val", "original"): 4730,
            ("val", "Deepfakes"): 4707,
            ("val", "Face2Face"): 4731,
            ("test", "original"): 4734,
            ("test", "Deepfakes"): 4715,
            ("test", "Face2Face"): 4736,
        }
        actual_counts = combined.groupby(["split", "method"]).size().to_dict()
        if actual_counts != expected_counts:
            raise RuntimeError(f"Unexpected M0 split counts: {actual_counts}")
        expected_output_hash = (
            "5d129f2de79a839230d7762d5911576be2f625cada43738bb8aa96ae362b1ebc"
        )
        if output_hash != expected_output_hash:
            raise RuntimeError(f"Unexpected M0 manifest SHA-256: {output_hash}")
    config = {
        "protocol": (
            "m0_ffpp_c23_full_pool_v1"
            if args.freeze_m0_ffpp
            else "unsampled_modular_pool_v1"
        ),
        "frozen_full_pool": args.freeze_m0_ffpp,
        "warning": (
            None
            if args.freeze_m0_ffpp
            else (
                "This is an unsampled pool. Do not compare M0-M2 until fixed fake "
                "counts and method quotas are frozen in a separate experiment manifest."
            )
        ),
        "data_root": str(data_root),
        "modules": args.modules,
        "module_manifest_sha256": input_hashes,
        "output_manifest": str(output),
        "output_manifest_sha256": output_hash,
        "images": len(combined),
        "by_split_method": {
            f"{split}:{method}": int(count)
            for (split, method), count in combined.groupby(["split", "method"]).size().items()
        },
        "checks": {
            "source_id_overlap": leakage,
            "group_overlap": group_leakage,
            "method_video_overlap": video_leakage,
            "missing_images": 0,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
