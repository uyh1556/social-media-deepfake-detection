#!/usr/bin/env python3
"""Select an equal, video-balanced test subset for every method."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-per-method", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank(seed: int, method: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{method}:{value}".encode()).hexdigest()


def balanced_sample(
    frame: pd.DataFrame,
    method: str,
    count: int,
    seed: int,
) -> pd.DataFrame:
    method_frame = frame[frame["method"] == method]
    if len(method_frame) < count:
        raise ValueError(f"{method}: requested {count}, available {len(method_frame)}")

    queues: list[deque[int]] = []
    for video_id, group in method_frame.groupby("video_id", sort=False):
        indices = sorted(
            group.index,
            key=lambda index: rank(seed, method, str(group.at[index, "sample_id"])),
        )
        queues.append(deque(indices))
    queues.sort(
        key=lambda queue: rank(
            seed,
            method,
            str(method_frame.at[queue[0], "video_id"]),
        )
    )

    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for queue in queues:
            if queue and len(selected) < count:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            raise RuntimeError(f"{method}: exhausted candidates")
    return frame.loc[selected]


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.global_manifest, dtype=str, keep_default_na=False)
    frame = frame[frame["split"] == "test"].copy()
    methods = sorted(frame["method"].unique())
    selected = pd.concat(
        [
            balanced_sample(
                frame,
                method,
                args.images_per_method,
                args.seed,
            )
            for method in methods
        ],
        ignore_index=True,
    ).sort_values(["method", "video_id", "sample_id"])

    counts = selected.groupby("method").size().to_dict()
    if set(counts.values()) != {args.images_per_method}:
        raise RuntimeError(counts)
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id in balanced test manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    metadata = {
        "protocol": "family_coverage_balanced_test_2000_v1",
        "seed": args.seed,
        "images_per_method": args.images_per_method,
        "methods": methods,
        "method_count": len(methods),
        "images": len(selected),
        "selection": "deterministic round-robin across logical video_id",
        "counts": counts,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
