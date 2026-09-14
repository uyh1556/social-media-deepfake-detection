#!/usr/bin/env python3
"""Create fixed-budget M0-M7 train/validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CONDITIONS = {
    "M0": ("Deepfakes", "Face2Face"),
    "M1": ("SimSwap", "BlendFace"),
    "M2": ("Wav2Lip", "FOMM"),
    "M3": ("StyleGAN3", "DiT"),
    "M4": ("SimSwap", "BlendFace", "Wav2Lip", "FOMM"),
    "M5": ("SimSwap", "BlendFace", "StyleGAN3", "DiT"),
    "M6": ("Wav2Lip", "FOMM", "StyleGAN3", "DiT"),
    "M7": ("SimSwap", "BlendFace", "Wav2Lip", "FOMM", "StyleGAN3", "DiT"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-real", type=int, default=19800)
    parser.add_argument("--train-fake", type=int, default=19800)
    parser.add_argument("--val-real", type=int, default=4680)
    parser.add_argument("--val-fake", type=int, default=4680)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(sample_id: str, seed: int, split: str, method: str) -> str:
    value = f"{seed}|{split}|{method}|{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select(frame: pd.DataFrame, count: int, seed: int, split: str, method: str) -> pd.DataFrame:
    if len(frame) < count:
        raise RuntimeError(
            f"Not enough {split}/{method}: need={count}, available={len(frame)}"
        )
    return (
        frame.assign(
            _rank=frame["sample_id"].map(
                lambda value: stable_rank(value, seed, split, method)
            )
        )
        .sort_values(["_rank", "sample_id"])
        .head(count)
        .drop(columns="_rank")
    )


def main() -> None:
    args = parse_args()
    source = args.global_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if args.train_fake % 12 or args.val_fake % 12:
        raise ValueError("Fake budgets must be divisible by 12 for 2/4/6 methods")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    development = frame[frame["split"].isin(["train", "val"])].copy()
    if (development["role"] == "protected_unseen").any():
        raise RuntimeError("Protected unseen rows entered train/validation pool")
    output_dir.mkdir(parents=True, exist_ok=True)

    real_selected: dict[str, pd.DataFrame] = {}
    for split, count in (("train", args.train_real), ("val", args.val_real)):
        candidates = development[
            (development["split"] == split) & (development["label"] == "real")
        ]
        real_selected[split] = select(candidates, count, args.seed, split, "original")

    summaries = {}
    for condition_id, fake_methods in CONDITIONS.items():
        pieces = []
        quotas = {}
        for split, fake_total in (("train", args.train_fake), ("val", args.val_fake)):
            pieces.append(real_selected[split])
            per_method = fake_total // len(fake_methods)
            quotas[split] = {method: per_method for method in fake_methods}
            for method in fake_methods:
                candidates = development[
                    (development["split"] == split)
                    & (development["label"] == "fake")
                    & (development["method"] == method)
                ]
                pieces.append(
                    select(candidates, per_method, args.seed, split, method)
                )
        selected = pd.concat(pieces, ignore_index=True).sort_values(
            ["split", "label", "method", "video_id", "frame_index", "sample_id"]
        )
        if selected["sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample IDs in {condition_id}")
        if set(selected["method"]) != {"original", *fake_methods}:
            raise RuntimeError(f"Method mismatch in {condition_id}")
        path = output_dir / f"{condition_id.lower()}_seed{args.seed}.csv"
        selected.to_csv(path, index=False)
        metadata = {
            "protocol": "family_coverage_fixed_budget_v1",
            "condition": condition_id,
            "seed": args.seed,
            "fake_methods": list(fake_methods),
            "sampling": "deterministic SHA-256 rank without replacement",
            "nested_method_sampling": (
                "Each method uses one stable ranking; smaller multi-family quotas "
                "are prefixes of the corresponding single-family selection."
            ),
            "budgets": {
                "train": {"real": args.train_real, "fake": args.train_fake},
                "val": {"real": args.val_real, "fake": args.val_fake},
            },
            "fake_quotas": quotas,
            "global_manifest": str(source),
            "global_manifest_sha256": sha256_file(source),
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "images": len(selected),
            "by_split_method": {
                f"{split}:{method}": int(count)
                for (split, method), count in selected.groupby(["split", "method"]).size().items()
            },
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[condition_id] = metadata
        print(condition_id, metadata["by_split_method"], flush=True)

    real_ids = {
        condition_id: set(
            pd.read_csv(output_dir / f"{condition_id.lower()}_seed{args.seed}.csv")
            .query("label == 'real'")["sample_id"]
        )
        for condition_id in CONDITIONS
    }
    if len({frozenset(value) for value in real_ids.values()}) != 1:
        raise RuntimeError("Real samples differ across conditions")
    (output_dir / "conditions_summary.json").write_text(
        json.dumps(
            {
                "protocol": "family_coverage_fixed_budget_v1",
                "global_manifest_sha256": sha256_file(source),
                "real_samples_identical_across_conditions": True,
                "conditions": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
