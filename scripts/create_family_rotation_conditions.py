#!/usr/bin/env python3
"""Create fixed-budget S1-S6/M1-M7 train-validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--global-manifest", type=Path)
    source.add_argument(
        "--module-root",
        type=Path,
        help="Extracted deepfake_family_rotation_v1 root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/family_rotation_v1/selections.json",
    )
    parser.add_argument(
        "--selections",
        nargs="+",
        choices=[f"S{i}" for i in range(1, 7)],
        default=[f"S{i}" for i in range(1, 7)],
    )
    return parser.parse_args()


def rank(value: str, seed: int, split: str, method: str) -> str:
    return hashlib.sha256(
        f"{seed}|{split}|{method}|{value}".encode("utf-8")
    ).hexdigest()


def select(
    frame: pd.DataFrame, split: str, method: str, count: int, seed: int
) -> pd.DataFrame:
    candidates = frame[(frame["split"] == split) & (frame["method"] == method)]
    if len(candidates) < count:
        raise RuntimeError(
            f"Not enough {split}/{method}: need={count}, available={len(candidates)}"
        )
    return (
        candidates.assign(
            _rank=candidates["sample_id"].map(
                lambda value: rank(value, seed, split, method)
            )
        )
        .sort_values(["_rank", "sample_id"])
        .head(count)
        .drop(columns="_rank")
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.global_manifest is not None:
        frame = pd.read_csv(args.global_manifest, dtype=str, keep_default_na=False)
    else:
        from create_family_rotation_method_archives import METHOD_SLUGS

        required_methods = {
            config["families"][family][letter]
            for selection in args.selections
            for family in config["families"]
            for letter in config["selections"][selection]
        }
        paths = [args.module_root / "manifests/modules/real_trainval.csv"]
        paths.extend(
            args.module_root / f"manifests/methods/{slug}_trainval.csv"
            for method, slug in METHOD_SLUGS.items()
            if method in required_methods
        )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        frame = pd.concat(
            [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths],
            ignore_index=True,
        ).drop_duplicates("sample_id")
    seed = int(config["sampling_seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    real = {
        split: select(
            frame,
            split,
            "original",
            int(config["budgets"][split]["real"]),
            seed,
        )
        for split in ("train", "val")
    }
    summary = {}
    for selection in args.selections:
        letters = config["selections"][selection]
        selection_dir = args.output_dir / selection.lower()
        selection_dir.mkdir()
        family_methods = {
            family: [config["families"][family][letter] for letter in letters]
            for family in config["families"]
        }
        for model, model_config in config["models"].items():
            methods = [
                method
                for family in model_config["families"]
                for method in family_methods[family]
            ]
            pieces = []
            quotas = {}
            for split in ("train", "val"):
                pieces.append(real[split])
                fake_total = int(config["budgets"][split]["fake"])
                if fake_total % len(methods):
                    raise ValueError(f"Budget is not divisible for {selection}/{model}")
                quota = fake_total // len(methods)
                quotas[split] = quota
                pieces.extend(
                    select(frame, split, method, quota, seed) for method in methods
                )
            selected = pd.concat(pieces, ignore_index=True).sort_values(
                ["split", "label", "method", "sample_id"]
            )
            if selected["sample_id"].duplicated().any():
                raise RuntimeError(f"Duplicate sample IDs: {selection}/{model}")
            csv_path = selection_dir / f"{model.lower()}_seed{seed}.csv"
            selected.to_csv(csv_path, index=False)
            metadata = {
                "protocol": config["protocol"],
                "selection": selection,
                "letters": letters,
                "model": model,
                "families": model_config["families"],
                "fake_methods": methods,
                "sampling_seed": seed,
                "per_method_quota": quotas,
                "images": len(selected),
            }
            (selection_dir / f"{model.lower()}_seed{seed}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary[f"{selection}_{model}"] = metadata
    (args.output_dir / "conditions_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
