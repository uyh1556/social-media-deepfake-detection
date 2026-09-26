#!/usr/bin/env python3
"""Create six single-method manifests for one frozen family rotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from create_family_rotation_method_archives import METHOD_SLUGS
PROTOCOL = "family_rotation_method_transfer_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, choices=[f"S{i}" for i in range(1, 7)])
    parser.add_argument("--source-manifest-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=root / "configs/family_rotation_v1/selections.json",
    )
    return parser.parse_args()


def stable_rank(sample_id: str, seed: int, split: str) -> str:
    return hashlib.sha256(
        f"{seed}|{split}|original|{sample_id}".encode("utf-8")
    ).hexdigest()


def select_real(frame: pd.DataFrame, split: str, count: int, seed: int) -> pd.DataFrame:
    rows = frame[(frame["split"] == split) & (frame["method"] == "original")].copy()
    if len(rows) < count:
        raise RuntimeError(f"Not enough Real rows for {split}: {len(rows)} < {count}")
    rows["_rank"] = rows["sample_id"].map(lambda x: stable_rank(x, seed, split))
    return rows.sort_values(["_rank", "sample_id"]).head(count).drop(columns="_rank")


def main() -> None:
    args = parse_args()
    config = json.loads(args.selection_config.read_text(encoding="utf-8"))
    selection = args.selection.upper()
    source_dir = args.source_manifest_root / selection.lower()
    output_dir = args.output_root / selection.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    parents = {}
    for model in ("M1", "M2", "M3"):
        path = source_dir / f"{model.lower()}_seed42.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        parents[model] = pd.read_csv(path, dtype=str, keep_default_na=False)

    real_sets = [frozenset(x.loc[x["label"] == "real", "sample_id"]) for x in parents.values()]
    if len(set(real_sets)) != 1:
        raise RuntimeError("M1-M3 Real pools differ")

    seed = int(config["sampling_seed"])
    reference = parents["M1"]
    real = {
        "train": select_real(reference, "train", 9900, seed),
        "val": select_real(reference, "val", 2340, seed),
    }
    letters = config["selections"][selection]
    source_models = {"FS": "M1", "FR": "M2", "EFS": "M3"}
    methods = {}
    hashes = {}

    for family in ("FS", "FR", "EFS"):
        parent = parents[source_models[family]]
        for letter in letters:
            method = config["families"][family][letter]
            slug = METHOD_SLUGS[method]
            fake = parent[(parent["label"] == "fake") & (parent["method"] == method)]
            counts = fake.groupby("split").size().to_dict()
            if counts != {"train": 9900, "val": 2340}:
                raise RuntimeError(f"Unexpected {method} counts: {counts}")
            selected = pd.concat([real["train"], real["val"], fake], ignore_index=True)
            selected = selected.sort_values(["split", "label", "method", "sample_id"])
            path = output_dir / f"{slug}_seed42.csv"
            temporary = path.with_suffix(".csv.tmp")
            selected.to_csv(temporary, index=False)
            if path.exists() and sha256_file(path) != sha256_file(temporary):
                temporary.unlink()
                raise FileExistsError(f"Existing manifest differs: {path}")
            if path.exists():
                temporary.unlink()
            else:
                temporary.replace(path)
            methods[method] = {"slug": slug, "family": family}
            hashes[method] = sha256_file(path)

    test = pd.read_csv(args.test_manifest, dtype=str, keep_default_na=False)
    expected_test_methods = {"original", *[m for fam in config["families"].values() for m in fam.values()]}
    if not expected_test_methods.issubset(set(test["method"])):
        missing = sorted(expected_test_methods - set(test["method"]))
        raise RuntimeError(f"Family-rotation test methods missing: {missing}")

    protocol = {
        "protocol": PROTOCOL,
        "selection": selection,
        "letters": letters,
        "training_manifest_sampling_seed": seed,
        "training_seeds": [42],
        "methods": methods,
        "evaluation_methods": list(methods),
        "budgets": {"train": {"real": 9900, "fake": 9900}, "val": {"real": 2340, "fake": 2340}},
        "manifest_sha256": hashes,
        "test_manifest_sha256": sha256_file(args.test_manifest),
        "expected_images_per_method": 2000,
        "run_name_template": "xception_rotation_transfer_{slug}_jpegmixed_v1_seed{seed}",
        "condition_name_template": "rotation_transfer_{slug}_jpeg_mixed",
        "preprocessing": {
            "name": "canonical256_jpegmixed_letterbox299",
            "canonical_size": 256,
            "train_jpeg_qualities": [75, 80, 85, 90, 95],
            "train_jpeg_sampling": "uniform",
            "validation_jpeg_quality": 95,
            "jpeg_subsampling": 2,
            "jpeg_optimize": False,
            "jpeg_progressive": False,
        },
        "evaluation_conditions": [
            {"name": "canonical_256_jpeg_q95", "jpeg_quality": 95, "primary": True},
            {"name": "canonical_256_jpeg_q90", "jpeg_quality": 90, "primary": False},
            {"name": "canonical_256_jpeg_q75", "jpeg_quality": 75, "primary": False},
        ],
    }
    protocol_path = output_dir / "protocol.json"
    payload = json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if protocol_path.exists() and protocol_path.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"Existing protocol differs: {protocol_path}")
    protocol_path.write_text(payload, encoding="utf-8")
    print("Methods:", ", ".join(methods), flush=True)
    print("Saved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
