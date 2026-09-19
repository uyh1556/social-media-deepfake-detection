#!/usr/bin/env python3
"""Create six balanced single-method manifests from frozen M1-M3 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROTOCOL = "method_transfer_matrix_v1"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=project_root / f"configs/{PROTOCOL}/protocol.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(sample_id: str, seed: int, split: str) -> str:
    value = f"{seed}|{split}|original|{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_real(
    frame: pd.DataFrame,
    *,
    split: str,
    count: int,
    seed: int,
) -> pd.DataFrame:
    candidates = frame[
        (frame["split"] == split)
        & (frame["label"] == "real")
        & (frame["method"] == "original")
    ].copy()
    if len(candidates) < count:
        raise RuntimeError(
            f"Not enough {split}/real rows: need={count}, have={len(candidates)}"
        )
    candidates["_rank"] = candidates["sample_id"].map(
        lambda value: stable_rank(value, seed, split)
    )
    return (
        candidates.sort_values(["_rank", "sample_id"])
        .head(count)
        .drop(columns="_rank")
    )


def validate_parent(
    path: Path,
    expected_hash: str,
    expected_methods: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Source manifest hash mismatch for {path.name}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "sample_id",
        "split",
        "label",
        "method",
        "family",
        "source_path",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample IDs in {path}")
    methods = set(frame.loc[frame["label"] == "fake", "method"])
    if methods != set(expected_methods):
        raise RuntimeError(
            f"Unexpected fake methods in {path}: {sorted(methods)}"
        )
    return frame


def main() -> None:
    args = parse_args()
    source_dir = args.source_manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.protocol_config.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {config_path}")

    parents: dict[str, pd.DataFrame] = {}
    for condition, item in protocol["source_manifests"].items():
        parents[condition] = validate_parent(
            source_dir / item["file"],
            item["sha256"],
            item["methods"],
        )

    real_sets = []
    for frame in parents.values():
        real_sets.append(
            frozenset(frame.loc[frame["label"] == "real", "sample_id"])
        )
    if len(set(real_sets)) != 1:
        raise RuntimeError("M1-M3 do not contain the same Real sample pool")

    reference = parents["M1"]
    seed = int(protocol["training_manifest_sampling_seed"])
    common_real = {
        split: select_real(
            reference,
            split=split,
            count=int(protocol["budgets"][split]["real"]),
            seed=seed,
        )
        for split in ("train", "val")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for method, definition in protocol["methods"].items():
        source = parents[definition["source_condition"]]
        pieces = []
        for split in ("train", "val"):
            fake = source[
                (source["split"] == split)
                & (source["label"] == "fake")
                & (source["method"] == method)
            ].copy()
            expected_fake = int(protocol["budgets"][split]["fake"])
            if len(fake) != expected_fake:
                raise RuntimeError(
                    f"Unexpected {split}/{method} count: "
                    f"expected={expected_fake}, actual={len(fake)}"
                )
            pieces.extend([common_real[split], fake])
        selected = pd.concat(pieces, ignore_index=True).sort_values(
            ["split", "label", "method", "video_id", "frame_index", "sample_id"]
        )
        if selected["sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample IDs for {method}")
        output_path = output_dir / f"{definition['slug']}_seed42.csv"
        temporary = output_path.with_suffix(".csv.tmp")
        selected.to_csv(temporary, index=False)
        generated_hash = sha256_file(temporary)
        expected_hash = protocol["manifest_sha256"][method]
        if not expected_hash.startswith("__") and generated_hash != expected_hash:
            temporary.unlink()
            raise ValueError(
                f"Generated manifest hash mismatch for {method}: "
                f"expected={expected_hash}, actual={generated_hash}"
            )
        if output_path.is_file():
            existing_hash = sha256_file(output_path)
            if existing_hash != generated_hash:
                temporary.unlink()
                raise FileExistsError(
                    f"Existing manifest differs: {output_path}"
                )
            temporary.unlink()
        else:
            temporary.replace(output_path)

        metadata = {
            "protocol": PROTOCOL,
            "method": method,
            "family": definition["family"],
            "seed": seed,
            "source_condition": definition["source_condition"],
            "source_manifest": str(
                source_dir
                / protocol["source_manifests"][definition["source_condition"]][
                    "file"
                ]
            ),
            "source_manifest_sha256": protocol["source_manifests"][
                definition["source_condition"]
            ]["sha256"],
            "manifest": str(output_path),
            "manifest_sha256": generated_hash,
            "budgets": protocol["budgets"],
            "real_sampling": (
                "common deterministic prefix of the frozen M1-M3 Real pool"
            ),
            "fake_sampling": "all rows for the method in its frozen M1-M3 manifest",
            "rows": int(len(selected)),
        }
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        summaries[method] = metadata
        print(method, generated_hash, flush=True)

    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "config_sha256": sha256_file(config_path),
                "real_samples_identical_across_methods": True,
                "methods": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Saved:", output_dir, flush=True)


if __name__ == "__main__":
    main()
