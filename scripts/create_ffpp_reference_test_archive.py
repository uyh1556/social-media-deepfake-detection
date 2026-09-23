#!/usr/bin/env python3
"""Create a balanced Deepfakes + Face2Face in-domain test TAR."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd


METHODS = ("Deepfakes", "Face2Face")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images-per-method", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank(sample_id: str, seed: int, method: str) -> str:
    return hashlib.sha256(
        f"{seed}|test|{method}|{sample_id}".encode("utf-8")
    ).hexdigest()


def select(frame: pd.DataFrame, method: str, count: int, seed: int) -> pd.DataFrame:
    candidates = frame[(frame["split"] == "test") & (frame["method"] == method)]
    if len(candidates) < count:
        raise RuntimeError(
            f"Not enough test/{method}: need={count}, available={len(candidates)}"
        )
    return (
        candidates.assign(
            _rank=candidates["sample_id"].map(lambda value: rank(value, seed, method))
        )
        .sort_values(["_rank", "sample_id"])
        .head(count)
        .drop(columns="_rank")
    )


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source), arcname=archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    selected = pd.concat(
        [select(frame, method, args.images_per_method, args.seed) for method in METHODS],
        ignore_index=True,
    ).sort_values(["method", "source_path"])
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ffpp_deepfakes_face2face_test_v1.tar"
    manifest_path = args.output_dir / "balanced_ffpp_test_2000_v1.csv"
    metadata_path = args.output_dir / "ffpp_test_inventory.json"
    for path in (output, manifest_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)

    missing = [
        path
        for value in selected["local_source_path"]
        if not (path := project_root / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(missing[0])

    metadata = {
        "protocol": "balanced_ffpp_in_domain_test_v1",
        "extract_root": "deepfake_family_rotation_v1",
        "seed": args.seed,
        "images": len(selected),
        "images_per_method": args.images_per_method,
        "methods": list(METHODS),
        "requires_real_from": "all_df40_methods_test_v1.tar",
    }
    manifest_path.write_text(selected.to_csv(index=False), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
        add_bytes(
            archive,
            "deepfake_family_rotation_v1/manifests/evaluation/"
            "balanced_ffpp_test_2000_v1.csv",
            selected.to_csv(index=False).encode(),
        )
        add_bytes(
            archive,
            "deepfake_family_rotation_v1/manifests/evaluation/"
            "balanced_ffpp_test_2000_v1.json",
            (json.dumps(metadata, indent=2) + "\n").encode(),
        )
        for row in selected.itertuples(index=False):
            add_file(
                archive,
                project_root / row.local_source_path,
                f"deepfake_family_rotation_v1/{row.source_path}",
            )
    print(f"Completed {output}", flush=True)


if __name__ == "__main__":
    main()
