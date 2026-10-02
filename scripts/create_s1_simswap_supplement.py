#!/usr/bin/env python3
"""Package original images needed by frozen S1 but absent from the rotation TAR."""

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from create_family_rotation_method_archives import add_bytes, add_file
from prepare_family_rotation_adaptive_learning import BUNDLE_SHA, ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ROOT / "data/exports/family_rotation_method_archives_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.archive_root / "df40_simswap_s1_supplement_v1.tar"
    if output.exists():
        raise FileExistsError(f"Preserve existing supplement: {output}")
    bundle = args.archive_root / "family_rotation_evaluation_manifests_v1.tar.xz"
    if sha(bundle) != BUNDLE_SHA:
        raise RuntimeError("Expected frozen checkpoint-compatible manifest bundle")
    with tarfile.open(bundle, "r:xz") as archive:
        frame = pd.read_csv(archive.extractfile("family_rotation_manifests/s1/m7_seed42.csv"), dtype=str, keep_default_na=False)
    prefix = "deepfake_family_rotation_v1/"
    with tarfile.open(args.archive_root / "df40_simswap_trainval_v1.tar", "r:") as archive:
        names = {m.name.removeprefix("./").removeprefix(prefix) for m in archive if m.isfile()}
    missing = frame[(frame.method == "SimSwap") & ~frame.source_path.isin(names)].copy()
    if missing.empty:
        print("Existing TAR already contains every frozen S1 SimSwap image.", flush=True)
        return
    if not set(missing.split) <= {"train", "val"}:
        raise ValueError("Supplement must contain only original train/validation rows")
    for row in missing.itertuples(index=False):
        relative = PurePosixPath(row.local_source_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe local path: {relative}")
        source = ROOT / row.local_source_path
        if not source.is_file() or sha(source) != row.content_sha256:
            raise RuntimeError(f"Original frozen image missing or changed: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    metadata = {"purpose": "restore exact frozen S1 image availability", "images": len(missing),
                "splits": missing.groupby("split").size().to_dict(), "manifest_bundle_sha256": BUNDLE_SHA}
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        add_bytes(archive, prefix + "manifests/s1_simswap_supplement.csv", missing.to_csv(index=False).encode())
        add_bytes(archive, prefix + "manifests/s1_simswap_supplement.json", json.dumps(metadata, indent=2).encode())
        for row in missing.itertuples(index=False):
            add_file(archive, ROOT / row.local_source_path, prefix + row.source_path)
    temporary.replace(output)
    print(f"Created {output}: {len(missing)} original images, {output.stat().st_size / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
