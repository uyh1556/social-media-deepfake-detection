#!/usr/bin/env python3
"""Verify modular TAR contents and write upload inventory/checksum files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    registry = json.loads((manifest_dir / "module_registry.json").read_text())
    root = registry["archive_root"]
    inventory = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "ffpp_df40_global_source_aware_v1",
        "archive_root": root,
        "archives": {},
    }

    for module, entry in registry["modules"].items():
        archive = archive_dir / entry["archive"]
        manifest_path = manifest_dir / "modules" / f"{module}.csv"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        expected_images = {f"{root}/{path}" for path in frame["source_path"]}
        expected_manifest_name = (
            f"{root}/manifests/ffpp_df40_global_v1/modules/{module}.csv"
        )
        with tarfile.open(archive, "r") as tar:
            members = [member for member in tar if member.isfile()]
            names = {member.name for member in members}
            image_names = {
                member.name
                for member in members
                if member.name.lower().endswith(".png")
            }
            unsafe = []
            unwanted = []
            for name in names:
                posix = PurePosixPath(name)
                if posix.is_absolute() or ".." in posix.parts:
                    unsafe.append(name)
                lowered = name.lower()
                if (
                    "/._" in lowered
                    or lowered.endswith(".ds_store")
                    or "/landmarks/" in lowered
                    or "/masks/" in lowered
                ):
                    unwanted.append(name)
            if image_names != expected_images:
                raise RuntimeError(
                    f"Archive image set mismatch for {module}: "
                    f"missing={len(expected_images - image_names)}, "
                    f"unexpected={len(image_names - expected_images)}"
                )
            if unsafe or unwanted:
                raise RuntimeError(
                    f"Unsafe/unwanted members in {archive.name}: "
                    f"unsafe={unsafe[:3]}, unwanted={unwanted[:3]}"
                )
            member = tar.getmember(expected_manifest_name)
            archived_manifest = tar.extractfile(member).read()
            if archived_manifest != manifest_path.read_bytes():
                raise RuntimeError(f"Embedded manifest mismatch for {module}")

        inventory["archives"][module] = {
            "filename": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "images": len(frame),
            "methods": entry["methods"],
            "splits": frame["split"].value_counts().to_dict(),
            "members": len(members),
            "embedded_manifest": expected_manifest_name,
            "checks": {
                "image_set_exact": True,
                "embedded_manifest_exact": True,
                "unsafe_paths": 0,
                "unwanted_metadata": 0,
            },
        }
        print(f"Verified {module}: {archive.name}", flush=True)

    inventory_path = archive_dir / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [
        f"{entry['sha256']}  {entry['filename']}"
        for _, entry in sorted(inventory["archives"].items())
    ]
    (archive_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
