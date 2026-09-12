#!/usr/bin/env python3
"""Create independently extractable dataset TAR modules from frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--module", action="append", dest="modules")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with path.open("rb") as handle:
        tar.addfile(info, handle)


def create_archive(
    archive: Path,
    archive_root: str,
    frame: pd.DataFrame,
    metadata: list[tuple[Path, str]],
    project_root: Path,
) -> None:
    if archive.exists():
        raise FileExistsError(archive)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    started = time.monotonic()
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as tar:
            for path, relative in metadata:
                normalized_add(tar, path, f"{archive_root}/{relative}")
            for index, row in enumerate(frame.itertuples(index=False), start=1):
                source = project_root / row.local_source_path
                if not source.is_file():
                    raise FileNotFoundError(source)
                normalized_add(tar, source, f"{archive_root}/{row.source_path}")
                if index % 5000 == 0 or index == len(frame):
                    elapsed = time.monotonic() - started
                    rate = index / elapsed if elapsed else 0.0
                    eta = (len(frame) - index) / rate if rate else 0.0
                    print(
                        f"{archive.name}: {index}/{len(frame)} | ETA={eta / 60:.1f} min",
                        flush=True,
                    )
        temporary.replace(archive)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = parse_args()
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()
    registry = json.loads((manifest_dir / "module_registry.json").read_text())
    archive_root = registry["archive_root"]
    modules = args.modules or list(registry["modules"])
    unknown = set(modules) - set(registry["modules"])
    if unknown:
        raise ValueError(f"Unknown modules: {sorted(unknown)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for module in modules:
        entry = registry["modules"][module]
        manifest_path = manifest_dir / "modules" / f"{module}.csv"
        module_info_path = manifest_dir / "modules" / f"{module}.json"
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        archive = output_dir / entry["archive"]
        metadata = [
            (
                manifest_path,
                f"manifests/ffpp_df40_global_v1/modules/{module}.csv",
            ),
            (
                module_info_path,
                f"manifests/ffpp_df40_global_v1/modules/{module}.json",
            ),
        ]
        if module == "real":
            metadata.extend(
                [
                    (
                        manifest_dir / "manifest.csv",
                        "manifests/ffpp_df40_global_v1/manifest.csv",
                    ),
                    (
                        manifest_dir / "split_summary.json",
                        "manifests/ffpp_df40_global_v1/split_summary.json",
                    ),
                    (
                        manifest_dir / "module_registry.json",
                        "manifests/ffpp_df40_global_v1/module_registry.json",
                    ),
                ]
            )
        create_archive(archive, archive_root, frame, metadata, project_root)
        results[module] = {
            "archive": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "images": len(frame),
            "metadata_members": len(metadata),
            "total_members": len(frame) + len(metadata),
        }
        print(json.dumps({module: results[module]}, ensure_ascii=False, indent=2))

    metadata_path = output_dir / "archive_metadata.json"
    previous = {}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        previous = existing.get("modules", {})
    previous.update(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": archive_root,
        "modules": previous,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
