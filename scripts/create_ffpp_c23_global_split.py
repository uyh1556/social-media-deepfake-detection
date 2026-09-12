#!/usr/bin/env python3
"""Create a portable FF++ c23 split manifest and Colab transfer TAR.

The split is not recalculated. Every source ID inherits its existing membership
from the project's frozen FF++ source-aware manifest. Source images stay in
place; only the portable archive contains a normalized image hierarchy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
FAKE_METHODS = ("Deepfakes", "Face2Face")
SOURCE_ID_PATTERN = re.compile(r"^\d{3}$")
PAIR_ID_PATTERN = re.compile(r"^(\d{3})_(\d{3})$")
MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "label",
    "method",
    "family",
    "domain",
    "compression",
    "preprocessing",
    "group_id",
    "video_id",
    "source_ids",
    "frame_index",
    "source_path",
    "local_source_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        default="ffpp_c23_df40_global_split_v1",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_anchor(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_to_split: dict[str, str] = {}
    source_to_group: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "group_id", "source_ids"}
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f"Anchor manifest is missing fields: {sorted(missing)}")
        for row in reader:
            if row["split"] not in SPLITS:
                raise ValueError(f"Unexpected split: {row['split']}")
            for source_id in row["source_ids"].replace("|", ",").split(","):
                source_id = source_id.strip()
                if not source_id:
                    continue
                old_split = source_to_split.setdefault(source_id, row["split"])
                old_group = source_to_group.setdefault(source_id, row["group_id"])
                if old_split != row["split"] or old_group != row["group_id"]:
                    raise RuntimeError(f"Inconsistent anchor membership for {source_id}")
    return source_to_split, source_to_group


def frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"Non-numeric frame filename: {path}") from error


def build_record(
    image_path: Path,
    split: str,
    label: str,
    method: str,
    group_id: str,
    video_id: str,
    source_ids: tuple[str, ...],
    project_root: Path,
) -> dict[str, Any]:
    archive_method = "original" if method == "original" else method
    archive_path = Path("images") / archive_method / video_id / image_path.name
    return {
        "sample_id": f"{method}:{video_id}:{image_path.name}",
        "split": split,
        "label": label,
        "method": method,
        "family": "real" if label == "real" else (
            "face_swap" if method == "Deepfakes" else "face_reenactment"
        ),
        "domain": "ff",
        "compression": "c23",
        "preprocessing": "df40_deepfakebench_aligned_face_256",
        "group_id": group_id,
        "video_id": video_id,
        "source_ids": "|".join(source_ids),
        "frame_index": frame_index(image_path),
        "source_path": archive_path.as_posix(),
        "local_source_path": image_path.relative_to(project_root).as_posix(),
        "_physical_path": image_path,
    }


def collect_records(
    processed_root: Path,
    real_root: Path,
    source_to_split: dict[str, str],
    source_to_group: dict[str, str],
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for video_dir in sorted(path for path in real_root.iterdir() if path.is_dir()):
        video_id = video_dir.name
        if not SOURCE_ID_PATTERN.fullmatch(video_id):
            raise ValueError(f"Unexpected real video directory: {video_dir}")
        images = sorted(video_dir.glob("*.png"), key=frame_index)
        split = source_to_split.get(video_id)
        if split is None:
            exclusions.append(
                {
                    "method": "original",
                    "video_id": video_id,
                    "source_ids": video_id,
                    "images": len(images),
                    "reason": "source_not_in_anchor_manifest",
                    "local_path": video_dir.relative_to(project_root).as_posix(),
                }
            )
            continue
        for image in images:
            records.append(
                build_record(
                    image,
                    split,
                    "real",
                    "original",
                    source_to_group[video_id],
                    video_id,
                    (video_id,),
                    project_root,
                )
            )

    for method in FAKE_METHODS:
        frames_root = processed_root / method / "ff" / "frames"
        if not frames_root.is_dir():
            raise FileNotFoundError(frames_root)
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            match = PAIR_ID_PATTERN.fullmatch(video_dir.name)
            if not match:
                raise ValueError(f"Unexpected fake video directory: {video_dir}")
            source_ids = match.groups()
            images = sorted(video_dir.glob("*.png"), key=frame_index)
            memberships = {source_to_split.get(source_id) for source_id in source_ids}
            if None in memberships:
                exclusions.append(
                    {
                        "method": method,
                        "video_id": video_dir.name,
                        "source_ids": "|".join(source_ids),
                        "images": len(images),
                        "reason": "source_not_in_anchor_manifest",
                        "local_path": video_dir.relative_to(project_root).as_posix(),
                    }
                )
                continue
            if len(memberships) != 1:
                raise RuntimeError(
                    f"Anchor leakage: {method}/{video_dir.name} -> {memberships}"
                )
            groups = {source_to_group[source_id] for source_id in source_ids}
            if len(groups) != 1:
                raise RuntimeError(
                    f"Anchor group mismatch: {method}/{video_dir.name} -> {groups}"
                )
            split = memberships.pop()
            group_id = groups.pop()
            for image in images:
                records.append(
                    build_record(
                        image,
                        split,
                        "fake",
                        method,
                        group_id,
                        video_dir.name,
                        source_ids,
                        project_root,
                    )
                )

    records.sort(
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["label"],
            row["method"],
            row["video_id"],
            row["frame_index"],
        )
    )
    exclusions.sort(key=lambda row: (row["method"], row["video_id"]))
    return records, exclusions


def validate(records: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    sample_ids = [row["sample_id"] for row in records]
    source_paths = [row["source_path"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs detected")
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError("Duplicate archive paths detected")

    source_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing: list[str] = []
    for row in records:
        for source_id in row["source_ids"].split("|"):
            source_splits[source_id].add(row["split"])
        group_splits[row["group_id"]].add(row["split"])
        video_splits[(row["method"], row["video_id"])].add(row["split"])
        physical = project_root / row["local_source_path"]
        if not physical.is_file():
            missing.append(str(physical))

    source_overlap = {key: value for key, value in source_splits.items() if len(value) > 1}
    group_overlap = {key: value for key, value in group_splits.items() if len(value) > 1}
    video_overlap = {key: value for key, value in video_splits.items() if len(value) > 1}
    if source_overlap or group_overlap or video_overlap or missing:
        raise RuntimeError(
            "Split validation failed: "
            f"sources={len(source_overlap)}, groups={len(group_overlap)}, "
            f"videos={len(video_overlap)}, missing={len(missing)}"
        )
    return {
        "duplicate_sample_ids": 0,
        "duplicate_archive_paths": 0,
        "source_id_overlap": 0,
        "group_overlap": 0,
        "method_video_overlap": 0,
        "missing_source_files": 0,
    }


def summarize(
    records: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    checks: dict[str, Any],
    anchor_manifest: Path,
) -> dict[str, Any]:
    split_counts: dict[str, Any] = {}
    for split in SPLITS:
        selected = [row for row in records if row["split"] == split]
        split_counts[split] = {
            "images": len(selected),
            "real": sum(row["label"] == "real" for row in selected),
            "fake": sum(row["label"] == "fake" for row in selected),
            "by_method": dict(Counter(row["method"] for row in selected)),
            "videos_by_method": {
                method: len(
                    {row["video_id"] for row in selected if row["method"] == method}
                )
                for method in ("original",) + FAKE_METHODS
            },
            "groups": len({row["group_id"] for row in selected}),
            "source_ids": len(
                {
                    source_id
                    for row in selected
                    for source_id in row["source_ids"].split("|")
                }
            ),
        }
    return {
        "created_at": utc_now(),
        "protocol": "global_source_aware_v1",
        "split_policy": (
            "Reuse the frozen FF++ source-aware membership without "
            "re-randomization. Unmapped sources are excluded."
        ),
        "anchor_manifest": str(anchor_manifest),
        "anchor_manifest_sha256": sha256(anchor_manifest),
        "images": len(records),
        "excluded_images": sum(int(row["images"]) for row in exclusions),
        "excluded_videos": len(exclusions),
        "splits": split_counts,
        "checks": checks,
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in MANIFEST_FIELDS}


def normalized_tar_info(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
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
    records: list[dict[str, Any]],
    metadata_files: list[Path],
) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    started = time.monotonic()
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path in metadata_files:
                normalized_tar_info(
                    tar,
                    path,
                    f"{archive_root}/{path.name}",
                )
            for index, row in enumerate(records, start=1):
                normalized_tar_info(
                    tar,
                    row["_physical_path"],
                    f"{archive_root}/{row['source_path']}",
                )
                if index % 5000 == 0 or index == len(records):
                    elapsed = time.monotonic() - started
                    rate = index / elapsed if elapsed else 0.0
                    remaining = (len(records) - index) / rate if rate else 0.0
                    print(
                        f"Archived {index}/{len(records)} images | "
                        f"ETA={remaining / 60:.1f} min",
                        flush=True,
                    )
        temporary.replace(archive)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    processed_root = args.processed_root.resolve()
    real_root = args.real_root.resolve()
    anchor_manifest = args.anchor_manifest.resolve()
    output_dir = args.output_dir.resolve()
    archive = args.archive.resolve()
    for path in (processed_root, real_root, anchor_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_to_split, source_to_group = load_anchor(anchor_manifest)
    records, exclusions = collect_records(
        processed_root,
        real_root,
        source_to_split,
        source_to_group,
        project_root,
    )
    checks = validate(records, project_root)
    summary = summarize(records, exclusions, checks, anchor_manifest)

    manifest_path = output_dir / "manifest.csv"
    exclusions_path = output_dir / "excluded_videos.csv"
    summary_path = output_dir / "split_summary.json"
    config_path = output_dir / "archive_config.json"
    write_csv(manifest_path, [public_row(row) for row in records], MANIFEST_FIELDS)
    write_csv(
        exclusions_path,
        exclusions,
        ("method", "video_id", "source_ids", "images", "reason", "local_path"),
    )
    write_json(summary_path, summary)
    write_json(
        config_path,
        {
            "archive_root": args.archive_root,
            "data_root_after_extraction": args.archive_root,
            "manifest_after_extraction": f"{args.archive_root}/manifest.csv",
            "images_in_archive": len(records),
            "includes_landmarks": False,
            "includes_images": True,
            "source_files_modified": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    create_archive(
        archive,
        args.archive_root,
        records,
        [manifest_path, exclusions_path, summary_path, config_path],
    )
    archive_metadata = {
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "image_members": len(records),
        "metadata_members": 4,
        "total_members": len(records) + 4,
        "created_at": utc_now(),
    }
    write_json(output_dir / "archive_metadata.json", archive_metadata)
    print(json.dumps(archive_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
