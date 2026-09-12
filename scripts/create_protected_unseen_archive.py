#!/usr/bin/env python3
"""Build the leakage-safe InSwapper/FOMM held-out manifest and Colab TAR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


METHODS = {
    "inswap": {
        "display_name": "InSwapper",
        "family": "face_swap",
        "source_kind": "paired_ff_ids",
    },
    "fomm": {
        "display_name": "FOMM",
        "family": "face_reenactment",
        "source_kind": "ff_id_and_driver",
    },
}
FIELDS = (
    "sample_id", "split", "label", "method", "family", "module",
    "domain", "compression", "preprocessing", "group_id", "video_id",
    "source_ids", "driver_id", "frame_index", "source_path",
    "local_source_path", "physical_splits", "duplicate_copies",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--df40-root", type=Path, default=Path("data/df40"))
    parser.add_argument(
        "--global-manifest",
        type=Path,
        default=Path("data/splits/ffpp_df40_global_v1/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/df40/splits/protected_unseen_v1"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(
            "data/exports/df40_protected_unseen_v1/"
            "df40_inswap_fomm_ff_global_test_v1.tar"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_video(method: str, video_id: str) -> tuple[tuple[str, ...], str]:
    parts = video_id.removeprefix("temp_").split("_", 1)
    if len(parts) != 2 or len(parts[0]) != 3 or not parts[0].isdigit():
        raise ValueError(f"Unexpected {method} video ID: {video_id}")
    if METHODS[method]["source_kind"] == "paired_ff_ids":
        if len(parts[1]) != 3 or not parts[1].isdigit():
            raise ValueError(f"Unexpected {method} video ID: {video_id}")
        return (parts[0], parts[1]), ""
    return (parts[0],), parts[1]


def json_expected(df40_root: Path, method: str, split: str) -> set[str]:
    path = df40_root / method / f"{method}_ff.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload[f"{method}_ff"][f"{method}_Fake"][split]
    return {
        f"{Path(frame).parent.name}/{Path(frame).name}"
        for video in videos.values()
        for frame in video["frames"]
    }


def validate_physical_data(df40_root: Path) -> dict:
    report = {}
    for method in METHODS:
        method_report = {}
        for split in ("train", "test"):
            root = df40_root / method / split / "ff"
            disk = {
                f"{path.parent.name}/{path.name}"
                for path in (root / "frames").rglob("*.png")
            }
            landmarks = {
                f"{path.parent.name}/{path.stem}.png"
                for path in (root / "landmarks").rglob("*.npy")
            }
            expected = json_expected(df40_root, method, split)
            if disk != expected:
                raise RuntimeError(
                    f"{method} {split} does not match JSON: "
                    f"disk={len(disk)}, expected={len(expected)}"
                )
            if landmarks != disk:
                raise RuntimeError(f"{method} {split} landmark mismatch")
            method_report[split] = {
                "images": len(disk),
                "videos": len({value.split("/", 1)[0] for value in disk}),
            }
        report[method] = method_report
    return report


def load_membership(master: pd.DataFrame) -> tuple[dict, dict]:
    source_membership = {}
    driver_splits: dict[str, set[str]] = defaultdict(set)
    for row in master.itertuples(index=False):
        for source_id in row.source_ids.split("|"):
            previous = source_membership.setdefault(
                source_id, (row.split, row.group_id)
            )
            if previous != (row.split, row.group_id):
                raise RuntimeError(f"Conflicting source membership: {source_id}")
        if row.driver_id:
            driver_splits[row.driver_id].add(row.split)
    return source_membership, driver_splits


def collect_candidates(df40_root: Path, method: str) -> tuple[dict, int]:
    candidates: dict[tuple[str, str], list[tuple[Path, str]]] = defaultdict(list)
    for physical_split in ("train", "test"):
        root = df40_root / method / physical_split / "ff" / "frames"
        for path in sorted(root.rglob("*.png")):
            candidates[(path.parent.name, path.name)].append((path, physical_split))
    duplicates = 0
    canonical = {}
    for key, copies in candidates.items():
        if len(copies) > 1:
            hashes = {sha256(path) for path, _ in copies}
            if len(hashes) != 1:
                raise RuntimeError(f"Non-identical duplicate: {method}/{key}")
            duplicates += 1
        copies.sort(key=lambda item: (item[1] != "test", item[0].as_posix()))
        canonical[key] = copies
    return canonical, duplicates


def protected_groups(real_rows: list[dict], fake_rows: list[dict]) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in real_rows:
        adjacency[f"base:{row['group_id']}"]
    for row in fake_rows:
        nodes = [f"base:{row['group_id']}"]
        if row["driver_id"]:
            nodes.append(f"driver:{row['driver_id']}")
        for node in nodes:
            adjacency[node].update(other for other in nodes if other != node)
    node_to_group = {}
    for start in sorted(adjacency):
        if start in node_to_group:
            continue
        queue = deque([start])
        seen = {start}
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        group = "unseen_" + hashlib.sha256(
            "|".join(sorted(seen)).encode()
        ).hexdigest()[:12]
        for node in seen:
            node_to_group[node] = group
    for row in real_rows + fake_rows:
        row["group_id"] = node_to_group[f"base:{row['group_id']}"]


def add_to_tar(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(source), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with source.open("rb") as handle:
        tar.addfile(info, handle)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    df40_root = args.df40_root.resolve()
    global_manifest = args.global_manifest.resolve()
    output_dir = args.output_dir.resolve()
    archive = args.archive.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if archive.exists():
        raise FileExistsError(archive)

    physical_report = validate_physical_data(df40_root)
    master = pd.read_csv(global_manifest, dtype=str, keep_default_na=False)
    source_membership, driver_splits = load_membership(master)
    real_frame = master[(master["module"] == "real") & (master["split"] == "test")]
    real_rows = []
    for row in real_frame.itertuples(index=False):
        real_rows.append({field: getattr(row, field, "") for field in FIELDS})
    exclusions = defaultdict(int)
    duplicate_counts = {}
    fake_rows = []

    for method, metadata in METHODS.items():
        candidates, duplicates = collect_candidates(df40_root, method)
        duplicate_counts[method] = duplicates
        for (video_id, frame_name), copies in candidates.items():
            source_ids, driver_id = parse_video(method, video_id)
            memberships = [source_membership.get(value) for value in source_ids]
            if any(value is None for value in memberships):
                exclusions[f"{method}:source_unmapped"] += 1
                continue
            splits = {value[0] for value in memberships}
            groups = {value[1] for value in memberships}
            if splits != {"test"}:
                exclusions[f"{method}:source_not_test"] += 1
                continue
            if len(groups) != 1:
                raise RuntimeError(f"Sources span global groups: {video_id}")
            if driver_id and driver_splits.get(driver_id, set()) - {"test"}:
                exclusions[f"{method}:driver_seen_outside_test"] += 1
                continue
            source, _ = copies[0]
            relative = source.relative_to(project_root).as_posix()
            fake_rows.append(
                {
                    "sample_id": f"{method}:{video_id}:{frame_name}",
                    "split": "test",
                    "label": "fake",
                    "method": metadata["display_name"],
                    "family": metadata["family"],
                    "module": "protected_unseen",
                    "domain": "ff",
                    "compression": "c23",
                    "preprocessing": "df40_provided_aligned_face_256",
                    "group_id": next(iter(groups)),
                    "video_id": video_id,
                    "source_ids": "|".join(source_ids),
                    "driver_id": driver_id,
                    "frame_index": int(Path(frame_name).stem),
                    "source_path": (
                        Path("images") / metadata["display_name"] / video_id / frame_name
                    ).as_posix(),
                    "local_source_path": relative,
                    "physical_splits": "|".join(sorted({item[1] for item in copies})),
                    "duplicate_copies": len(copies),
                }
            )

    protected_groups(real_rows, fake_rows)
    fake_rows.sort(key=lambda row: (row["method"], row["video_id"], row["frame_index"]))
    real_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    binary_rows = real_rows + fake_rows
    if {row["split"] for row in binary_rows} != {"test"}:
        raise RuntimeError("Protected manifest contains a non-test row")
    if len({row["sample_id"] for row in binary_rows}) != len(binary_rows):
        raise RuntimeError("Duplicate sample IDs in protected manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    fake_manifest = output_dir / "fake_test.csv"
    binary_manifest = output_dir / "binary_test.csv"
    write_csv(fake_manifest, fake_rows)
    write_csv(binary_manifest, binary_rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "protected_unseen_ffpp_global_test_v1",
        "role": "evaluation_only_never_train_or_validate",
        "global_manifest": str(global_manifest),
        "global_manifest_sha256": sha256(global_manifest),
        "physical_data": physical_report,
        "deduplicated_logical_samples": duplicate_counts,
        "excluded": dict(sorted(exclusions.items())),
        "real_test_images": len(real_rows),
        "fake_test_images": len(fake_rows),
        "fake_by_method": {
            method: int(count)
            for method, count in pd.Series(
                [row["method"] for row in fake_rows]
            ).value_counts().sort_index().items()
        },
        "videos_by_method": {
            method: len({row["video_id"] for row in fake_rows if row["method"] == method})
            for method in ("InSwapper", "FOMM")
        },
        "fake_manifest_sha256": sha256(fake_manifest),
        "binary_manifest_sha256": sha256(binary_manifest),
        "cdf_preserved_locally_not_in_archive": {
            method: len(list((df40_root / method / "test/cdf/frames").rglob("*.png")))
            for method in METHODS
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".tar.partial")
    try:
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as tar:
            for metadata_path in (fake_manifest, binary_manifest, summary_path):
                add_to_tar(
                    tar,
                    metadata_path,
                    "deepfake_data_v1/manifests/protected_unseen_v1/"
                    + metadata_path.name,
                )
            for index, row in enumerate(fake_rows, start=1):
                add_to_tar(
                    tar,
                    project_root / row["local_source_path"],
                    "deepfake_data_v1/" + row["source_path"],
                )
                if index % 1000 == 0 or index == len(fake_rows):
                    print(f"Archived {index}/{len(fake_rows)}", flush=True)
        partial.replace(archive)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    archive_info = {
        "archive": archive.name,
        "archive_root": "deepfake_data_v1",
        "size_bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "fake_images": len(fake_rows),
        "metadata_members": 3,
        "requires_existing_real_module": "df40_real_ff_c23_global_v1.tar",
    }
    (archive.parent / "inventory.json").write_text(
        json.dumps(archive_info, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (archive.parent / "SHA256SUMS.txt").write_text(
        f"{archive_info['sha256']}  {archive.name}\n", encoding="utf-8"
    )
    print(json.dumps({**summary, **archive_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
