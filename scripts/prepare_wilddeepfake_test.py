#!/usr/bin/env python3
"""Build a deterministic, sequence-balanced WildDeepfake test package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


HF_REPOSITORY = "xingjunm/WildDeepfake"
HF_REVISION = "f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4"
EXPECTED_SEQUENCES = {"real": 396, "fake": 410}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/wilddeepfake/deepfake_in_the_wild"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/wilddeepfake/evaluation_16frames_v1"),
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/exports/wilddeepfake_v1"),
    )
    parser.add_argument("--frames-per-sequence", type=int, default=16)
    return parser.parse_args()


def numeric_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.split(".", 1)[0]), path.name
    except ValueError:
        return 10**12, path.name


def frame_key(member: tarfile.TarInfo) -> tuple[int, str]:
    name = PurePosixPath(member.name).name
    try:
        return int(PurePosixPath(name).stem), name
    except ValueError:
        return 10**12, name


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("frames-per-sequence must be positive")
    if total < count:
        raise ValueError(f"Cannot select {count} unique frames from {total}")
    if count == 1:
        return [0]
    indices = [index * (total - 1) // (count - 1) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError((total, count, indices))
    return indices


def parse_member_path(
    member: tarfile.TarInfo,
    expected_label: str,
) -> tuple[str, str, str]:
    parts = PurePosixPath(member.name.lstrip("./")).parts
    if len(parts) != 4:
        raise ValueError(f"Unexpected member path: {member.name}")
    archive_id, label, sequence_id, filename = parts
    if label != expected_label:
        raise ValueError(
            f"Label mismatch in {member.name}: expected {expected_label}, found {label}"
        )
    if PurePosixPath(filename).suffix.lower() != ".png":
        raise ValueError(f"Unexpected image extension: {member.name}")
    return archive_id, sequence_id, filename


def validate_png(data: bytes, source: str) -> tuple[int, int]:
    if len(data) < 26 or data[:8] != PNG_SIGNATURE:
        raise ValueError(f"Invalid PNG: {source}")
    width, height = struct.unpack(">II", data[16:24])
    bit_depth = data[24]
    color_type = data[25]
    if (width, height, bit_depth, color_type) != (224, 224, 8, 2):
        raise ValueError(
            f"Unexpected PNG properties for {source}: "
            f"{width}x{height}, bit_depth={bit_depth}, color_type={color_type}"
        )
    return width, height


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def ensure_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    export_dir = args.export_dir.resolve()
    frames_per_sequence = args.frames_per_sequence

    ensure_new_directory(output_root)
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_path = export_dir / "wilddeepfake_test_16frames_v1.tar"
    if archive_path.exists():
        raise FileExistsError(f"Export archive already exists: {archive_path}")

    manifest_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    source_archive_count: dict[str, int] = {}
    source_archive_bytes: dict[str, int] = {}
    original_frame_count: dict[str, int] = {}
    selected_frame_count: dict[str, int] = {}

    for label in ("real", "fake"):
        source_dir = source_root / f"{label}_test"
        archives = sorted(source_dir.glob("*.tar.gz"), key=numeric_key)
        if not archives:
            raise FileNotFoundError(f"No source archives found in {source_dir}")
        source_archive_count[label] = len(archives)
        source_archive_bytes[label] = sum(path.stat().st_size for path in archives)

        for archive_path_in in archives:
            groups: dict[tuple[str, str], list[tarfile.TarInfo]] = defaultdict(list)
            with tarfile.open(archive_path_in, mode="r:") as source_tar:
                for member in source_tar:
                    if not member.isfile():
                        continue
                    archive_id, sequence_id, _ = parse_member_path(member, label)
                    if archive_id != archive_path_in.name.split(".", 1)[0]:
                        raise ValueError(
                            f"Archive ID mismatch: {archive_path_in.name} contains {member.name}"
                        )
                    groups[(archive_id, sequence_id)].append(member)

                for (archive_id, sequence_id), members in sorted(groups.items()):
                    members.sort(key=frame_key)
                    selected_indices = evenly_spaced_indices(
                        len(members),
                        frames_per_sequence,
                    )
                    sequence_uid = f"{label}:{archive_id}:{sequence_id}"
                    sequence_rows.append(
                        {
                            "sequence_uid": sequence_uid,
                            "label": label,
                            "target": 1 if label == "fake" else 0,
                            "archive_id": archive_id,
                            "sequence_id": sequence_id,
                            "original_frames": len(members),
                            "selected_frames": frames_per_sequence,
                            "source_archive": archive_path_in.relative_to(
                                source_root.parent.parent
                            ).as_posix(),
                        }
                    )
                    original_frame_count[label] = (
                        original_frame_count.get(label, 0) + len(members)
                    )

                    for selection_order, member_index in enumerate(selected_indices):
                        member = members[member_index]
                        _, _, filename = parse_member_path(member, label)
                        handle = source_tar.extractfile(member)
                        if handle is None:
                            raise RuntimeError(f"Could not read {member.name}")
                        data = handle.read()
                        width, height = validate_png(data, member.name)

                        relative_output = (
                            Path("images")
                            / label
                            / archive_id
                            / sequence_id
                            / filename
                        )
                        output_path = output_root / relative_output
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with output_path.open("xb") as output_handle:
                            output_handle.write(data)

                        frame_number = PurePosixPath(filename).stem
                        sample_id = (
                            f"wilddeepfake:{label}:{archive_id}:"
                            f"{sequence_id}:{frame_number}"
                        )
                        manifest_rows.append(
                            {
                                "sample_id": sample_id,
                                "dataset": "WildDeepfake",
                                "split": "test",
                                "label": label,
                                "target": 1 if label == "fake" else 0,
                                "sequence_uid": sequence_uid,
                                "archive_id": archive_id,
                                "sequence_id": sequence_id,
                                "frame_number": frame_number,
                                "selection_order": selection_order,
                                "source_sequence_frames": len(members),
                                "source_member_index": member_index,
                                "source_archive": archive_path_in.relative_to(
                                    source_root.parent.parent
                                ).as_posix(),
                                "source_member": member.name,
                                "relative_path": relative_output.as_posix(),
                                "width": width,
                                "height": height,
                                "format": "PNG",
                                "mode": "RGB",
                                "bytes": len(data),
                                "sha256": sha256_bytes(data),
                            }
                        )
                        selected_frame_count[label] = (
                            selected_frame_count.get(label, 0) + 1
                        )

    sequence_counts = Counter(row["label"] for row in sequence_rows)
    image_counts = Counter(row["label"] for row in manifest_rows)
    if dict(sequence_counts) != EXPECTED_SEQUENCES:
        raise RuntimeError(
            f"Unexpected sequence counts: {dict(sequence_counts)} != {EXPECTED_SEQUENCES}"
        )
    expected_images = {
        label: count * frames_per_sequence
        for label, count in EXPECTED_SEQUENCES.items()
    }
    if dict(image_counts) != expected_images:
        raise RuntimeError(
            f"Unexpected image counts: {dict(image_counts)} != {expected_images}"
        )
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    relative_paths = [str(row["relative_path"]) for row in manifest_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample IDs in output manifest")
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Duplicate output paths in output manifest")

    manifest_rows.sort(
        key=lambda row: (
            str(row["label"]),
            int(str(row["archive_id"])),
            int(str(row["sequence_id"])),
            int(row["selection_order"]),
        )
    )
    sequence_rows.sort(
        key=lambda row: (
            str(row["label"]),
            int(str(row["archive_id"])),
            int(str(row["sequence_id"])),
        )
    )

    manifest_path = output_root / "manifest.csv"
    sequence_manifest_path = output_root / "sequences.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with sequence_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)

    inventory = {
        "protocol": "wilddeepfake_test_16frames_per_sequence_v1",
        "dataset": "WildDeepfake",
        "source_repository": HF_REPOSITORY,
        "source_revision": HF_REVISION,
        "source_partitions": ["real_test", "fake_test"],
        "source_archives": source_archive_count,
        "source_archive_bytes": source_archive_bytes,
        "source_images": original_frame_count,
        "selection": {
            "frames_per_sequence": frames_per_sequence,
            "ordering": "ascending numeric frame filename",
            "indices": "floor(i * (N - 1) / (K - 1)) for i=0..K-1",
            "includes_sequence_endpoints": True,
            "random_sampling": False,
        },
        "output_sequences": dict(sequence_counts),
        "output_images": dict(image_counts),
        "output_total_sequences": len(sequence_rows),
        "output_total_images": len(manifest_rows),
        "image_properties": {
            "format": "PNG",
            "mode": "RGB",
            "width": 224,
            "height": 224,
        },
        "evaluation_note": (
            "Use sequence_uid as the independent evaluation unit and aggregate "
            "the 16 frame scores within each sequence before sequence-level metrics."
        ),
    }
    inventory_path = output_root / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    package_root = "wilddeepfake_test_16frames_v1"
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as output_tar:
        output_tar.add(
            output_root,
            arcname=package_root,
            recursive=True,
            filter=deterministic_tar_filter,
        )

    manifests_dir = export_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, manifests_dir / "manifest.csv")
    shutil.copy2(sequence_manifest_path, manifests_dir / "sequences.csv")
    shutil.copy2(inventory_path, export_dir / "inventory.json")

    checksums = {
        archive_path.name: sha256_file(archive_path),
        "inventory.json": sha256_file(export_dir / "inventory.json"),
        "manifests/manifest.csv": sha256_file(manifests_dir / "manifest.csv"),
        "manifests/sequences.csv": sha256_file(manifests_dir / "sequences.csv"),
    }
    checksum_path = export_dir / "SHA256SUMS.txt"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )

    result = {
        **inventory,
        "output_root": str(output_root),
        "export_archive": str(archive_path),
        "export_archive_bytes": archive_path.stat().st_size,
        "export_archive_sha256": checksums[archive_path.name],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
