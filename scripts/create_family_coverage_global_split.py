#!/usr/bin/env python3
"""Build the FF-only global source split for the M0-M7 family study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")
PROTOCOL_NAME = "family_coverage_global_source_aware_v1"
METHODS = {
    "simswap": ("SimSwap", "FS", "seen", "pair"),
    "blendface": ("BlendFace", "FS", "seen", "pair"),
    "inswap": ("InSwapper", "FS", "protected_unseen", "pair"),
    "facedancer": ("FaceDancer", "FS", "protected_unseen", "pair"),
    "wav2lip": ("Wav2Lip", "FR", "seen", "appearance_external_driver"),
    "fomm": ("FOMM", "FR", "seen", "driver_appearance"),
    "sadtalker": (
        "SadTalker", "FR", "protected_unseen", "appearance_external_driver"
    ),
    "hyperreenact": (
        "HyperReenact",
        "FR",
        "protected_unseen",
        "driver_appearance",
    ),
    "stylegan3": ("StyleGAN3", "EFS", "seen", "latent_seed"),
    "dit": ("DiT", "EFS", "seen", "latent_seed"),
    "styleganxl": ("StyleGAN-XL", "EFS", "protected_unseen", "latent_seed"),
    "pixart": ("PixArt-alpha", "EFS", "protected_unseen", "pixart_sample"),
}
PLANNED_MISSING_METHODS: tuple[str, ...] = ()
FIELDS = (
    "sample_id",
    "split",
    "label",
    "method",
    "family",
    "role",
    "module",
    "domain",
    "compression",
    "preprocessing",
    "group_id",
    "video_id",
    "source_ids",
    "driver_id",
    "generation_id",
    "frame_index",
    "content_sha256",
    "source_path",
    "local_source_path",
    "physical_splits",
    "duplicate_copies",
)
MODULE_NAMES = (
    "real_trainval",
    "ffpp_fake_trainval",
    "df40_fs_trainval",
    "df40_fr_trainval",
    "df40_efs_trainval",
    "all_methods_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_video_id(value: str) -> str:
    return value.removeprefix("temp_")


def validate_ff_id(value: str, context: str) -> None:
    if len(value) != 3 or not value.isdigit():
        raise ValueError(f"Invalid FF++ ID in {context}: {value!r}")


def parse_video_id(kind: str, video_id: str) -> tuple[tuple[str, ...], str]:
    value = normalize_video_id(video_id)
    if "##" in value:
        parts = value.split("##", 1)
    else:
        parts = value.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected {kind} video ID: {video_id}")
    first, second = parts
    validate_ff_id(first, video_id)
    if kind == "pair":
        validate_ff_id(second, video_id)
        return (first, second), ""
    if not second:
        raise ValueError(f"Missing driver ID: {video_id}")
    if kind == "appearance_external_driver":
        return (first,), second
    if kind == "driver_appearance":
        validate_ff_id(second, video_id)
        return (second,), first
    raise ValueError(f"Unsupported identity parser: {kind}")


def flatten_string_lists(value: object) -> list[str]:
    result: list[str] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            if current and all(isinstance(item, str) for item in current):
                result.extend(current)
            else:
                stack.extend(current)
    return result


def json_fake_keys(path: Path) -> tuple[set[tuple[str, str]], dict[str, int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    root = next(iter(document.values()))
    fake_nodes = [value for key, value in root.items() if key.lower().endswith("_fake")]
    if len(fake_nodes) != 1:
        raise RuntimeError(f"Cannot identify one fake branch in {path}")
    fake = fake_nodes[0]
    keys: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for physical_split in ("train", "test"):
        paths = flatten_string_lists(fake.get(physical_split, {}))
        split_keys = {
            (normalize_video_id(Path(value).parent.name), Path(value).name)
            for value in paths
        }
        counts[physical_split] = len(split_keys)
        keys.update(split_keys)
    return keys, counts


def load_anchor(path: Path, project_root: Path) -> tuple[list[dict], dict, dict]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "sample_id",
        "split",
        "label",
        "method",
        "family",
        "domain",
        "compression",
        "preprocessing",
        "video_id",
        "source_ids",
        "frame_index",
        "local_source_path",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Anchor columns missing: {sorted(missing)}")
    frame = frame[frame["method"].isin(["original", "Deepfakes", "Face2Face"])]
    source_to_split: dict[str, str] = {}
    source_to_anchor_group: dict[str, str] = {}
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        local_path = project_root / record["local_source_path"]
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        sources = tuple(value for value in record["source_ids"].split("|") if value)
        for source_id in sources:
            old_split = source_to_split.setdefault(source_id, record["split"])
            old_group = source_to_anchor_group.setdefault(source_id, record["group_id"])
            if old_split != record["split"] or old_group != record["group_id"]:
                raise RuntimeError(f"Inconsistent anchor source: {source_id}")
        family = "real" if record["label"] == "real" else (
            "FS" if record["method"] == "Deepfakes" else "FR"
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "label": record["label"],
                "method": record["method"],
                "family": family,
                "role": "real" if record["label"] == "real" else "baseline_seen",
                "domain": "ff",
                "compression": "c23",
                "preprocessing": record["preprocessing"],
                "video_id": record["video_id"],
                "source_ids": sources,
                "driver_id": "",
                "generation_id": "",
                "frame_index": record["frame_index"],
                "content_sha256": sha256_file(local_path),
                "local_source_path": record["local_source_path"],
                "physical_splits": "anchor",
                "duplicate_copies": 1,
                "graph_nodes": tuple(f"ff:{value}" for value in sources),
                "_path": local_path,
            }
        )
    return rows, source_to_split, source_to_anchor_group


def candidate_rank(value: dict) -> tuple[int, str]:
    return (0 if value["physical_split"] == "train" else 1, value["path"].as_posix())


def deterministic_efs_split(generation_id: str) -> str:
    value = int(hashlib.sha256(f"seed42:{generation_id}".encode()).hexdigest()[:8], 16)
    ratio = value / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def efs_generation_id(source_kind: str, filename: str) -> str:
    stem = Path(filename).stem
    if source_kind == "latent_seed":
        seed = stem.removeprefix("seed")
        if not seed.isdigit():
            raise ValueError(f"Invalid latent-seed filename: {filename}")
        return f"latent:{int(seed)}"
    if source_kind == "pixart_sample":
        return f"pixart:{stem}"
    raise ValueError(f"Unsupported EFS parser: {source_kind}")


def collect_efs_method(
    data_root: Path,
    method_key: str,
    project_root: Path,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    display, family, role, source_kind = METHODS[method_key]
    method_root = data_root / "df40" / method_key
    json_path = method_root / f"{method_key}_ff.json"
    allowed_keys, json_counts = json_fake_keys(json_path)
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    physical_pngs = 0
    ignored_physical = 0
    for physical_split in ("train", "test"):
        frames_root = method_root / physical_split / "ff" / "frames"
        if not frames_root.is_dir():
            raise FileNotFoundError(frames_root)
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            logical_video = normalize_video_id(video_dir.name)
            for image_path in sorted(video_dir.glob("*.png")):
                physical_pngs += 1
                key = (logical_video, image_path.name)
                if key not in allowed_keys:
                    ignored_physical += 1
                    continue
                candidates[key].append(
                    {"physical_split": physical_split, "path": image_path}
                )
    missing_json = allowed_keys - set(candidates)
    if missing_json:
        raise FileNotFoundError(
            f"{method_key} is missing {len(missing_json)} JSON-listed images; "
            f"first={sorted(missing_json)[:3]}"
        )

    by_hash: dict[str, list[dict]] = defaultdict(list)
    physical_duplicates: list[dict] = []
    for (outer_id, filename), copies in sorted(candidates.items()):
        copies.sort(key=candidate_rank)
        hashes = [sha256_file(copy["path"]) for copy in copies]
        if len(set(hashes)) != 1:
            raise RuntimeError(
                f"Non-identical physical duplicate: {method_key}/{outer_id}/{filename}"
            )
        content_hash = hashes[0]
        if len(copies) > 1:
            physical_duplicates.append(
                {
                    "method": display,
                    "video_id": outer_id,
                    "filename": filename,
                    "copies": len(copies),
                    "sha256": content_hash,
                    "reason": "same_logical_path_multiple_physical_splits",
                }
            )
        by_hash[content_hash].append(
            {
                "outer_id": outer_id,
                "filename": filename,
                "generation_id": efs_generation_id(source_kind, filename),
                "copies": copies,
            }
        )

    rows: list[dict] = []
    exclusions: list[dict] = []
    duplicates = list(physical_duplicates)
    for content_hash, logical_copies in sorted(by_hash.items()):
        generation_ids = {copy["generation_id"] for copy in logical_copies}
        if len(generation_ids) != 1:
            raise RuntimeError(
                f"One EFS image has multiple generation IDs: {display} "
                f"{sorted(generation_ids)}"
            )
        generation_id = generation_ids.pop()
        assigned_split = deterministic_efs_split(generation_id)
        all_physical = [item for copy in logical_copies for item in copy["copies"]]
        all_physical.sort(key=candidate_rank)
        canonical = all_physical[0]
        if len(logical_copies) > 1:
            duplicates.append(
                {
                    "method": display,
                    "video_id": generation_id,
                    "filename": logical_copies[0]["filename"],
                    "copies": len(all_physical),
                    "sha256": content_hash,
                    "reason": "same_content_multiple_outer_folders",
                }
            )
        if role == "protected_unseen" and assigned_split != "test":
            exclusions.append(
                {
                    "method": display,
                    "video_id": generation_id,
                    "filename": logical_copies[0]["filename"],
                    "reason": "protected_unseen_non_test_generation",
                }
            )
            continue
        portable_id = generation_id.replace(":", "_")
        rows.append(
            {
                "sample_id": f"{method_key}:{content_hash}",
                "split": assigned_split,
                "label": "fake",
                "method": display,
                "family": family,
                "role": role,
                "domain": "ff",
                "compression": "c23",
                "preprocessing": "df40_efs_generated_face_256",
                "video_id": portable_id,
                "source_ids": (),
                "driver_id": "",
                "generation_id": generation_id,
                "frame_index": Path(logical_copies[0]["filename"]).stem,
                "content_sha256": content_hash,
                "local_source_path": canonical["path"].relative_to(project_root).as_posix(),
                "physical_splits": "|".join(
                    split
                    for split in ("train", "test")
                    if split in {copy["physical_split"] for copy in all_physical}
                ),
                "duplicate_copies": len(all_physical),
                "graph_nodes": (f"efs:{generation_id}",),
                "_path": canonical["path"],
            }
        )
    audit = {
        "method": display,
        "family": family,
        "role": role,
        "json_unique_paths": len(allowed_keys),
        "json_by_physical_split": json_counts,
        "physical_pngs": physical_pngs,
        "ignored_physical_pngs_not_in_json": ignored_physical,
        "unique_content_images": len(by_hash),
        "content_duplicates_removed": len(allowed_keys) - len(by_hash),
        "retained_images": len(rows),
        "excluded_images": len(exclusions),
        "split_policy": "deterministic 70/15/15 by shared EFS generation ID",
    }
    return rows, exclusions, duplicates, audit


def collect_method(
    data_root: Path,
    method_key: str,
    source_to_split: dict[str, str],
    project_root: Path,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    display, family, role, source_kind = METHODS[method_key]
    if family == "EFS":
        return collect_efs_method(data_root, method_key, project_root)
    method_root = data_root / "df40" / method_key
    json_path = method_root / f"{method_key}_ff.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    allowed_keys, json_counts = json_fake_keys(json_path)
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    physical_pngs = 0
    ignored_physical = 0
    for physical_split in ("train", "test"):
        frames_root = method_root / physical_split / "ff" / "frames"
        if not frames_root.is_dir():
            raise FileNotFoundError(frames_root)
        for video_dir in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            logical_video = normalize_video_id(video_dir.name)
            for image_path in sorted(video_dir.glob("*.png")):
                physical_pngs += 1
                key = (logical_video, image_path.name)
                if key not in allowed_keys:
                    ignored_physical += 1
                    continue
                candidates[key].append(
                    {
                        "physical_split": physical_split,
                        "path": image_path,
                    }
                )
    missing_json = allowed_keys - set(candidates)
    if missing_json:
        raise FileNotFoundError(
            f"{method_key} is missing {len(missing_json)} JSON-listed images; "
            f"first={sorted(missing_json)[:3]}"
        )

    rows: list[dict] = []
    exclusions: list[dict] = []
    duplicates: list[dict] = []
    for (video_id, filename), copies in sorted(candidates.items()):
        copies.sort(key=candidate_rank)
        canonical = copies[0]
        hashes = [sha256_file(copy["path"]) for copy in copies]
        if len(set(hashes)) != 1:
            raise RuntimeError(
                f"Non-identical duplicate: {method_key}/{video_id}/{filename}"
            )
        content_hash = hashes[0]
        if len(copies) > 1:
            duplicates.append(
                {
                    "method": display,
                    "video_id": video_id,
                    "filename": filename,
                    "copies": len(copies),
                    "sha256": content_hash,
                    "reason": "same_logical_path_multiple_physical_splits",
                }
            )
        source_ids, driver_id = parse_video_id(source_kind, video_id)
        anchored_driver = (
            driver_id
            if len(driver_id) == 3 and driver_id.isdigit()
            else ""
        )
        anchor_ids = source_ids + ((anchored_driver,) if anchored_driver else ())
        graph_nodes = tuple(f"ff:{value}" for value in source_ids)
        if driver_id:
            graph_nodes += (
                f"ff:{driver_id}"
                if anchored_driver
                else f"driver:{driver_id}",
            )
        memberships = {source_to_split.get(value) for value in anchor_ids}
        if None in memberships:
            reason = "source_or_driver_not_in_anchor"
        elif len(memberships) != 1:
            reason = "source_or_driver_crosses_anchor_splits"
        else:
            reason = ""
        assigned_split = next(iter(memberships)) if len(memberships) == 1 else ""
        if not reason and role == "protected_unseen" and assigned_split != "test":
            reason = "protected_unseen_non_test_source"
        if reason:
            exclusions.append(
                {
                    "method": display,
                    "video_id": video_id,
                    "filename": filename,
                    "reason": reason,
                }
            )
            continue
        local_path = canonical["path"].relative_to(project_root).as_posix()
        rows.append(
            {
                "sample_id": f"{method_key}:{video_id}:{filename}",
                "split": assigned_split,
                "label": "fake",
                "method": display,
                "family": family,
                "role": role,
                "domain": "ff",
                "compression": "c23",
                "preprocessing": (
                    "df40_efs_generated_face_256"
                    if family == "EFS"
                    else "df40_provided_aligned_face_256"
                ),
                "video_id": video_id,
                "source_ids": source_ids,
                "driver_id": driver_id,
                "generation_id": "",
                "frame_index": Path(filename).stem,
                "content_sha256": content_hash,
                "local_source_path": local_path,
                "physical_splits": "|".join(
                    split
                    for split in ("train", "test")
                    if split in {copy["physical_split"] for copy in copies}
                ),
                "duplicate_copies": len(copies),
                "graph_nodes": graph_nodes,
                "_path": canonical["path"],
            }
        )
    priority = {"train": 0, "val": 1, "test": 2}
    by_content: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_content[row["content_sha256"]].append(row)
    deduplicated_rows: list[dict] = []
    content_duplicates_removed = 0
    for content_hash, content_rows in by_content.items():
        content_rows.sort(
            key=lambda row: (
                -priority[row["split"]],
                row["_path"].as_posix(),
            )
        )
        retained = content_rows[0]
        deduplicated_rows.append(retained)
        for duplicate in content_rows[1:]:
            content_duplicates_removed += 1
            exclusions.append(
                {
                    "method": display,
                    "video_id": duplicate["video_id"],
                    "filename": duplicate["_path"].name,
                    "reason": (
                        "content_duplicate_cross_split"
                        if duplicate["split"] != retained["split"]
                        else "content_duplicate_same_split"
                    ),
                }
            )
            duplicates.append(
                {
                    "method": display,
                    "video_id": duplicate["video_id"],
                    "filename": duplicate["_path"].name,
                    "copies": len(content_rows),
                    "sha256": content_hash,
                    "reason": "same_content_multiple_logical_paths",
                }
            )
    rows = deduplicated_rows
    audit = {
        "method": display,
        "family": family,
        "role": role,
        "json_unique_images": len(allowed_keys),
        "json_membership_policy": "union of JSON train and test fake paths",
        "json_by_physical_split": json_counts,
        "physical_pngs": physical_pngs,
        "ignored_physical_pngs_not_in_json": ignored_physical,
        "logical_duplicates": len(duplicates),
        "content_duplicates_removed": content_duplicates_removed,
        "retained_images": len(rows),
        "excluded_images": len(exclusions),
    }
    return rows, exclusions, duplicates, audit


def protect_external_drivers(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep a non-FF driving clip in only its most protected split."""
    priority = {"train": 0, "val": 1, "test": 2}
    driver_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        driver_id = row["driver_id"]
        if driver_id and not (len(driver_id) == 3 and driver_id.isdigit()):
            driver_splits[driver_id].add(row["split"])
    protected_split = {
        driver_id: max(splits, key=lambda value: priority[value])
        for driver_id, splits in driver_splits.items()
    }
    retained = []
    excluded = []
    for row in rows:
        driver_id = row["driver_id"]
        if (
            driver_id
            and driver_id in protected_split
            and row["split"] != protected_split[driver_id]
        ):
            excluded.append(
                {
                    "method": row["method"],
                    "video_id": row["video_id"],
                    "filename": row["_path"].name,
                    "reason": "external_driver_used_in_protected_split",
                }
            )
        else:
            retained.append(row)
    return retained, excluded


def deduplicate_global_content(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Keep each decoded file payload once across all methods and paths."""
    role_priority = {
        "protected_unseen": 0,
        "rotation_candidate": 1,
        "seen": 1,
        "baseline_seen": 2,
        "real": 3,
    }
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["content_sha256"]].append(row)
    retained: list[dict] = []
    exclusions: list[dict] = []
    duplicates: list[dict] = []
    for content_hash, content_rows in by_hash.items():
        labels = {row["label"] for row in content_rows}
        splits = {row["split"] for row in content_rows}
        if len(labels) != 1 or len(splits) != 1:
            raise RuntimeError(
                f"Cross-label/split content duplicate: {content_hash} "
                f"labels={labels}, splits={splits}"
            )
        content_rows.sort(
            key=lambda row: (
                role_priority[row["role"]],
                row["method"],
                row["_path"].as_posix(),
            )
        )
        retained.append(content_rows[0])
        for duplicate in content_rows[1:]:
            exclusions.append(
                {
                    "method": duplicate["method"],
                    "video_id": duplicate["video_id"],
                    "filename": duplicate["_path"].name,
                    "reason": "global_content_duplicate_same_split",
                }
            )
            duplicates.append(
                {
                    "method": duplicate["method"],
                    "video_id": duplicate["video_id"],
                    "filename": duplicate["_path"].name,
                    "copies": len(content_rows),
                    "sha256": content_hash,
                    "reason": "same_content_across_methods_or_paths",
                }
            )
    return retained, exclusions, duplicates


def attach_global_groups(rows: list[dict]) -> int:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        nodes = row["graph_nodes"]
        for first in nodes:
            adjacency[first]
            for second in nodes:
                if first != second:
                    adjacency[first].add(second)
    node_to_group: dict[str, str] = {}
    components = 0
    for start in sorted(adjacency):
        if start in node_to_group:
            continue
        queue = deque([start])
        seen = {start}
        nodes = []
        while queue:
            node = queue.popleft()
            nodes.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        group_id = "global_" + hashlib.sha256(
            "|".join(sorted(nodes)).encode("utf-8")
        ).hexdigest()[:12]
        for node in nodes:
            node_to_group[node] = group_id
        components += 1
    for row in rows:
        groups = {node_to_group[value] for value in row["graph_nodes"]}
        if len(groups) != 1:
            raise RuntimeError(f"Row spans global groups: {row['sample_id']}")
        row["group_id"] = groups.pop()
    return components


def module_name(row: dict) -> str:
    if row["split"] == "test":
        return "all_methods_test"
    if row["label"] == "real":
        return "real_trainval"
    if row["method"] in {"Deepfakes", "Face2Face"}:
        return "ffpp_fake_trainval"
    if row["family"] == "FS":
        return "df40_fs_trainval"
    if row["family"] == "FR":
        return "df40_fr_trainval"
    if row["family"] == "EFS":
        return "df40_efs_trainval"
    raise RuntimeError(f"Cannot assign module: {row['sample_id']}")


def public_row(row: dict) -> dict[str, object]:
    method_key = row["method"].lower().replace("-", "").replace(" ", "")
    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "label": row["label"],
        "method": row["method"],
        "family": row["family"],
        "role": row["role"],
        "module": row["module"],
        "domain": row["domain"],
        "compression": row["compression"],
        "preprocessing": row["preprocessing"],
        "group_id": row["group_id"],
        "video_id": row["video_id"],
        "source_ids": "|".join(row["source_ids"]),
        "driver_id": row["driver_id"],
        "generation_id": row["generation_id"],
        "frame_index": row["frame_index"],
        "content_sha256": row["content_sha256"],
        "source_path": (
            Path("images") / method_key / row["video_id"] / row["_path"].name
        ).as_posix(),
        "local_source_path": row["local_source_path"],
        "physical_splits": row["physical_splits"],
        "duplicate_copies": row["duplicate_copies"],
    }


def validate(rows: list[dict], project_root: Path) -> dict[str, int]:
    frame = pd.DataFrame(rows)
    exploded_sources = frame.assign(
        source_ids=frame["source_ids"].str.split("|")
    ).explode("source_ids")
    exploded_sources = exploded_sources[exploded_sources["source_ids"] != ""]
    checks = {
        "sample_id_duplicates": int(frame["sample_id"].duplicated().sum()),
        "portable_path_duplicates": int(frame["source_path"].duplicated().sum()),
        "missing_local_files": sum(
            not (project_root / value).is_file()
            for value in frame["local_source_path"]
        ),
        "source_id_overlap": int(
            exploded_sources.groupby("source_ids")["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "driver_id_overlap": int(
            frame[frame["driver_id"] != ""]
            .groupby("driver_id")["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "group_overlap": int(
            frame.groupby("group_id")["split"].nunique().gt(1).sum()
        ),
        "method_video_overlap": int(
            frame.groupby(["method", "video_id"])["split"]
            .nunique()
            .gt(1)
            .sum()
        ),
        "content_hash_overlap": int(
            frame.groupby("content_sha256")["split"].nunique().gt(1).sum()
        ),
        "content_hash_duplicate_rows": int(
            frame["content_sha256"].duplicated().sum()
        ),
        "content_label_conflicts": int(
            frame.groupby("content_sha256")["label"].nunique().gt(1).sum()
        ),
        "protected_unseen_non_test": int(
            ((frame["role"] == "protected_unseen") & (frame["split"] != "test")).sum()
        ),
        "non_ff_domain": int((frame["domain"] != "ff").sum()),
    }
    if any(checks.values()):
        raise RuntimeError(f"Global split validation failed: {checks}")
    return checks


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    anchor = args.anchor_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    base_rows, source_to_split, _ = load_anchor(anchor, project_root)
    all_rows = list(base_rows)
    exclusions: list[dict] = []
    duplicates: list[dict] = []
    audits: dict[str, dict] = {}
    for method_key in METHODS:
        rows, method_exclusions, method_duplicates, audit = collect_method(
            data_root, method_key, source_to_split, project_root
        )
        all_rows.extend(rows)
        exclusions.extend(method_exclusions)
        duplicates.extend(method_duplicates)
        audits[method_key] = audit
        print(
            f"{audit['method']}: retained={audit['retained_images']}, "
            f"excluded={audit['excluded_images']}",
            flush=True,
        )
    all_rows, driver_exclusions = protect_external_drivers(all_rows)
    exclusions.extend(driver_exclusions)
    all_rows, global_exclusions, global_duplicates = deduplicate_global_content(
        all_rows
    )
    exclusions.extend(global_exclusions)
    duplicates.extend(global_duplicates)
    components = attach_global_groups(all_rows)
    for row in all_rows:
        row["module"] = module_name(row)
    public = [public_row(row) for row in all_rows]
    public.sort(
        key=lambda row: (
            SPLITS.index(str(row["split"])),
            str(row["label"]),
            str(row["method"]),
            str(row["video_id"]),
            str(row["frame_index"]),
        )
    )
    checks = validate(public, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", public)
    for method in sorted({str(row["method"]) for row in public}):
        key = method.lower().replace("-", "").replace(" ", "")
        write_csv(
            output_dir / "methods" / f"{key}.csv",
            [row for row in public if row["method"] == method],
        )
    for module in MODULE_NAMES:
        write_csv(
            output_dir / "modules" / f"{module}.csv",
            [row for row in public if row["module"] == module],
        )
    exclusion_path = output_dir / "excluded_samples.csv"
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions).to_csv(exclusion_path, index=False)
    pd.DataFrame(duplicates).to_csv(output_dir / "duplicate_samples.csv", index=False)
    frame = pd.DataFrame(public)
    test_methods = sorted(
        frame[(frame["split"] == "test") & (frame["label"] == "fake")]["method"].unique()
    )
    summary = {
        "protocol": PROTOCOL_NAME,
        "domain": "ff_only",
        "anchor_manifest": str(anchor),
        "anchor_manifest_sha256": sha256_file(anchor),
        "global_components": components,
        "images": len(public),
        "planned_missing_methods": list(PLANNED_MISSING_METHODS),
        "test_fake_methods": test_methods,
        "test_fake_method_count": len(test_methods),
        "splits": {
            split: {
                "images": int((frame["split"] == split).sum()),
                "by_method": frame[frame["split"] == split]["method"].value_counts().to_dict(),
            }
            for split in SPLITS
        },
        "modules": {
            module: int((frame["module"] == module).sum())
            for module in MODULE_NAMES
        },
        "method_audits": audits,
        "exclusions_by_method_reason": {
            f"{method}:{reason}": int(count)
            for (method, reason), count in pd.DataFrame(exclusions).groupby(
                ["method", "reason"]
            ).size().items()
        },
        "duplicate_logical_samples": len(duplicates),
        "checks": checks,
    }
    write_json(output_dir / "split_summary.json", summary)
    write_json(
        output_dir / "method_registry.json",
        {
            key: {
                "display_name": value[0],
                "family": value[1],
                "role": value[2],
                "source_parser": value[3],
            }
            for key, value in METHODS.items()
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
