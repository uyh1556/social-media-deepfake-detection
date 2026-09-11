import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image


METHODS = ("original", "Deepfakes", "Face2Face")
METHOD_CODES = {
    "original": "original",
    "Deepfakes": "deepfakes",
    "Face2Face": "face2face",
}
RESOLUTION_QUOTAS = {
    (640, 480): 4,
    (1280, 720): 4,
    (1920, 1080): 3,
    (854, 480): 2,
    (656, 480): 2,
    (600, 480): 1,
    (654, 480): 1,
    (960, 720): 1,
    (720, 480): 1,
    (800, 480): 1,
}
REQUIRED_COLUMNS = {
    "split",
    "label",
    "method",
    "group_id",
    "video_id",
    "source_ids",
    "source_path",
}
STEM_PATTERN = re.compile(
    r"^(?P<first>\d{3})(?:_(?P<second>\d{3}))?_frame(?P<frame>\d+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 20 matched validation triplets as exact-resolution "
            "Instagram carousel batches of at most 10 images."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_split/source_aware/manifest.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("instagram_pipeline/calibration_v2"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def match_key(row):
    match = STEM_PATTERN.fullmatch(Path(row.source_path).stem)
    if match is None:
        raise ValueError(f"Unexpected source filename: {row.source_path}")
    return "|".join(
        (
            row.group_id,
            match.group("first"),
            str(int(match.group("frame"))),
        )
    )


def load_candidates(manifest_path, data_root):
    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "group_id": str,
            "video_id": str,
            "source_ids": str,
            "source_path": str,
        },
    )
    if missing := REQUIRED_COLUMNS - set(manifest.columns):
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    validation = manifest[manifest["split"].eq("val")].copy()
    if validation.empty:
        raise ValueError("Validation split is empty.")
    validation["match_key"] = [
        match_key(row) for row in validation.itertuples()
    ]

    by_method = {
        method: frame.set_index("match_key", verify_integrity=True)
        for method, frame in validation.groupby("method")
    }
    if set(METHODS) - set(by_method):
        raise ValueError("Validation split does not contain all methods.")
    common_keys = set.intersection(
        *(set(by_method[method].index) for method in METHODS)
    )
    rows = []
    for key in sorted(common_keys):
        original = by_method["original"].loc[key]
        original_path = data_root / original.source_path
        if not original_path.is_file():
            raise FileNotFoundError(original_path)
        with Image.open(original_path) as image:
            resolution = image.size
        group_id, first_id, frame_number = key.split("|")
        rows.append(
            {
                "key": key,
                "group_id": group_id,
                "first_id": first_id,
                "frame_number": int(frame_number),
                "resolution": resolution,
            }
        )
    return rows, by_method


def select_triplets(candidates, seed):
    rng = random.Random(seed)
    selected = []
    used_groups = set()
    for resolution, quota in RESOLUTION_QUOTAS.items():
        eligible = [
            item
            for item in candidates
            if item["resolution"] == resolution
            and item["group_id"] not in used_groups
        ]
        rng.shuffle(eligible)
        if len(eligible) < quota:
            raise RuntimeError(
                f"Not enough unique groups for {resolution}: "
                f"need {quota}, found {len(eligible)}"
            )
        for item in eligible[:quota]:
            selected.append(item)
            used_groups.add(item["group_id"])
    if len(selected) != 20 or len(used_groups) != 20:
        raise RuntimeError("Expected 20 triplets from 20 unique groups.")
    rng.shuffle(selected)
    return selected


def balanced_chunks(items, maximum=10):
    chunk_count = math.ceil(len(items) / maximum)
    base, extra = divmod(len(items), chunk_count)
    chunks = []
    start = 0
    for number in range(chunk_count):
        size = base + (1 if number < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    upload_dir = output_dir / "upload"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    candidates, by_method = load_candidates(manifest_path, data_root)
    selected = select_triplets(candidates, args.seed)
    output_dir.mkdir(parents=True)
    upload_dir.mkdir()

    sample_rows = []
    flat_samples = []
    for triplet_number, item in enumerate(selected, start=1):
        for method in METHODS:
            row = by_method[method].loc[item["key"]]
            flat_samples.append(
                {
                    "triplet_id": f"igweb_cal_v2_t{triplet_number:02d}",
                    "triplet_number": triplet_number,
                    "resolution_stratum": (
                        f"{item['resolution'][0]}x{item['resolution'][1]}"
                    ),
                    "matched_first_id": item["first_id"],
                    "matched_frame_number": item["frame_number"],
                    "method": method,
                    "row": row,
                }
            )

    grouped = {}
    for item in flat_samples:
        method = item["method"]
        row = item["row"]
        source = data_root / row.source_path
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            width, height = image.size
        item["source"] = source
        item["width"] = width
        item["height"] = height
        grouped.setdefault((width, height), []).append(item)

    batch_specs = []
    for resolution in sorted(grouped):
        items = sorted(
            grouped[resolution],
            key=lambda item: (
                item["triplet_number"], METHODS.index(item["method"])
            ),
        )
        for chunk in balanced_chunks(items):
            batch_specs.append((resolution, chunk))

    batch_rows = []
    upload_order = 0
    for batch_number, (resolution, batch) in enumerate(
        batch_specs, start=1
    ):
        batch_id = f"igweb_cal_v2_b{batch_number:02d}"
        width, height = resolution
        for carousel_position, item in enumerate(batch, start=1):
            upload_order += 1
            method = item["method"]
            row = item["row"]
            sample_id = f"{item['triplet_id']}_{METHOD_CODES[method]}"
            source = item["source"]

            batch_dir = upload_dir / f"batch_{batch_number:02d}"
            batch_dir.mkdir(exist_ok=True)
            destination = batch_dir / (
                f"{carousel_position:02d}_{sample_id}.png"
            )
            shutil.copy2(source, destination)
            source_hash = sha256_file(source)
            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"Copy hash mismatch: {sample_id}")

            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "upload_order": upload_order,
                    "batch_id": batch_id,
                    "batch_number": batch_number,
                    "carousel_position": carousel_position,
                    "batch_resolution": f"{width}x{height}",
                    "triplet_id": item["triplet_id"],
                    "triplet_number": item["triplet_number"],
                    "split": row.split,
                    "label": row.label,
                    "method": method,
                    "group_id": row.group_id,
                    "video_id": row.video_id,
                    "source_ids": row.source_ids,
                    "matched_first_id": item["matched_first_id"],
                    "matched_frame_number": item["matched_frame_number"],
                    "resolution_stratum": item["resolution_stratum"],
                    "source_path": row.source_path,
                    "upload_path": str(
                        destination.relative_to(output_dir)
                    ),
                    "source_width": width,
                    "source_height": height,
                    "source_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                }
            )

        batch_rows.append(
            {
                "batch_id": batch_id,
                "batch_number": batch_number,
                "image_count": len(batch),
                "source_resolution": f"{width}x{height}",
                "caption": (
                    "Academic deepfake-detection research. "
                    f"batch_id={batch_id}"
                ),
                "uploaded_at_utc": "",
                "post_url": "",
                "status": "pending",
                "notes": "",
            }
        )

    write_csv(output_dir / "calibration_manifest.csv", sample_rows)
    write_csv(output_dir / "batch_manifest.csv", batch_rows)
    protocol = {
        "name": "instagram_web_calibration_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Characterize real Instagram transformation across methods and "
            "source resolutions before preprocessing-model retraining"
        ),
        "source_split": "val",
        "split_protocol": "custom_source_aware_v1",
        "seed": args.seed,
        "triplets": 20,
        "sample_count": 60,
        "unique_source_groups": 20,
        "selection": (
            "exact matched first video ID and frame number across original, "
            "Deepfakes, and Face2Face; one triplet per source group; "
            "fixed original-resolution quotas"
        ),
        "original_resolution_quotas": {
            f"{width}x{height}": quota
            for (width, height), quota in RESOLUTION_QUOTAS.items()
        },
        "upload_client": "Instagram web",
        "post_type": (
            f"{len(batch_rows)} exact-resolution carousel posts with at "
            "most ten images each"
        ),
        "batching": (
            "Every carousel contains one exact source resolution only; "
            "groups larger than ten are split into balanced chunks"
        ),
        "editing": "No filter, enhancement, manual crop, or caption change",
        "media_quality": "Highest-quality upload enabled",
        "pairing": "batch_id in caption plus one-based carousel position",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(sample_rows)} images in {upload_dir}")
    print(f"Batches: {len(batch_rows)} exact-resolution carousels")
    print(f"Sample manifest: {output_dir / 'calibration_manifest.csv'}")
    print(f"Batch manifest: {output_dir / 'batch_manifest.csv'}")


if __name__ == "__main__":
    main()
