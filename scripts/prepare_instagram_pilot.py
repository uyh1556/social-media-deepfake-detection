import argparse
import csv
import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image


METHODS = ("original", "Deepfakes", "Face2Face")
RESOLUTION_STRATA = ((640, 480), (1280, 720), (1920, 1080))
REQUIRED_COLUMNS = {
    "split",
    "label",
    "method",
    "group_id",
    "video_id",
    "source_ids",
    "source_path",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic train-only Instagram web pilot."
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
        default=Path("instagram_pipeline/pilot_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_size(path):
    with Image.open(path) as image:
        return image.size


def load_manifest(path):
    frame = pd.read_csv(
        path,
        dtype={
            "group_id": str,
            "video_id": str,
            "source_ids": str,
            "source_path": str,
        },
    )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    return frame


def choose_groups(train, data_root, rng):
    methods_by_group = train.groupby("group_id")["method"].agg(set)
    eligible_groups = sorted(
        group_id
        for group_id, methods in methods_by_group.items()
        if set(METHODS).issubset(methods)
    )

    original = train[
        train["method"].eq("original")
        & train["group_id"].isin(eligible_groups)
    ].copy()
    original["resolution"] = original["source_path"].map(
        lambda value: image_size(data_root / value)
    )

    selected = []
    used = set()
    for resolution in RESOLUTION_STRATA:
        candidates = sorted(
            set(
                original.loc[
                    original["resolution"].map(
                        lambda value: value == resolution
                    ),
                    "group_id",
                ]
            )
            - used
        )
        if not candidates:
            raise RuntimeError(
                f"No eligible train group for resolution {resolution}."
            )
        group_id = rng.choice(candidates)
        selected.append((group_id, resolution))
        used.add(group_id)
    return selected


def choose_row(rows, data_root, rng, required_resolution=None):
    candidates = rows.sort_values("source_path").copy()
    if required_resolution is not None:
        mask = candidates["source_path"].map(
            lambda value: image_size(data_root / value)
            == required_resolution
        )
        candidates = candidates[mask]
    if candidates.empty:
        raise RuntimeError("No image satisfies the pilot selection rule.")
    return candidates.iloc[rng.randrange(len(candidates))]


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    upload_dir = output_dir / "upload"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )

    manifest = load_manifest(manifest_path)
    train = manifest[manifest["split"].eq("train")].reset_index(drop=True)
    if train.empty:
        raise ValueError("The train split is empty.")

    rng = random.Random(args.seed)
    selected_groups = choose_groups(train, data_root, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir()

    manifest_rows = []
    upload_rows = []
    method_code = {
        "original": "original",
        "Deepfakes": "deepfakes",
        "Face2Face": "face2face",
    }

    upload_order = 0
    for stratum_index, (group_id, resolution) in enumerate(
        selected_groups, start=1
    ):
        for method in METHODS:
            upload_order += 1
            rows = train[
                train["group_id"].eq(group_id)
                & train["method"].eq(method)
            ]
            required_resolution = resolution if method == "original" else None
            row = choose_row(rows, data_root, rng, required_resolution)
            source = data_root / row["source_path"]
            if not source.is_file():
                raise FileNotFoundError(source)

            sample_id = (
                f"igweb_pilot_v1_s{stratum_index:02d}_"
                f"{method_code[method]}"
            )
            destination = upload_dir / f"{sample_id}.png"
            shutil.copy2(source, destination)

            source_hash = sha256_file(source)
            copied_hash = sha256_file(destination)
            if source_hash != copied_hash:
                raise RuntimeError(f"Copy hash mismatch: {sample_id}")

            width, height = image_size(source)
            caption = (
                "Academic deepfake-detection research. "
                f"sample_id={sample_id}"
            )
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "upload_order": upload_order,
                    "split": row["split"],
                    "label": row["label"],
                    "method": row["method"],
                    "group_id": row["group_id"],
                    "video_id": row["video_id"],
                    "source_ids": row["source_ids"],
                    "resolution_stratum": f"{resolution[0]}x{resolution[1]}",
                    "source_path": row["source_path"],
                    "upload_path": str(destination.relative_to(output_dir)),
                    "source_width": width,
                    "source_height": height,
                    "source_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                    "caption": caption,
                }
            )
            upload_rows.append(
                {
                    "sample_id": sample_id,
                    "upload_order": upload_order,
                    "caption": caption,
                    "uploaded_at_utc": "",
                    "post_url": "",
                    "status": "pending",
                    "notes": "",
                }
            )

    manifest_fields = list(manifest_rows[0])
    upload_fields = list(upload_rows[0])
    write_csv(output_dir / "pilot_manifest.csv", manifest_rows, manifest_fields)
    write_csv(output_dir / "upload_log.csv", upload_rows, upload_fields)

    protocol = {
        "name": "instagram_web_pilot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Technical upload/download and ID-pairing validation only",
        "source_split": "train",
        "split_protocol": "custom_source_aware_v1",
        "seed": args.seed,
        "sample_count": len(manifest_rows),
        "methods": list(METHODS),
        "original_resolution_strata": [
            f"{width}x{height}" for width, height in RESOLUTION_STRATA
        ],
        "upload_client": "Instagram web",
        "post_type": "single-image feed post",
        "editing": "No filter, enhancement, or manual crop",
        "account": "Dedicated research account; identifier not recorded",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "data_root": str(data_root),
        "selected_groups": [group_id for group_id, _ in selected_groups],
    }
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)

    print(f"Prepared {len(manifest_rows)} images in {upload_dir}")
    print(f"Pilot manifest: {output_dir / 'pilot_manifest.csv'}")
    print(f"Upload log: {output_dir / 'upload_log.csv'}")
    print("Selected groups:", protocol["selected_groups"])


if __name__ == "__main__":
    main()
