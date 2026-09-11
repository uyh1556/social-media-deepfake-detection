import argparse
import csv
import hashlib
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import instaloader
import pandas as pd
from PIL import Image


BATCH_ID_PATTERN = re.compile(r"\bbatch_id=([A-Za-z0-9_-]+)\b")
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download ordered Instagram calibration carousel items and "
            "pair them by caption batch_id plus carousel position."
        )
    )
    parser.add_argument("--username", required=True)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("instagram_pipeline/calibration_v2"),
    )
    parser.add_argument("--session-file", type=Path, default=None)
    parser.add_argument("--max-posts", type=int, default=20)
    return parser.parse_args()


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def inspect_image(content):
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        image_format = image.format
        width, height = image.size
    if image_format not in FORMAT_EXTENSIONS:
        raise ValueError(f"Unsupported downloaded format: {image_format}")
    return image_format, width, height


def extract_batch_id(caption):
    match = BATCH_ID_PATTERN.search(caption or "")
    return match.group(1) if match else None


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    calibration_dir = args.calibration_dir.resolve()
    source_path = calibration_dir / "calibration_manifest.csv"
    batch_path = calibration_dir / "batch_manifest.csv"
    download_dir = calibration_dir / "download"
    output_path = calibration_dir / "download_manifest.csv"
    if args.max_posts < 1:
        raise ValueError("--max-posts must be positive.")
    for path in [source_path, batch_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError(f"Download manifest exists: {output_path}")
    if download_dir.exists() and any(download_dir.iterdir()):
        raise FileExistsError(f"Download directory is not empty: {download_dir}")

    source = pd.read_csv(source_path)
    batches = pd.read_csv(batch_path)
    expected_batches = set(batches["batch_id"])
    expected_counts = source.groupby("batch_id").size().to_dict()
    if source["sample_id"].duplicated().any():
        raise ValueError("Calibration sample IDs are not unique.")
    if set(source["batch_id"]) != expected_batches:
        raise ValueError("Sample and batch manifests disagree.")

    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
    )
    session_file = (
        str(args.session_file.resolve()) if args.session_file else None
    )
    loader.load_session_from_file(args.username, filename=session_file)
    if loader.test_login() is None:
        raise RuntimeError("Instaloader session is not logged in.")

    profile = instaloader.Profile.from_username(
        loader.context, args.username
    )
    matched_posts = {}
    for index, post in enumerate(profile.get_posts()):
        if index >= args.max_posts:
            break
        batch_id = extract_batch_id(post.caption)
        if batch_id not in expected_batches:
            continue
        if batch_id in matched_posts:
            raise RuntimeError(f"Duplicate Instagram batch ID: {batch_id}")
        expected_type = (
            "GraphImage" if expected_counts[batch_id] == 1 else "GraphSidecar"
        )
        if post.typename != expected_type:
            raise RuntimeError(
                f"Expected {expected_type} for {batch_id}, "
                f"got {post.typename}"
            )
        matched_posts[batch_id] = post
    missing = expected_batches - set(matched_posts)
    if missing:
        raise RuntimeError(
            "Calibration carousel posts not found: "
            + ", ".join(sorted(missing))
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    downloaded_at = datetime.now(timezone.utc).isoformat()
    for batch_id in batches.sort_values("batch_number")["batch_id"]:
        post = matched_posts[batch_id]
        nodes = (
            [(False, post.url)]
            if post.typename == "GraphImage"
            else [
                (node.is_video, node.display_url)
                for node in post.get_sidecar_nodes()
            ]
        )
        expected = source[source["batch_id"].eq(batch_id)].sort_values(
            "carousel_position"
        )
        if len(nodes) != len(expected):
            raise RuntimeError(
                f"{batch_id}: expected {len(expected)} items, got {len(nodes)}"
            )
        for source_row, (is_video, display_url) in zip(
            expected.itertuples(), nodes
        ):
            if is_video:
                raise RuntimeError(f"Unexpected video in {batch_id}")
            response = loader.context.get_raw(display_url)
            response.raise_for_status()
            content = response.content
            image_format, width, height = inspect_image(content)
            extension = FORMAT_EXTENSIONS[image_format]
            destination = download_dir / f"{source_row.sample_id}{extension}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(destination)
            rows.append(
                {
                    "sample_id": source_row.sample_id,
                    "batch_id": batch_id,
                    "carousel_position": source_row.carousel_position,
                    "download_path": str(
                        destination.relative_to(calibration_dir)
                    ),
                    "shortcode": post.shortcode,
                    "permalink": (
                        f"https://www.instagram.com/p/{post.shortcode}/"
                    ),
                    "posted_at_utc": post.date_utc.replace(
                        tzinfo=timezone.utc
                    ).isoformat(),
                    "downloaded_at_utc": downloaded_at,
                    "target_format": image_format,
                    "target_width": width,
                    "target_height": height,
                    "target_bytes": len(content),
                    "target_sha256": sha256_bytes(content),
                    "content_type": response.headers.get(
                        "Content-Type", ""
                    ),
                    "etag": response.headers.get("ETag", ""),
                }
            )
            print(
                f"Downloaded {source_row.sample_id}: "
                f"{width}x{height} {image_format}"
            )

    if len(rows) != len(source):
        raise RuntimeError("Downloaded row count does not match manifest.")
    write_manifest(output_path, rows)
    print(f"Downloaded {len(rows)} paired calibration images.")
    print(f"Download manifest: {output_path}")


if __name__ == "__main__":
    main()
