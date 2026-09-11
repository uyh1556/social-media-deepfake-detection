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


SAMPLE_ID_PATTERN = re.compile(r"\bsample_id=([A-Za-z0-9_-]+)\b")
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download only the caption-matched Instagram web pilot posts."
        )
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Dedicated research-account username used by Instaloader.",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("instagram_pipeline/pilot_v1"),
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        default=None,
        help="Optional Instaloader session file. Default: its normal location.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=20,
        help="Maximum number of recent posts inspected; only pilot IDs download.",
    )
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


def extract_sample_id(caption):
    match = SAMPLE_ID_PATTERN.search(caption or "")
    return match.group(1) if match else None


def write_manifest(path, rows):
    fieldnames = [
        "sample_id",
        "download_path",
        "shortcode",
        "permalink",
        "posted_at_utc",
        "downloaded_at_utc",
        "target_format",
        "target_width",
        "target_height",
        "target_bytes",
        "target_sha256",
        "content_type",
        "etag",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    pilot_dir = args.pilot_dir.resolve()
    source_manifest_path = pilot_dir / "pilot_manifest.csv"
    download_dir = pilot_dir / "download"
    download_manifest_path = pilot_dir / "download_manifest.csv"

    if args.max_posts < 1:
        raise ValueError("--max-posts must be positive.")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    if download_manifest_path.exists():
        raise FileExistsError(
            f"Download manifest already exists: {download_manifest_path}"
        )
    if download_dir.exists() and any(download_dir.iterdir()):
        raise FileExistsError(
            f"Download directory is not empty: {download_dir}"
        )

    source_manifest = pd.read_csv(source_manifest_path)
    if "sample_id" not in source_manifest.columns:
        raise ValueError("pilot_manifest.csv has no sample_id column.")
    expected_ids = set(source_manifest["sample_id"])
    if len(expected_ids) != len(source_manifest):
        raise ValueError("Pilot sample IDs are not unique.")

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
        sample_id = extract_sample_id(post.caption)
        if sample_id not in expected_ids:
            continue
        if sample_id in matched_posts:
            raise RuntimeError(f"Duplicate Instagram sample ID: {sample_id}")
        if post.typename != "GraphImage":
            raise RuntimeError(
                f"Pilot post is not a single image: {sample_id} "
                f"({post.typename})"
            )
        matched_posts[sample_id] = post

    missing_ids = expected_ids - set(matched_posts)
    if missing_ids:
        raise RuntimeError(
            "Pilot posts not found within --max-posts: "
            + ", ".join(sorted(missing_ids))
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    downloaded_at = datetime.now(timezone.utc).isoformat()
    ordered_ids = source_manifest.sort_values("upload_order")["sample_id"]
    for sample_id in ordered_ids:
        post = matched_posts[sample_id]
        response = loader.context.get_raw(post.url)
        response.raise_for_status()
        content = response.content
        image_format, width, height = inspect_image(content)
        extension = FORMAT_EXTENSIONS[image_format]
        destination = download_dir / f"{sample_id}{extension}"
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(destination)

        rows.append(
            {
                "sample_id": sample_id,
                "download_path": str(destination.relative_to(pilot_dir)),
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
                "content_type": response.headers.get("Content-Type", ""),
                "etag": response.headers.get("ETag", ""),
            }
        )
        print(
            f"Downloaded {sample_id}: {width}x{height} "
            f"{image_format}, {len(content)} bytes"
        )

    write_manifest(download_manifest_path, rows)
    print(f"Downloaded {len(rows)} matched pilot images.")
    print(f"Download manifest: {download_manifest_path}")


if __name__ == "__main__":
    main()
