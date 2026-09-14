#!/usr/bin/env python3
"""Extract one JSON-defined FF split from a mixed DF40 ZIP archive."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def flatten_frames(value: object) -> list[str]:
    result: list[str] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            if all(isinstance(item, str) for item in current):
                result.extend(current)
            else:
                stack.extend(current)
    return result


def main() -> None:
    args = parse_args()
    document = json.loads(args.json.read_text(encoding="utf-8"))
    dataset = next(iter(document.values()))
    fake_nodes = [
        value for key, value in dataset.items() if key.lower().endswith("_fake")
    ]
    if len(fake_nodes) != 1:
        raise RuntimeError("Expected exactly one *_Fake branch")

    listed_paths = flatten_frames(fake_nodes[0].get(args.split, {}))
    expected = {
        (Path(value).parent.name.removeprefix("temp_"), Path(value).name)
        for value in listed_paths
    }

    with zipfile.ZipFile(args.archive) as archive:
        members: dict[tuple[str, str], zipfile.ZipInfo] = {}
        ff_pngs = 0
        ignored_non_ff = 0
        for info in archive.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.suffix.lower() != ".png":
                continue
            if "ff" not in path.parts:
                ignored_non_ff += 1
                continue
            ff_pngs += 1
            key = (path.parent.name.removeprefix("temp_"), path.name)
            if key in members:
                raise RuntimeError(f"Duplicate FF ZIP member: {key}")
            members[key] = info

        missing = expected - set(members)
        if missing:
            raise FileNotFoundError(
                f"Archive lacks {len(missing)} JSON-listed {args.split} images; "
                f"first={sorted(missing)[:3]}"
            )

        frames_root = args.output_root / args.split / "ff" / "frames"
        for index, (video_id, filename) in enumerate(sorted(expected), start=1):
            destination = frames_root / video_id / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            with archive.open(members[(video_id, filename)]) as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            if index % 500 == 0:
                print(f"Extracted {index}/{len(expected)}", flush=True)

    print(
        json.dumps(
            {
                "split": args.split,
                "json_images": len(expected),
                "archive_ff_images": ff_pngs,
                "extracted": len(expected),
                "ignored_non_ff_images": ignored_non_ff,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
