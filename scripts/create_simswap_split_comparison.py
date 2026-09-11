import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create current and graph-disjoint binary SimSwap test manifests."
        )
    )
    parser.add_argument("--df40-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_membership(path):
    frame = pd.read_csv(
        path, dtype={"video_id": str, "group_id": str, "source_ids": str}
    )
    original = frame[
        (frame["label"] == "real") & (frame["method"] == "original")
    ][["video_id", "split", "group_id"]].drop_duplicates()
    if original.groupby("video_id")["split"].nunique().gt(1).any():
        raise RuntimeError("A source ID occurs in multiple source splits.")
    return {
        row.video_id: {"split": row.split, "group_id": row.group_id}
        for row in original.itertuples(index=False)
    }


def test_components(video_ids):
    adjacency = defaultdict(set)
    for video_id in video_ids:
        parts = video_id.split("_")
        if len(parts) != 2 or any(
            len(part) != 3 or not part.isdigit() for part in parts
        ):
            raise ValueError(f"Unexpected SimSwap video ID: {video_id}")
        first, second = parts
        adjacency[first].add(second)
        adjacency[second].add(first)
    mapping = {}
    seen = set()
    for source_id in sorted(adjacency):
        if source_id in seen:
            continue
        queue = deque([source_id])
        seen.add(source_id)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        group_id = "__".join(sorted(component))
        for member in component:
            mapping[member] = group_id
    return mapping


def build_manifest(df40_root, membership, strict):
    fake_root = df40_root / "simswap/test/ff/frames"
    real_root = df40_root / "real/ff/c23/frames"
    if not fake_root.is_dir():
        raise FileNotFoundError(fake_root)
    if not real_root.is_dir():
        raise FileNotFoundError(real_root)

    video_dirs = sorted(path for path in fake_root.iterdir() if path.is_dir())
    components = test_components([path.name for path in video_dirs])
    retained_fake_dirs = []
    retained_source_ids = set()
    video_membership = []
    for video_dir in video_dirs:
        first, second = video_dir.name.split("_")
        source_ids = (first, second)
        splits = tuple(
            membership.get(source_id, {"split": "missing"})["split"]
            for source_id in source_ids
        )
        eligible = all(split == "test" for split in splits)
        video_membership.append(
            {
                "video_id": video_dir.name,
                "source_ids": "|".join(source_ids),
                "source_split_membership": "|".join(splits),
                "graph_source_disjoint": eligible,
                "frames": len(list(video_dir.glob("*.png"))),
            }
        )
        if strict and not eligible:
            continue
        retained_fake_dirs.append(video_dir)
        retained_source_ids.update(source_ids)

    rows = []
    for source_id in sorted(retained_source_ids):
        video_dir = real_root / source_id
        if not video_dir.is_dir():
            raise FileNotFoundError(video_dir)
        split_name = membership.get(source_id, {"split": "missing"})["split"]
        for image_path in sorted(video_dir.glob("*.png")):
            rows.append(
                {
                    "split": "test",
                    "label": "real",
                    "method": "original",
                    "group_id": components[source_id],
                    "video_id": source_id,
                    "source_ids": source_id,
                    "source_split_membership": split_name,
                    "source_path": image_path.relative_to(df40_root).as_posix(),
                }
            )
    for video_dir in retained_fake_dirs:
        first, second = video_dir.name.split("_")
        splits = [
            membership.get(source_id, {"split": "missing"})["split"]
            for source_id in (first, second)
        ]
        for image_path in sorted(video_dir.glob("*.png")):
            rows.append(
                {
                    "split": "test",
                    "label": "fake",
                    "method": "SimSwap",
                    "group_id": components[first],
                    "video_id": video_dir.name,
                    "source_ids": f"{first}|{second}",
                    "source_split_membership": "|".join(splits),
                    "source_path": image_path.relative_to(df40_root).as_posix(),
                }
            )
    frame = pd.DataFrame(rows)
    audit = pd.DataFrame(video_membership)
    return frame, audit, sorted(retained_source_ids)


def describe(frame):
    return {
        "images": len(frame),
        "real_images": int((frame["label"] == "real").sum()),
        "fake_images": int((frame["label"] == "fake").sum()),
        "real_videos": int(
            frame[frame["label"] == "real"]["video_id"].nunique()
        ),
        "fake_videos": int(
            frame[frame["label"] == "fake"]["video_id"].nunique()
        ),
    }


def main():
    args = parse_args()
    df40_root = args.df40_root.resolve()
    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    membership = source_membership(source_manifest)
    current, audit, current_real_ids = build_manifest(
        df40_root, membership, strict=False
    )
    graph, _, graph_real_ids = build_manifest(
        df40_root, membership, strict=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    current.to_csv(output_dir / "current_physical_test.csv", index=False)
    graph.to_csv(output_dir / "graph_source_disjoint_test.csv", index=False)
    audit.to_csv(output_dir / "video_source_membership.csv", index=False)
    (output_dir / "required_real_video_ids.txt").write_text(
        "\n".join(current_real_ids) + "\n", encoding="utf-8"
    )
    summary = {
        "protocol": "simswap_current_vs_graph_source_disjoint_v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "current_physical_test": describe(current),
        "graph_source_disjoint_test": describe(graph),
        "graph_source_ids": graph_real_ids,
        "comparison_scope": (
            "Effect of overlap with the existing FF++ checkpoint's source "
            "train/validation membership; this does not measure SimSwap "
            "train-test duplicate leakage because the checkpoint was not "
            "trained on SimSwap."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
