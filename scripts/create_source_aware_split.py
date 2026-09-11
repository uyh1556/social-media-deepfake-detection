import argparse
import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path


DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a source-aware train/val/test split for FF++ frame images."
    )
    parser.add_argument("--input-root", default="source_images_raw")
    parser.add_argument("--output-root", default="dataset_split/source_aware")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks. Symlinks are the default.",
    )
    return parser.parse_args()


def video_id_from_frame(path):
    return path.stem.split("_frame", 1)[0]


def fake_source_ids(video_id):
    parts = video_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected fake video id: {video_id}")
    return parts[0], parts[1]


def collect_files(input_root):
    input_root = Path(input_root)
    files = {
        "original": sorted((input_root / "original").glob("*.png")),
        "Deepfakes": sorted((input_root / "Deepfakes").glob("*.png")),
        "Face2Face": sorted((input_root / "Face2Face").glob("*.png")),
    }
    missing = [name for name, paths in files.items() if not paths]
    if missing:
        raise FileNotFoundError(f"No PNG files found for: {', '.join(missing)}")
    return files


def build_components(files):
    nodes = set()
    adj = defaultdict(set)

    for path in files["original"]:
        source_id = video_id_from_frame(path)
        nodes.add(source_id)
        adj[source_id]

    for method in ("Deepfakes", "Face2Face"):
        for path in files[method]:
            src_a, src_b = fake_source_ids(video_id_from_frame(path))
            nodes.update((src_a, src_b))
            adj[src_a].add(src_b)
            adj[src_b].add(src_a)

    seen = set()
    components = []
    for node in sorted(nodes):
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adj[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))

    return components


def make_group_maps(files, components):
    source_to_group = {}
    for component in components:
        group_id = "__".join(component)
        for source_id in component:
            source_to_group[source_id] = group_id

    group_items = defaultdict(list)

    for path in files["original"]:
        source_id = video_id_from_frame(path)
        group_id = source_to_group[source_id]
        group_items[group_id].append(
            {
                "path": path,
                "label": "real",
                "method": "original",
                "video_id": source_id,
                "source_ids": source_id,
            }
        )

    for method in ("Deepfakes", "Face2Face"):
        for path in files[method]:
            video_id = video_id_from_frame(path)
            src_a, src_b = fake_source_ids(video_id)
            group_id = source_to_group[src_a]
            if source_to_group[src_b] != group_id:
                raise RuntimeError(f"Source leakage graph error for {video_id}")
            group_items[group_id].append(
                {
                    "path": path,
                    "label": "fake",
                    "method": method,
                    "video_id": video_id,
                    "source_ids": f"{src_a},{src_b}",
                }
            )

    return group_items


def assign_splits(group_items, seed, ratios):
    split_names = ("train", "val", "test")
    groups = list(group_items.items())
    random.Random(seed).shuffle(groups)

    total = sum(len(items) for _, items in groups)
    targets = {
        "train": total * ratios[0],
        "val": total * ratios[1],
        "test": total * ratios[2],
    }
    counts = {name: 0 for name in split_names}
    assignments = {}

    # Greedy assignment by current fill ratio keeps image counts close to targets.
    for group_id, items in sorted(groups, key=lambda x: len(x[1]), reverse=True):
        split = min(split_names, key=lambda name: counts[name] / targets[name])
        assignments[group_id] = split
        counts[split] += len(items)

    return assignments


def safe_link_or_copy(src, dst, copy=False):
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_split(files, group_items, assignments, output_root, copy=False):
    output_root = Path(output_root)
    manifest_rows = []

    for group_id, items in sorted(group_items.items()):
        split = assignments[group_id]
        for item in items:
            src = item["path"]
            frame_name = src.name
            if item["label"] == "real":
                rel_dst = Path(split) / "real" / frame_name
            else:
                rel_dst = Path(split) / "fake" / item["method"] / frame_name

            dst = output_root / rel_dst
            safe_link_or_copy(src, dst, copy=copy)
            manifest_rows.append(
                {
                    "split": split,
                    "label": item["label"],
                    "method": item["method"],
                    "group_id": group_id,
                    "video_id": item["video_id"],
                    "source_ids": item["source_ids"],
                    "source_path": str(src),
                    "split_path": str(rel_dst),
                }
            )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_rows


def summarize(manifest_rows, group_items, assignments, output_root, args):
    summary = {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "storage": "copy" if args.copy else "symlink",
        "total_groups": len(group_items),
        "total_images": len(manifest_rows),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        rows = [row for row in manifest_rows if row["split"] == split]
        groups = sorted({row["group_id"] for row in rows})
        summary["splits"][split] = {
            "groups": len(groups),
            "images": len(rows),
            "real": sum(row["label"] == "real" for row in rows),
            "fake": sum(row["label"] == "fake" for row in rows),
            "Deepfakes": sum(row["method"] == "Deepfakes" for row in rows),
            "Face2Face": sum(row["method"] == "Face2Face" for row in rows),
        }

    output_root = Path(output_root)
    with (output_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    group_split_path = output_root / "group_split.csv"
    with group_split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "split", "num_images"])
        writer.writeheader()
        for group_id in sorted(group_items):
            writer.writerow(
                {
                    "group_id": group_id,
                    "split": assignments[group_id],
                    "num_images": len(group_items[group_id]),
                }
            )

    return summary


def main():
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    files = collect_files(args.input_root)
    components = build_components(files)
    group_items = make_group_maps(files, components)
    assignments = assign_splits(group_items, args.seed, ratios)
    manifest_rows = write_split(files, group_items, assignments, args.output_root, copy=args.copy)
    summary = summarize(manifest_rows, group_items, assignments, args.output_root, args)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
