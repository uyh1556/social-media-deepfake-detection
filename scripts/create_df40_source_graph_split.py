import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


METHODS = {
    "simswap": {
        "display_name": "SimSwap",
        "family": "face_swap",
        "source_kind": "paired_ff_ids",
    },
    "blendface": {
        "display_name": "BlendFace",
        "family": "face_swap",
        "source_kind": "paired_ff_ids",
    },
    "sadtalker": {
        "display_name": "SadTalker",
        "family": "face_reenactment",
        "source_kind": "ff_id_and_driver",
    },
    "wav2lip": {
        "display_name": "Wav2Lip",
        "family": "face_reenactment",
        "source_kind": "ff_id_and_driver",
    },
}
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "sample_id",
    "split",
    "label",
    "method",
    "family",
    "domain",
    "compression",
    "group_id",
    "video_id",
    "source_ids",
    "driver_id",
    "source_path",
    "physical_splits",
    "duplicate_copies",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create one leakage-safe source-graph split shared by the four "
            "retained DF40 FF-domain methods and their real pool."
        )
    )
    parser.add_argument("--df40-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ff_id(value, context):
    if len(value) != 3 or not value.isdigit():
        raise ValueError(f"Invalid FF++ source ID in {context}: {value!r}")


def parse_fake_video(method, physical_video_id):
    video_id = physical_video_id.removeprefix("temp_")
    source_kind = METHODS[method]["source_kind"]
    if source_kind == "paired_ff_ids":
        parts = video_id.split("_")
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected {method} video ID: {physical_video_id}"
            )
        first, second = parts
        validate_ff_id(first, physical_video_id)
        validate_ff_id(second, physical_video_id)
        return video_id, (first, second), ""

    if "##" in video_id:
        source_id, driver_id = video_id.split("##", 1)
    else:
        parts = video_id.split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected {method} video ID: {physical_video_id}"
            )
        source_id, driver_id = parts
    validate_ff_id(source_id, physical_video_id)
    if not driver_id:
        raise ValueError(f"Missing driver ID: {physical_video_id}")
    return video_id, (source_id,), driver_id


def connect(adjacency, nodes):
    for node in nodes:
        adjacency[node]
    for first in nodes:
        for second in nodes:
            if first != second:
                adjacency[first].add(second)


def candidate_preference(candidate):
    return (
        candidate["physical_video_id"].startswith("temp_"),
        0 if candidate["physical_split"] == "train" else 1,
        candidate["source_path"].as_posix(),
    )


def collect_data(df40_root):
    adjacency = defaultdict(set)
    real_records = []
    real_root = df40_root / "real/ff/c23/frames"
    if not real_root.is_dir():
        raise FileNotFoundError(real_root)

    for video_dir in sorted(path for path in real_root.iterdir() if path.is_dir()):
        validate_ff_id(video_dir.name, video_dir)
        node = f"ff:{video_dir.name}"
        adjacency[node]
        images = sorted(video_dir.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG files in {video_dir}")
        for image_path in images:
            real_records.append(
                {
                    "sample_id": f"real:{video_dir.name}:{image_path.name}",
                    "label": "real",
                    "method": "original",
                    "family": "real",
                    "video_id": video_dir.name,
                    "source_ids": (video_dir.name,),
                    "driver_id": "",
                    "source_path": image_path,
                    "physical_splits": "real_pool",
                    "duplicate_copies": 1,
                    "graph_nodes": (node,),
                }
            )

    candidates = defaultdict(list)
    for method in METHODS:
        for physical_split in ("train", "test"):
            frames_root = df40_root / method / physical_split / "ff/frames"
            if not frames_root.is_dir():
                raise FileNotFoundError(frames_root)
            for video_dir in sorted(
                path for path in frames_root.iterdir() if path.is_dir()
            ):
                video_id, source_ids, driver_id = parse_fake_video(
                    method, video_dir.name
                )
                graph_nodes = tuple(f"ff:{value}" for value in source_ids)
                if driver_id:
                    graph_nodes += (f"driver:{driver_id}",)
                connect(adjacency, graph_nodes)
                images = sorted(video_dir.glob("*.png"))
                if not images:
                    raise FileNotFoundError(f"No PNG files in {video_dir}")
                for image_path in images:
                    logical_key = (method, video_id, image_path.name)
                    candidates[logical_key].append(
                        {
                            "physical_split": physical_split,
                            "physical_video_id": video_dir.name,
                            "source_path": image_path,
                            "source_ids": source_ids,
                            "driver_id": driver_id,
                            "graph_nodes": graph_nodes,
                        }
                    )

    fake_records = []
    duplicate_rows = []
    for (method, video_id, frame_name), copies in sorted(candidates.items()):
        copies = sorted(copies, key=candidate_preference)
        canonical = copies[0]
        hashes = None
        if len(copies) > 1:
            hashes = [sha256_file(copy["source_path"]) for copy in copies]
            if len(set(hashes)) != 1:
                raise RuntimeError(
                    "A repeated logical sample is not byte-identical: "
                    f"{method}/{video_id}/{frame_name}"
                )
            duplicate_rows.append(
                {
                    "method": METHODS[method]["display_name"],
                    "video_id": video_id,
                    "frame_name": frame_name,
                    "copies": len(copies),
                    "physical_splits": "|".join(
                        sorted({copy["physical_split"] for copy in copies})
                    ),
                    "sha256": hashes[0],
                    "canonical_path": canonical["source_path"].relative_to(
                        df40_root
                    ).as_posix(),
                    "duplicate_paths": "|".join(
                        copy["source_path"].relative_to(df40_root).as_posix()
                        for copy in copies[1:]
                    ),
                }
            )
        fake_records.append(
            {
                "sample_id": f"{method}:{video_id}:{frame_name}",
                "label": "fake",
                "method": METHODS[method]["display_name"],
                "method_key": method,
                "family": METHODS[method]["family"],
                "video_id": video_id,
                "source_ids": canonical["source_ids"],
                "driver_id": canonical["driver_id"],
                "source_path": canonical["source_path"],
                "physical_splits": "|".join(
                    split
                    for split in ("train", "test")
                    if split in {copy["physical_split"] for copy in copies}
                ),
                "duplicate_copies": len(copies),
                "graph_nodes": canonical["graph_nodes"],
            }
        )
    return adjacency, real_records, fake_records, duplicate_rows


def build_components(adjacency):
    seen = set()
    node_to_group = {}
    groups = {}
    for start in sorted(adjacency):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        nodes = []
        while queue:
            current = queue.popleft()
            nodes.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        nodes = tuple(sorted(nodes))
        digest = hashlib.sha256("|".join(nodes).encode("utf-8")).hexdigest()
        group_id = f"g_{digest[:12]}"
        if group_id in groups:
            raise RuntimeError(f"Group hash collision: {group_id}")
        groups[group_id] = nodes
        for node in nodes:
            node_to_group[node] = group_id
    return groups, node_to_group


def attach_groups(records, node_to_group):
    for record in records:
        group_ids = {node_to_group[node] for node in record["graph_nodes"]}
        if len(group_ids) != 1:
            raise RuntimeError(
                f"Graph construction failed for {record['sample_id']}"
            )
        record["group_id"] = group_ids.pop()


def assign_groups(groups, records, ratios, seed):
    weights = {group_id: Counter() for group_id in groups}
    for record in records:
        stratum = (
            "real" if record["label"] == "real" else record["method_key"]
        )
        weights[record["group_id"]][stratum] += 1

    rng = random.Random(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    tie_order = {group_id: index for index, group_id in enumerate(group_ids)}
    group_ids.sort(
        key=lambda group_id: (
            -sum(weights[group_id].values()),
            tie_order[group_id],
        )
    )
    total_images = sum(sum(weight.values()) for weight in weights.values())
    targets = {
        split: total_images * ratios[split]
        for split in SPLITS
    }
    counts = {split: Counter() for split in SPLITS}
    group_counts = Counter()
    assignments = {}
    for group_id in group_ids:
        split = min(
            SPLITS,
            key=lambda candidate: (
                sum(counts[candidate].values()) / targets[candidate],
                group_counts[candidate] / (len(groups) * ratios[candidate]),
                SPLITS.index(candidate),
            ),
        )
        assignments[group_id] = split
        counts[split].update(weights[group_id])
        group_counts[split] += 1
    return assignments, weights


def manifest_row(record, split, df40_root):
    return {
        "sample_id": record["sample_id"],
        "split": split,
        "label": record["label"],
        "method": record["method"],
        "family": record["family"],
        "domain": "ff",
        "compression": "c23",
        "group_id": record["group_id"],
        "video_id": record["video_id"],
        "source_ids": "|".join(record["source_ids"]),
        "driver_id": record["driver_id"],
        "source_path": record["source_path"].relative_to(
            df40_root
        ).as_posix(),
        "physical_splits": record["physical_splits"],
        "duplicate_copies": record["duplicate_copies"],
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(rows):
    result = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        method_counts = Counter(row["method"] for row in selected)
        result[split] = {
            "images": len(selected),
            "real": sum(row["label"] == "real" for row in selected),
            "fake": sum(row["label"] == "fake" for row in selected),
            "methods": dict(sorted(method_counts.items())),
            "videos": len({(row["method"], row["video_id"]) for row in selected}),
            "groups": len({row["group_id"] for row in selected}),
        }
    return result


def validate(rows, groups, assignments, df40_root):
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sample IDs remain in the manifest.")
    if len({row["source_path"] for row in rows}) != len(rows):
        raise RuntimeError("One physical file is used by multiple samples.")
    missing = [
        row["source_path"]
        for row in rows
        if not (df40_root / row["source_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing manifest paths: {missing[:10]}")

    group_splits = defaultdict(set)
    source_splits = defaultdict(set)
    driver_splits = defaultdict(set)
    video_splits = defaultdict(set)
    for row in rows:
        split = row["split"]
        group_splits[row["group_id"]].add(split)
        for source_id in row["source_ids"].split("|"):
            source_splits[source_id].add(split)
        if row["driver_id"]:
            driver_splits[row["driver_id"]].add(split)
        video_splits[(row["method"], row["video_id"])].add(split)
    checks = {
        "group_overlap": sum(len(value) > 1 for value in group_splits.values()),
        "source_id_overlap": sum(
            len(value) > 1 for value in source_splits.values()
        ),
        "driver_id_overlap": sum(
            len(value) > 1 for value in driver_splits.values()
        ),
        "video_id_overlap": sum(
            len(value) > 1 for value in video_splits.values()
        ),
    }
    if any(checks.values()):
        raise RuntimeError(f"Leakage validation failed: {checks}")
    if set(groups) != set(assignments):
        raise RuntimeError("Not every graph component received a split.")
    return checks


def main():
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0.")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("Every split ratio must be positive.")

    df40_root = args.df40_root.resolve()
    output_dir = args.output_dir.resolve()
    if not df40_root.is_dir():
        raise FileNotFoundError(df40_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    adjacency, real_records, fake_records, duplicate_rows = collect_data(
        df40_root
    )
    groups, node_to_group = build_components(adjacency)
    attach_groups(real_records, node_to_group)
    attach_groups(fake_records, node_to_group)
    all_records = real_records + fake_records
    assignments, weights = assign_groups(
        groups, all_records, ratios, args.seed
    )
    rows = [
        manifest_row(
            record,
            assignments[record["group_id"]],
            df40_root,
        )
        for record in all_records
    ]
    rows.sort(
        key=lambda row: (
            SPLITS.index(row["split"]),
            row["label"],
            row["method"],
            row["video_id"],
            row["source_path"],
        )
    )
    checks = validate(rows, groups, assignments, df40_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", rows, MANIFEST_FIELDS)
    real_rows = [row for row in rows if row["label"] == "real"]
    fake_rows = [row for row in rows if row["label"] == "fake"]
    write_csv(output_dir / "real.csv", real_rows, MANIFEST_FIELDS)
    write_csv(output_dir / "fake.csv", fake_rows, MANIFEST_FIELDS)
    for method, metadata in METHODS.items():
        display_name = metadata["display_name"]
        selected_fake = [row for row in fake_rows if row["method"] == display_name]
        write_csv(
            output_dir / "methods" / f"{method}.csv",
            selected_fake,
            MANIFEST_FIELDS,
        )
        binary_rows = sorted(
            real_rows + selected_fake,
            key=lambda row: (
                SPLITS.index(row["split"]),
                row["label"],
                row["video_id"],
                row["source_path"],
            ),
        )
        write_csv(
            output_dir / "binary" / f"{method}.csv",
            binary_rows,
            MANIFEST_FIELDS,
        )

    group_rows = []
    for group_id in sorted(groups):
        nodes = groups[group_id]
        weight = weights[group_id]
        group_rows.append(
            {
                "group_id": group_id,
                "split": assignments[group_id],
                "source_ids": "|".join(
                    node.removeprefix("ff:")
                    for node in nodes
                    if node.startswith("ff:")
                ),
                "driver_ids": "|".join(
                    node.removeprefix("driver:")
                    for node in nodes
                    if node.startswith("driver:")
                ),
                "real_images": weight["real"],
                "simswap_images": weight["simswap"],
                "blendface_images": weight["blendface"],
                "sadtalker_images": weight["sadtalker"],
                "wav2lip_images": weight["wav2lip"],
                "total_images": sum(weight.values()),
            }
        )
    write_csv(
        output_dir / "group_split.csv",
        group_rows,
        tuple(group_rows[0]),
    )
    duplicate_fields = (
        tuple(duplicate_rows[0])
        if duplicate_rows
        else (
            "method",
            "video_id",
            "frame_name",
            "copies",
            "physical_splits",
            "sha256",
            "canonical_path",
            "duplicate_paths",
        )
    )
    write_csv(
        output_dir / "duplicate_samples.csv",
        duplicate_rows,
        duplicate_fields,
    )

    method_duplicate_counts = Counter(
        row["method"] for row in duplicate_rows
    )
    summary = {
        "protocol": "df40_four_method_source_graph_v1",
        "created_from": str(df40_root),
        "storage": "manifest_only_source_files_unchanged",
        "seed": args.seed,
        "ratios": ratios,
        "graph_policy": {
            "face_swap": "connect both FF++ source IDs",
            "face_reenactment": (
                "connect the FF++ source ID and normalized driving-clip ID"
            ),
            "cross_method": (
                "one shared graph across all four methods and the real pool"
            ),
        },
        "deduplication_policy": (
            "Within-method repeated logical samples are retained once only "
            "after byte-identical SHA-256 verification; source files remain "
            "unchanged."
        ),
        "unique_images": len(rows),
        "real_images": len(real_rows),
        "fake_images": len(fake_rows),
        "graph_components": len(groups),
        "ff_source_ids": len(
            {
                node
                for nodes in groups.values()
                for node in nodes
                if node.startswith("ff:")
            }
        ),
        "driver_ids": len(
            {
                node
                for nodes in groups.values()
                for node in nodes
                if node.startswith("driver:")
            }
        ),
        "deduplicated_logical_samples": len(duplicate_rows),
        "removed_duplicate_copies": sum(
            row["copies"] - 1 for row in duplicate_rows
        ),
        "duplicates_by_method": dict(sorted(method_duplicate_counts.items())),
        "split_summary": count_rows(rows),
        "leakage_checks": checks,
        "notes": [
            "The generated files do not copy, move, rename, or delete images.",
            "The manifest preserves original DF40 physical split provenance.",
            "Training mixtures must still freeze equal per-method counts and a matched real pool.",
        ],
    }
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["output_sha256"] = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*.csv"))
    }
    config_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
