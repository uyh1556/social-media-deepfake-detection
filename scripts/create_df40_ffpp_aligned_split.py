import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from create_df40_source_graph_split import (
    MANIFEST_FIELDS,
    METHODS,
    SPLITS,
    attach_groups,
    build_components,
    collect_data,
    count_rows,
    manifest_row,
    sha256_file,
    validate,
    write_csv,
)


PROTECTION_PRIORITY = {"train": 0, "val": 1, "test": 2}
ALIGNED_FIELDS = MANIFEST_FIELDS + ("ffpp_group_ids",)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create DF40 manifests aligned to an existing FF++ source-aware "
            "split without changing the FF++ assignments."
        )
    )
    parser.add_argument("--df40-root", type=Path, required=True)
    parser.add_argument("--ffpp-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_ffpp_membership(path):
    frame = pd.read_csv(
        path,
        dtype={"split": str, "group_id": str, "source_ids": str},
        keep_default_na=False,
    )
    required = {"split", "group_id", "source_ids"}
    if missing := required - set(frame.columns):
        raise ValueError(f"FF++ manifest missing columns: {sorted(missing)}")
    source_to_split = {}
    source_to_group = {}
    for row in frame[list(required)].drop_duplicates().itertuples(index=False):
        for source_id in row.source_ids.replace("|", ",").split(","):
            source_id = source_id.strip()
            if not source_id:
                continue
            previous_split = source_to_split.setdefault(source_id, row.split)
            previous_group = source_to_group.setdefault(source_id, row.group_id)
            if previous_split != row.split or previous_group != row.group_id:
                raise RuntimeError(
                    f"Inconsistent FF++ membership for source {source_id}"
                )
    if set(source_to_split.values()) - set(SPLITS):
        raise ValueError("FF++ manifest contains unexpected split names.")
    return source_to_split, source_to_group


def provisional_split(record, source_to_split):
    splits = {source_to_split.get(source_id) for source_id in record["source_ids"]}
    if None in splits:
        return None, "source_not_in_ffpp_manifest"
    if len(splits) != 1:
        raise RuntimeError(
            "A DF40 sample connects FF++ sources assigned to different splits: "
            f"{record['sample_id']} -> {sorted(splits)}"
        )
    return splits.pop(), ""


def protect_drivers(records):
    driver_splits = defaultdict(set)
    for record in records:
        if record["driver_id"]:
            driver_splits[record["driver_id"]].add(record["aligned_split"])
    driver_assignment = {
        driver_id: max(splits, key=lambda split: PROTECTION_PRIORITY[split])
        for driver_id, splits in driver_splits.items()
    }
    conflicts = {
        driver_id: splits
        for driver_id, splits in driver_splits.items()
        if len(splits) > 1
    }
    retained = []
    excluded = []
    for record in records:
        driver_id = record["driver_id"]
        if driver_id and driver_assignment[driver_id] != record["aligned_split"]:
            excluded.append((record, "driver_used_in_protected_split"))
        else:
            retained.append(record)
    return retained, excluded, driver_assignment, conflicts


def rebuild_retained_graph(records):
    adjacency = defaultdict(set)
    for record in records:
        nodes = record["graph_nodes"]
        for node in nodes:
            adjacency[node]
        for first in nodes:
            for second in nodes:
                if first != second:
                    adjacency[first].add(second)
    return adjacency


def assign_components(groups, source_to_split):
    assignments = {}
    for group_id, nodes in groups.items():
        splits = {
            source_to_split[node.removeprefix("ff:")]
            for node in nodes
            if node.startswith("ff:")
        }
        if len(splits) != 1:
            raise RuntimeError(
                f"Aligned component spans multiple FF++ splits: {group_id}"
            )
        assignments[group_id] = splits.pop()
    return assignments


def exclusion_row(record, reason, df40_root, source_to_split):
    known_splits = sorted(
        {
            source_to_split[source_id]
            for source_id in record["source_ids"]
            if source_id in source_to_split
        }
    )
    return {
        "sample_id": record["sample_id"],
        "label": record["label"],
        "method": record["method"],
        "video_id": record["video_id"],
        "source_ids": "|".join(record["source_ids"]),
        "driver_id": record["driver_id"],
        "known_ffpp_splits": "|".join(known_splits),
        "reason": reason,
        "source_path": record["source_path"].relative_to(
            df40_root
        ).as_posix(),
    }


def main():
    args = parse_args()
    df40_root = args.df40_root.resolve()
    ffpp_manifest = args.ffpp_manifest.resolve()
    output_dir = args.output_dir.resolve()
    for path in (df40_root, ffpp_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    source_to_split, source_to_ffpp_group = load_ffpp_membership(ffpp_manifest)
    _, real_records, fake_records, duplicate_rows = collect_data(df40_root)
    all_records = real_records + fake_records

    provisional = []
    exclusions = []
    for record in all_records:
        split, reason = provisional_split(record, source_to_split)
        if reason:
            exclusions.append((record, reason))
            continue
        record["aligned_split"] = split
        provisional.append(record)

    retained, driver_exclusions, driver_assignment, driver_conflicts = (
        protect_drivers(provisional)
    )
    exclusions.extend(driver_exclusions)
    adjacency = rebuild_retained_graph(retained)
    groups, node_to_group = build_components(adjacency)
    attach_groups(retained, node_to_group)
    assignments = assign_components(groups, source_to_split)

    rows = []
    for record in retained:
        if assignments[record["group_id"]] != record["aligned_split"]:
            raise RuntimeError(f"Split alignment failed: {record['sample_id']}")
        row = manifest_row(record, record["aligned_split"], df40_root)
        row["ffpp_group_ids"] = "|".join(
            sorted(
                {
                    source_to_ffpp_group[source_id]
                    for source_id in record["source_ids"]
                }
            )
        )
        rows.append(row)
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
    write_csv(output_dir / "manifest.csv", rows, ALIGNED_FIELDS)
    real_rows = [row for row in rows if row["label"] == "real"]
    fake_rows = [row for row in rows if row["label"] == "fake"]
    write_csv(output_dir / "real.csv", real_rows, ALIGNED_FIELDS)
    write_csv(output_dir / "fake.csv", fake_rows, ALIGNED_FIELDS)
    for method, metadata in METHODS.items():
        selected_fake = [
            row for row in fake_rows
            if row["method"] == metadata["display_name"]
        ]
        write_csv(
            output_dir / "methods" / f"{method}.csv",
            selected_fake,
            ALIGNED_FIELDS,
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
            ALIGNED_FIELDS,
        )

    group_rows = []
    for group_id in sorted(groups):
        nodes = groups[group_id]
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
                "ffpp_group_ids": "|".join(
                    sorted(
                        {
                            source_to_ffpp_group[node.removeprefix("ff:")]
                            for node in nodes
                            if node.startswith("ff:")
                        }
                    )
                ),
                "images": sum(row["group_id"] == group_id for row in rows),
            }
        )
    write_csv(output_dir / "group_split.csv", group_rows, tuple(group_rows[0]))

    exclusion_rows = [
        exclusion_row(record, reason, df40_root, source_to_split)
        for record, reason in exclusions
    ]
    exclusion_fields = (
        "sample_id",
        "label",
        "method",
        "video_id",
        "source_ids",
        "driver_id",
        "known_ffpp_splits",
        "reason",
        "source_path",
    )
    write_csv(
        output_dir / "excluded_samples.csv",
        exclusion_rows,
        exclusion_fields,
    )
    duplicate_fields = tuple(duplicate_rows[0])
    write_csv(
        output_dir / "duplicate_samples.csv",
        duplicate_rows,
        duplicate_fields,
    )

    exclusion_counts = Counter(
        (record["method"], reason) for record, reason in exclusions
    )
    summary = {
        "protocol": "df40_ffpp_source_aligned_v1",
        "df40_root": str(df40_root),
        "ffpp_anchor_manifest": str(ffpp_manifest),
        "ffpp_anchor_manifest_sha256": sha256_file(ffpp_manifest),
        "storage": "manifest_only_source_files_unchanged",
        "split_policy": (
            "Every mapped DF40 FF++ source ID retains its existing FF++ "
            "train/val/test assignment."
        ),
        "driver_conflict_policy": (
            "A driver reused across FF++ splits is retained only in the most "
            "protected available split using test > val > train priority."
        ),
        "mapped_ffpp_source_ids": len(source_to_split),
        "unmapped_df40_source_ids": sorted(
            {
                source_id
                for record, reason in exclusions
                if reason == "source_not_in_ffpp_manifest"
                for source_id in record["source_ids"]
                if source_id not in source_to_split
            }
        ),
        "driver_ids": len(driver_assignment),
        "cross_split_driver_conflicts": len(driver_conflicts),
        "driver_conflict_assignments": dict(
            sorted(
                Counter(
                    driver_assignment[driver_id]
                    for driver_id in driver_conflicts
                ).items()
            )
        ),
        "retained_unique_images": len(rows),
        "excluded_unique_images": len(exclusion_rows),
        "exclusions_by_method_and_reason": {
            f"{method}:{reason}": count
            for (method, reason), count in sorted(exclusion_counts.items())
        },
        "physical_duplicate_logical_samples": len(duplicate_rows),
        "split_summary": count_rows(rows),
        "graph_components_after_alignment": len(groups),
        "leakage_checks": checks,
        "notes": [
            "The existing FF++ split and every source image remain unchanged.",
            "Only the FF++-aligned membership is intended for extending the existing M0 checkpoint.",
            "For M1, use only aligned DF40 train candidates and keep M0 validation/evaluation conditions fixed.",
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
