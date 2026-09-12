#!/usr/bin/env python3
"""Build one leakage-safe FF++/DF40 master manifest and module manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")
MODULE_METHODS = {
    "real": ("original",),
    "ffpp_fake": ("Deepfakes", "Face2Face"),
    "blendface": ("BlendFace",),
    "sadtalker": ("SadTalker",),
    "simswap": ("SimSwap",),
    "wav2lip": ("Wav2Lip",),
}
ARCHIVE_NAMES = {
    "real": "df40_real_ff_c23_global_v1.tar",
    "ffpp_fake": "ffpp_fake_c23_df40_preprocessed_global_v1.tar",
    "blendface": "df40_blendface_ff_global_v1.tar",
    "sadtalker": "df40_sadtalker_ff_global_v1.tar",
    "simswap": "df40_simswap_ff_global_v1.tar",
    "wav2lip": "df40_wav2lip_ff_global_v1.tar",
}
FIELDS = (
    "sample_id",
    "split",
    "label",
    "method",
    "family",
    "module",
    "domain",
    "compression",
    "preprocessing",
    "group_id",
    "video_id",
    "source_ids",
    "driver_id",
    "frame_index",
    "source_path",
    "local_source_path",
    "ffpp_group_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffpp-manifest", type=Path, required=True)
    parser.add_argument("--df40-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=FIELDS, quoting=csv.QUOTE_MINIMAL)


def normalize_source_ids(value: str) -> str:
    return "|".join(part.strip() for part in value.replace(",", "|").split("|") if part.strip())


def module_for_method(method: str) -> str:
    for module, methods in MODULE_METHODS.items():
        if method in methods:
            return module
    raise ValueError(f"No module configured for method: {method}")


def archive_source_path(method: str, video_id: str, filename: str) -> str:
    return (Path("images") / method / video_id / filename).as_posix()


def prepare_ffpp(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    required = {
        "sample_id", "split", "label", "method", "family", "domain",
        "compression", "preprocessing", "group_id", "video_id",
        "source_ids", "frame_index", "local_source_path",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"FF++ manifest missing columns: {sorted(missing)}")
    selected = frame[frame["method"].isin(("original", "Deepfakes", "Face2Face"))].copy()
    selected["module"] = selected["method"].map(module_for_method)
    selected["source_ids"] = selected["source_ids"].map(normalize_source_ids)
    selected["driver_id"] = ""
    selected["ffpp_group_ids"] = selected["group_id"]
    selected["source_path"] = selected.apply(
        lambda row: archive_source_path(
            row["method"], row["video_id"], Path(row["local_source_path"]).name
        ),
        axis=1,
    )
    for relative in selected["local_source_path"]:
        if not (project_root / relative).is_file():
            raise FileNotFoundError(project_root / relative)
    return selected


def prepare_df40(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    required = {
        "sample_id", "split", "label", "method", "family", "domain",
        "compression", "group_id", "video_id", "source_ids", "driver_id",
        "source_path", "ffpp_group_ids",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"DF40 manifest missing columns: {sorted(missing)}")
    selected = frame[frame["method"].isin(("BlendFace", "SadTalker", "SimSwap", "Wav2Lip"))].copy()
    selected["module"] = selected["method"].map(module_for_method)
    selected["source_ids"] = selected["source_ids"].map(normalize_source_ids)
    selected["preprocessing"] = "df40_provided_aligned_face_256"
    selected["frame_index"] = selected["source_path"].map(lambda path: int(Path(path).stem))
    selected["local_source_path"] = selected["source_path"].map(
        lambda path: (Path("data/df40") / path).as_posix()
    )
    selected["source_path"] = selected.apply(
        lambda row: archive_source_path(
            row["method"], row["video_id"], Path(row["local_source_path"]).name
        ),
        axis=1,
    )
    for relative in selected["local_source_path"]:
        if not (project_root / relative).is_file():
            raise FileNotFoundError(project_root / relative)
    return selected


def rebuild_global_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        nodes = [f"source:{value}" for value in row.source_ids.split("|") if value]
        if row.driver_id:
            nodes.append(f"driver:{row.driver_id}")
        for first in nodes:
            adjacency[first]
            for second in nodes:
                if first != second:
                    adjacency[first].add(second)

    node_to_group: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for start in sorted(adjacency):
        if start in node_to_group:
            continue
        queue = deque([start])
        component: list[str] = []
        seen = {start}
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        nodes = sorted(component)
        group_id = "global_" + hashlib.sha256("|".join(nodes).encode()).hexdigest()[:12]
        groups[group_id] = nodes
        for node in nodes:
            node_to_group[node] = group_id

    def row_group(row: pd.Series) -> str:
        groups_for_row = {
            node_to_group[f"source:{value}"]
            for value in row["source_ids"].split("|") if value
        }
        if row["driver_id"]:
            groups_for_row.add(node_to_group[f"driver:{row['driver_id']}"])
        if len(groups_for_row) != 1:
            raise RuntimeError(f"Row spans global groups: {row['sample_id']}")
        return groups_for_row.pop()

    result = frame.copy()
    result["group_id"] = result.apply(row_group, axis=1)
    return result, groups


def validate(frame: pd.DataFrame, project_root: Path) -> dict[str, int]:
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise RuntimeError(f"Duplicate sample IDs: {duplicates}")
    if frame["source_path"].duplicated().any():
        duplicates = frame.loc[frame["source_path"].duplicated(), "source_path"].head().tolist()
        raise RuntimeError(f"Duplicate portable paths: {duplicates}")
    unexpected_splits = set(frame["split"]) - set(SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected splits: {sorted(unexpected_splits)}")

    source_splits: dict[str, set[str]] = defaultdict(set)
    driver_splits: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        for source_id in row.source_ids.split("|"):
            source_splits[source_id].add(row.split)
        if row.driver_id:
            driver_splits[row.driver_id].add(row.split)
    checks = {
        "sample_id_duplicates": int(frame["sample_id"].duplicated().sum()),
        "portable_path_duplicates": int(frame["source_path"].duplicated().sum()),
        "source_id_overlap": sum(len(value) > 1 for value in source_splits.values()),
        "driver_id_overlap": sum(len(value) > 1 for value in driver_splits.values()),
        "group_overlap": int(frame.groupby("group_id")["split"].nunique().gt(1).sum()),
        "method_video_overlap": int(
            frame.groupby(["method", "video_id"])["split"].nunique().gt(1).sum()
        ),
        "missing_local_files": sum(
            not (project_root / relative).is_file()
            for relative in frame["local_source_path"]
        ),
    }
    if any(checks.values()):
        raise RuntimeError(f"Global manifest validation failed: {checks}")
    return checks


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    ffpp_manifest = args.ffpp_manifest.resolve()
    df40_manifest = args.df40_manifest.resolve()
    output_dir = args.output_dir.resolve()
    for path in (ffpp_manifest, df40_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ffpp = pd.read_csv(ffpp_manifest, dtype=str, keep_default_na=False)
    df40 = pd.read_csv(df40_manifest, dtype=str, keep_default_na=False)
    combined = pd.concat(
        [prepare_ffpp(ffpp, project_root), prepare_df40(df40, project_root)],
        ignore_index=True,
    )
    combined, groups = rebuild_global_groups(combined)
    combined = combined.sort_values(
        ["split", "label", "method", "video_id", "frame_index"],
        key=lambda values: values.map({"train": 0, "val": 1, "test": 2}).fillna(values)
        if values.name == "split" else values,
    ).reset_index(drop=True)
    checks = validate(combined, project_root)

    master_path = output_dir / "manifest.csv"
    write_csv(master_path, combined)
    module_summaries = {}
    for module in MODULE_METHODS:
        selected = combined[combined["module"] == module].reset_index(drop=True)
        write_csv(output_dir / "modules" / f"{module}.csv", selected)
        module_summary = {
            "module": module,
            "archive": ARCHIVE_NAMES[module],
            "methods": list(MODULE_METHODS[module]),
            "images": len(selected),
            "splits": {
                split: {
                    "images": int((selected["split"] == split).sum()),
                    "by_method": selected[selected["split"] == split]["method"].value_counts().to_dict(),
                }
                for split in SPLITS
            },
            "manifest_sha256": sha256(output_dir / "modules" / f"{module}.csv"),
        }
        module_summaries[module] = module_summary
        write_json(output_dir / "modules" / f"{module}.json", module_summary)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "ffpp_df40_global_source_aware_v1",
        "archive_root": "deepfake_data_v1",
        "split_policy": (
            "FF++ c23 and retained DF40 rows inherit the existing FF++-aligned "
            "source membership; DF40 driver conflicts remain excluded."
        ),
        "inputs": {
            "ffpp_manifest": str(ffpp_manifest),
            "ffpp_manifest_sha256": sha256(ffpp_manifest),
            "df40_manifest": str(df40_manifest),
            "df40_manifest_sha256": sha256(df40_manifest),
        },
        "master_manifest_sha256": sha256(master_path),
        "images": len(combined),
        "groups": len(groups),
        "splits": {
            split: {
                "images": int((combined["split"] == split).sum()),
                "real": int(((combined["split"] == split) & (combined["label"] == "real")).sum()),
                "fake": int(((combined["split"] == split) & (combined["label"] == "fake")).sum()),
                "by_method": combined[combined["split"] == split]["method"].value_counts().to_dict(),
            }
            for split in SPLITS
        },
        "modules": module_summaries,
        "checks": checks,
    }
    write_json(output_dir / "split_summary.json", summary)
    write_json(
        output_dir / "module_registry.json",
        {
            "archive_root": "deepfake_data_v1",
            "modules": {
                module: {
                    "archive": ARCHIVE_NAMES[module],
                    "manifest": f"manifests/ffpp_df40_global_v1/modules/{module}.csv",
                    "methods": list(methods),
                }
                for module, methods in MODULE_METHODS.items()
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
