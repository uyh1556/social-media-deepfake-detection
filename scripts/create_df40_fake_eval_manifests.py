import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


METHODS = {
    "simswap": {"display_name": "SimSwap", "family": "face_swap"},
    "blendface": {"display_name": "BlendFace", "family": "face_swap"},
    "sadtalker": {
        "display_name": "SadTalker",
        "family": "face_reenactment",
    },
    "wav2lip": {
        "display_name": "Wav2Lip",
        "family": "face_reenactment",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create all-local and source-disjoint fake-only DF40 test manifests."
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


def source_ids_for_video(method, video_id):
    if method in {"simswap", "blendface"}:
        match = re.fullmatch(r"(\d{3})_(\d{3})", video_id)
        if match is None:
            raise ValueError(f"Unexpected {method} video ID: {video_id}")
        return list(match.groups())
    match = re.match(r"^(\d{3})_", video_id)
    if match is None:
        raise ValueError(f"Unexpected {method} video ID: {video_id}")
    return [match.group(1)]


def load_source_membership(path):
    frame = pd.read_csv(
        path,
        dtype={"video_id": str, "group_id": str, "source_ids": str},
    )
    required = {"split", "label", "method", "video_id", "group_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    originals = frame[
        (frame["label"] == "real") & (frame["method"] == "original")
    ][["video_id", "split", "group_id"]].drop_duplicates()
    conflicts = originals.groupby("video_id")["split"].nunique().gt(1)
    if conflicts.any():
        raise RuntimeError("Source IDs occur in multiple source splits.")
    return {
        row.video_id: {"split": row.split, "group_id": row.group_id}
        for row in originals.itertuples(index=False)
    }


def build_rows(df40_root, membership):
    rows = []
    video_audit = []
    for method, metadata in METHODS.items():
        frames_root = df40_root / method / "test" / "ff" / "frames"
        if not frames_root.is_dir():
            raise FileNotFoundError(frames_root)
        video_dirs = sorted(path for path in frames_root.iterdir() if path.is_dir())
        if not video_dirs:
            raise RuntimeError(f"No video directories: {frames_root}")
        for video_dir in video_dirs:
            source_ids = source_ids_for_video(method, video_dir.name)
            records = [membership.get(source_id) for source_id in source_ids]
            source_splits = [
                record["split"] if record is not None else "missing"
                for record in records
            ]
            strict = all(split == "test" for split in source_splits)
            known_groups = {
                record["group_id"] for record in records if record is not None
            }
            if len(known_groups) > 1:
                raise RuntimeError(
                    f"Connected source IDs map to different groups: "
                    f"{method}/{video_dir.name} -> {sorted(known_groups)}"
                )
            group_id = (
                next(iter(known_groups))
                if known_groups
                else "missing__" + "__".join(source_ids)
            )
            frames = sorted(video_dir.glob("*.png"))
            if not frames:
                raise RuntimeError(f"No PNG frames: {video_dir}")
            video_audit.append(
                {
                    "method": metadata["display_name"],
                    "family": metadata["family"],
                    "video_id": video_dir.name,
                    "source_ids": "|".join(source_ids),
                    "source_split_membership": "|".join(source_splits),
                    "source_disjoint_test": strict,
                    "frames": len(frames),
                }
            )
            for frame_path in frames:
                rows.append(
                    {
                        "split": "test",
                        "label": "fake",
                        "method": metadata["display_name"],
                        "family": metadata["family"],
                        "group_id": group_id,
                        "video_id": video_dir.name,
                        "source_ids": "|".join(source_ids),
                        "source_split_membership": "|".join(source_splits),
                        "source_disjoint_test": strict,
                        "source_path": frame_path.relative_to(df40_root).as_posix(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(video_audit)


def count_by_method(frame):
    return {
        method: int(count)
        for method, count in frame.groupby("method", sort=True).size().items()
    }


def main():
    args = parse_args()
    df40_root = args.df40_root.resolve()
    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not df40_root.is_dir():
        raise FileNotFoundError(df40_root)
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    membership = load_source_membership(source_manifest)
    all_rows, video_audit = build_rows(df40_root, membership)
    strict_rows = all_rows[all_rows["source_disjoint_test"]].reset_index(drop=True)
    if strict_rows.empty:
        raise RuntimeError("The source-disjoint manifest would be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "all_local_test_fake.csv"
    strict_path = output_dir / "source_disjoint_test_fake.csv"
    audit_path = output_dir / "video_source_membership.csv"
    all_rows.to_csv(all_path, index=False)
    strict_rows.to_csv(strict_path, index=False)
    video_audit.to_csv(audit_path, index=False)

    video_status = defaultdict(Counter)
    for row in video_audit.itertuples(index=False):
        video_status[row.method][row.source_split_membership] += 1
    summary = {
        "protocol": "df40_ff_fake_only_eval_v1",
        "scope": "locally retained DF40 FF-domain fake test frames only",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "df40_root": str(df40_root),
        "all_local": {
            "images": len(all_rows),
            "videos": len(video_audit),
            "images_by_method": count_by_method(all_rows),
        },
        "source_disjoint": {
            "rule": "all parsed FF++ source IDs belong to source test split",
            "images": len(strict_rows),
            "videos": int(
                strict_rows[["method", "video_id"]].drop_duplicates().shape[0]
            ),
            "images_by_method": count_by_method(strict_rows),
        },
        "video_source_membership_by_method": {
            method: dict(sorted(counts.items()))
            for method, counts in sorted(video_status.items())
        },
        "limitations": [
            "All rows are fake; binary ROC-AUC, specificity, and precision cannot be computed.",
            "DF40 images are 256x256 processed face crops, while the checkpoint was trained on FF++ raw full frames.",
            "The all-local manifest contains source identities seen during checkpoint training or validation.",
        ],
    }
    (output_dir / "config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"All-local manifest: {all_path}")
    print(f"Source-disjoint manifest: {strict_path}")


if __name__ == "__main__":
    main()
