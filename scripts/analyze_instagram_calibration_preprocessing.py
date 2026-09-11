import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from mtcnn import MTCNN

from analyze_center_crop_faces import (
    analyze_dataset,
    resolve_resize_transform,
    summarize as summarize_faces,
)
from analyze_preprocessing_similarity import (
    analyze_pairs,
    resolve_standard_xception_geometry,
    save_visual,
    sha256_file,
    summarize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Standard Xception masking and face preservation on the "
            "real Instagram calibration pairs."
        )
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("instagram_pipeline/calibration_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--visual-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pairs(repo_root, calibration_dir):
    source_path = calibration_dir / "calibration_manifest.csv"
    target_path = calibration_dir / "download_manifest.csv"
    source = pd.read_csv(
        source_path,
        dtype={"group_id": str, "video_id": str, "source_path": str},
    )
    target = pd.read_csv(target_path)
    if source["sample_id"].duplicated().any():
        raise ValueError("Duplicate source sample IDs.")
    if target["sample_id"].duplicated().any():
        raise ValueError("Duplicate target sample IDs.")
    frame = source.merge(
        target,
        on=["sample_id", "batch_id", "carousel_position"],
        validate="one_to_one",
    )
    if len(frame) != 60:
        raise ValueError(f"Expected 60 calibration pairs, found {len(frame)}")
    pairs = []
    for row in frame.sort_values("upload_order").itertuples():
        pairs.append(
            {
                "dataset": "real_instagram_calibration_v2",
                "sample_id": row.sample_id,
                "split": row.split,
                "label": row.label,
                "method": row.method,
                "group_id": row.group_id,
                "video_id": row.video_id,
                "source_resolution": (
                    f"{row.source_width}x{row.source_height}"
                ),
                "source_path": repo_root / row.source_path,
                "target_path": calibration_dir / row.download_path,
                "source_reference_path": row.source_path,
            }
        )
    return pairs, source_path, target_path


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    repo_root = Path(__file__).resolve().parent
    calibration_dir = args.calibration_dir.resolve()
    output_dir = calibration_dir / "analysis"
    similarity_path = output_dir / "preprocessing_pair_metrics.csv"
    face_path = output_dir / "face_coverage.csv"
    summary_path = output_dir / "preprocessing_summary.json"
    outputs = [similarity_path, face_path, summary_path]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Analysis outputs exist; use --overwrite.")

    pairs, source_manifest, target_manifest = load_pairs(
        repo_root, calibration_dir
    )
    geometry, data_config, transform_repr = (
        resolve_standard_xception_geometry()
    )
    similarity, prepared = analyze_pairs(pairs, geometry)

    resize, _, _ = resolve_resize_transform()
    faces = analyze_dataset(
        pairs, resize, MTCNN(device="CPU:0"), args.batch_size
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    similarity.to_csv(similarity_path, index=False)
    faces.to_csv(face_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "real_instagram_calibration_v2",
        "pairs": len(pairs),
        "standard_xception_unchanged": True,
        "data_config": data_config,
        "transform": transform_repr,
        "input_manifests": {
            str(source_manifest.relative_to(repo_root)): sha256_file(
                source_manifest
            ),
            str(target_manifest.relative_to(repo_root)): sha256_file(
                target_manifest
            ),
        },
        "similarity": summarize(similarity),
        "similarity_by_method": {
            method: summarize(group)
            for method, group in similarity.groupby("method", sort=True)
        },
        "similarity_by_resolution": {
            resolution: summarize(group)
            for resolution, group in similarity.groupby(
                "source_resolution", sort=True
            )
        },
        "face_coverage": summarize_faces(faces),
        "face_coverage_by_method": {
            method: summarize_faces(group)
            for method, group in faces.groupby("method", sort=True)
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    visual_dir = output_dir / "preprocessing_visuals"
    visual_dir.mkdir(exist_ok=True)
    sample_ids = sorted(prepared)
    selected = random.Random(args.seed).sample(
        sample_ids, min(args.visual_count, len(sample_ids))
    )
    for sample_id in selected:
        save_visual(visual_dir / f"{sample_id}.png", prepared[sample_id])

    print(f"Pairs: {len(pairs)}")
    print(f"Similarity: {similarity_path}")
    print(f"Face coverage: {face_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
