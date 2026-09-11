import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from mtcnn import MTCNN

from analyze_preprocessing_similarity import (
    load_actual_pilot_pairs,
    load_proxy_pairs,
    resolve_standard_xception_geometry,
)
from extract_ffpp_frames import is_valid_face_detection


CROP_SIZE = 299


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-detect faces and quantify preservation by the unchanged "
            "Standard Xception center crop."
        )
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("instagram_pipeline/pilot_v1"),
    )
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=Path("instagram_pipeline/proxy_web_q94_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/preprocessing_diagnosis_v1"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--proxy-limit",
        type=int,
        default=None,
        help="Smoke-test only; omit for the complete proxy test set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_resize_transform():
    _, data_config, transform_repr = resolve_standard_xception_geometry()
    import timm

    model = timm.create_model("xception", pretrained=False, num_classes=2)
    transform = timm.data.create_transform(
        **data_config, is_training=False
    )
    return transform.transforms[0], data_config, transform_repr


def crop_box(width, height):
    left = int(round((width - CROP_SIZE) / 2.0))
    top = int(round((height - CROP_SIZE) / 2.0))
    return left, top, left + CROP_SIZE, top + CROP_SIZE


def select_valid_face(detections, image_shape):
    valid = [
        result
        for result in detections
        if is_valid_face_detection(result, image_shape)
    ]
    if not valid:
        return None
    return max(valid, key=lambda result: result.get("confidence", 0.0))


def face_crop_metrics(face, width, height):
    if face is None:
        return {
            "detected": False,
            "confidence": None,
            "bbox_x": None,
            "bbox_y": None,
            "bbox_w": None,
            "bbox_h": None,
            "bbox_normalized": None,
            "fully_in_crop": None,
            "partially_cut": None,
            "fully_outside_crop": None,
            "center_outside_crop": None,
            "bbox_visible_fraction": None,
            "visible_face_area_ratio_299": None,
        }
    x, y, box_width, box_height = [float(value) for value in face["box"]]
    x2 = x + box_width
    y2 = y + box_height
    left, top, right, bottom = crop_box(width, height)
    intersection_width = max(0.0, min(x2, right) - max(x, left))
    intersection_height = max(0.0, min(y2, bottom) - max(y, top))
    intersection_area = intersection_width * intersection_height
    bbox_area = box_width * box_height
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    fully_in = x >= left and y >= top and x2 <= right and y2 <= bottom
    fully_outside = intersection_area == 0.0
    return {
        "detected": True,
        "confidence": float(face["confidence"]),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": box_width,
        "bbox_h": box_height,
        "bbox_normalized": (
            x / width,
            y / height,
            x2 / width,
            y2 / height,
        ),
        "fully_in_crop": fully_in,
        "partially_cut": intersection_area > 0.0 and not fully_in,
        "fully_outside_crop": fully_outside,
        "center_outside_crop": not (
            left <= center_x <= right and top <= center_y <= bottom
        ),
        "bbox_visible_fraction": intersection_area / bbox_area,
        "visible_face_area_ratio_299": intersection_area / (CROP_SIZE**2),
    }


def normalized_iou(first, second):
    if first is None or second is None:
        return None
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else None


def prepare_entries(pairs, resize):
    entries = []
    for pair_index, pair in enumerate(pairs):
        for side in ["raw", "target"]:
            path = pair["source_path"] if side == "raw" else pair["target_path"]
            with PIL.Image.open(path) as image:
                resized = resize(image.convert("RGB"))
            entries.append(
                {
                    "pair_index": pair_index,
                    "side": side,
                    "array": np.asarray(resized, dtype=np.uint8),
                    "width": resized.width,
                    "height": resized.height,
                }
            )
    return entries


def detect_entries(entries, detector, batch_size):
    results = {}
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        detections = detector.detect_faces(
            [entry["array"] for entry in batch],
            fit_to_image=True,
        )
        if len(detections) != len(batch):
            raise RuntimeError("MTCNN batch output length mismatch.")
        for entry, detected in zip(batch, detections):
            face = select_valid_face(detected, entry["array"].shape)
            results[(entry["pair_index"], entry["side"])] = (
                face_crop_metrics(face, entry["width"], entry["height"])
            )
        completed = min(start + batch_size, len(entries))
        if completed % 200 == 0 or completed == len(entries):
            print(f"Face detections: {completed}/{len(entries)}")
    return results


def analyze_dataset(pairs, resize, detector, batch_size):
    entries = prepare_entries(pairs, resize)
    detections = detect_entries(entries, detector, batch_size)
    rows = []
    for index, pair in enumerate(pairs):
        raw = detections[(index, "raw")]
        target = detections[(index, "target")]
        row = {
            key: value
            for key, value in pair.items()
            if key not in {"source_path", "target_path"}
        }
        row["source_path"] = str(pair["source_path"])
        row["target_path"] = str(pair["target_path"])
        for side, values in [("raw", raw), ("target", target)]:
            for key, value in values.items():
                if key != "bbox_normalized":
                    row[f"{side}_{key}"] = value
        row["raw_target_normalized_bbox_iou"] = normalized_iou(
            raw["bbox_normalized"], target["bbox_normalized"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def mean_or_none(values):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else None


def distribution(values):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not len(numeric):
        return None
    return {
        "n": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def summarize(frame):
    pairs = len(frame)
    raw_detected = frame["raw_detected"].astype(bool)
    target_detected = frame["target_detected"].astype(bool)
    both = raw_detected & target_detected

    def detected_fraction(side, column):
        mask = raw_detected if side == "raw" else target_detected
        if not mask.any():
            return None
        return float(frame.loc[mask, f"{side}_{column}"].astype(bool).mean())

    return {
        "pairs": int(pairs),
        "raw_valid_detection_count": int(raw_detected.sum()),
        "target_valid_detection_count": int(target_detected.sum()),
        "both_valid_detection_count": int(both.sum()),
        "raw_detection_rate": float(raw_detected.mean()),
        "target_detection_rate": float(target_detected.mean()),
        "both_detection_rate": float(both.mean()),
        "raw_full_bbox_in_crop_rate_among_detected": detected_fraction(
            "raw", "fully_in_crop"
        ),
        "target_full_bbox_in_crop_rate_among_detected": detected_fraction(
            "target", "fully_in_crop"
        ),
        "raw_full_bbox_in_crop_count": int(
            frame.loc[raw_detected, "raw_fully_in_crop"].astype(bool).sum()
        ),
        "target_full_bbox_in_crop_count": int(
            frame.loc[
                target_detected, "target_fully_in_crop"
            ].astype(bool).sum()
        ),
        "raw_partially_cut_rate_among_detected": detected_fraction(
            "raw", "partially_cut"
        ),
        "target_partially_cut_rate_among_detected": detected_fraction(
            "target", "partially_cut"
        ),
        "raw_partially_cut_count": int(
            frame.loc[raw_detected, "raw_partially_cut"].astype(bool).sum()
        ),
        "target_partially_cut_count": int(
            frame.loc[
                target_detected, "target_partially_cut"
            ].astype(bool).sum()
        ),
        "raw_fully_outside_crop_count": int(
            frame.loc[
                raw_detected, "raw_fully_outside_crop"
            ].astype(bool).sum()
        ),
        "target_fully_outside_crop_count": int(
            frame.loc[
                target_detected, "target_fully_outside_crop"
            ].astype(bool).sum()
        ),
        "raw_fully_outside_crop_rate_among_detected": detected_fraction(
            "raw", "fully_outside_crop"
        ),
        "target_fully_outside_crop_rate_among_detected": detected_fraction(
            "target", "fully_outside_crop"
        ),
        "raw_center_outside_rate_among_detected": detected_fraction(
            "raw", "center_outside_crop"
        ),
        "target_center_outside_rate_among_detected": detected_fraction(
            "target", "center_outside_crop"
        ),
        "raw_center_outside_count": int(
            frame.loc[
                raw_detected, "raw_center_outside_crop"
            ].astype(bool).sum()
        ),
        "target_center_outside_count": int(
            frame.loc[
                target_detected, "target_center_outside_crop"
            ].astype(bool).sum()
        ),
        "raw_visible_face_area_ratio_299": distribution(
            frame.loc[raw_detected, "raw_visible_face_area_ratio_299"]
        ),
        "target_visible_face_area_ratio_299": distribution(
            frame.loc[
                target_detected, "target_visible_face_area_ratio_299"
            ]
        ),
        "mean_raw_target_normalized_bbox_iou": mean_or_none(
            frame.loc[both, "raw_target_normalized_bbox_iou"]
        ),
    }


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    repo_root = Path(__file__).resolve().parent
    pilot_dir = args.pilot_dir.resolve()
    proxy_dir = args.proxy_dir.resolve()
    output_dir = args.output_dir.resolve()
    actual_path = output_dir / "real_instagram_pilot_face_coverage.csv"
    proxy_path = output_dir / "instagram_proxy_test_face_coverage.csv"
    summary_path = output_dir / "face_coverage_summary.json"
    if not args.overwrite and any(
        path.exists() for path in [actual_path, proxy_path, summary_path]
    ):
        raise FileExistsError("Face-coverage outputs exist; use --overwrite.")

    resize, data_config, transform_repr = resolve_resize_transform()
    actual_pairs, _ = load_actual_pilot_pairs(repo_root, pilot_dir)
    proxy_pairs, _ = load_proxy_pairs(repo_root, proxy_dir)
    if args.proxy_limit is not None:
        proxy_pairs = proxy_pairs[: args.proxy_limit]

    detector = MTCNN(device="CPU:0")
    actual = analyze_dataset(
        actual_pairs, resize, detector, args.batch_size
    )
    proxy = analyze_dataset(proxy_pairs, resize, detector, args.batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual.to_csv(actual_path, index=False)
    proxy.to_csv(proxy_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Standard Xception center-crop face preservation",
        "standard_xception_unchanged": True,
        "bbox_source": (
            "MTCNN re-detection on each image after exact Xception Resize(333); "
            "the extraction-time bboxes were not stored"
        ),
        "selection": (
            "highest-confidence detection passing the original extraction "
            "confidence, area, aspect-ratio, boundary, and center-Y filters"
        ),
        "data_config": data_config,
        "transform": transform_repr,
        "datasets": {
            "real_instagram_pilot": summarize(actual),
            "instagram_proxy_test": summarize(proxy),
        },
        "proxy_by_method": {
            method: summarize(group)
            for method, group in proxy.groupby("method", sort=True)
        },
        "proxy_by_source_resolution": {
            resolution: summarize(group)
            for resolution, group in proxy.groupby(
                "source_resolution", sort=True
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "tensorflow": tf.__version__,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Pilot pairs: {len(actual)}")
    print(f"Proxy pairs: {len(proxy)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
