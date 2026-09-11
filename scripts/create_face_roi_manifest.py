import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from mtcnn import MTCNN
from PIL import Image

CONFIDENCE_THRESHOLD = 0.90
MIN_FACE_AREA_RATIO = 0.02
MAX_FACE_AREA_RATIO = 0.60
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
MAX_CENTER_Y_RATIO = 0.75
REQUIRED_COLUMNS = {
    "split",
    "label",
    "method",
    "group_id",
    "video_id",
    "source_ids",
    "source_path",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a frozen raw-image Face-ROI manifest without changing "
            "the source dataset or split membership."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expansion", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--max-detection-side", type=int, default=1024)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
    )
    return parser.parse_args()


def select_device(requested):
    gpu_available = bool(tf.config.list_physical_devices("GPU"))
    if requested == "gpu" and not gpu_available:
        raise RuntimeError("TensorFlow cannot see a GPU.")
    if requested == "cpu":
        return "CPU:0"
    if requested == "gpu" or gpu_available:
        return "GPU:0"
    return "CPU:0"


def resize_for_detection(image, max_side):
    width, height = image.size
    if max(width, height) <= max_side:
        return image.copy()
    scale = max_side / max(width, height)
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(size, resample=Image.Resampling.BILINEAR)


def is_valid_detection(result, image_shape):
    confidence = float(result.get("confidence", 0.0))
    box = result.get("box")
    if confidence < CONFIDENCE_THRESHOLD or box is None or len(box) != 4:
        return False
    x, y, width, height = [float(value) for value in box]
    image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0:
        return False
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return False
    face_area_ratio = width * height / (image_width * image_height)
    aspect_ratio = height / width
    center_y = y + height / 2.0
    return (
        MIN_FACE_AREA_RATIO <= face_area_ratio <= MAX_FACE_AREA_RATIO
        and MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
        and center_y <= image_height * MAX_CENTER_Y_RATIO
    )


def select_face(detections, image_shape):
    valid = [
        detection
        for detection in detections
        if is_valid_detection(detection, image_shape)
    ]
    if not valid:
        return None, 0
    return max(valid, key=lambda item: float(item["confidence"])), len(valid)


def create_detection_row(
    manifest_index,
    image_size,
    detection_size,
    detections,
    expansion,
):
    detection_width, detection_height = detection_size
    face, valid_count = select_face(
        detections, (detection_height, detection_width, 3)
    )
    common = {
        "manifest_index": manifest_index,
        "roi_source_width": image_size[0],
        "roi_source_height": image_size[1],
        "roi_detection_width": detection_width,
        "roi_detection_height": detection_height,
        "roi_detection_count": len(detections),
        "roi_valid_detection_count": valid_count,
        "roi_expansion": expansion,
    }
    if face is None:
        return {
            **common,
            "roi_status": "fallback_letterbox",
            "roi_face_confidence": None,
            "roi_face_x1_normalized": None,
            "roi_face_y1_normalized": None,
            "roi_face_x2_normalized": None,
            "roi_face_y2_normalized": None,
            "roi_center_x_normalized": None,
            "roi_center_y_normalized": None,
            "roi_side_width_fraction": None,
            "roi_side_height_fraction": None,
            "roi_requires_padding": None,
        }

    x, y, width, height = [float(value) for value in face["box"]]
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    side = expansion * max(width, height)
    left = center_x - side / 2.0
    top = center_y - side / 2.0
    right = left + side
    bottom = top + side
    return {
        **common,
        "roi_status": "detected",
        "roi_face_confidence": float(face["confidence"]),
        "roi_face_x1_normalized": x / detection_width,
        "roi_face_y1_normalized": y / detection_height,
        "roi_face_x2_normalized": (x + width) / detection_width,
        "roi_face_y2_normalized": (y + height) / detection_height,
        "roi_center_x_normalized": center_x / detection_width,
        "roi_center_y_normalized": center_y / detection_height,
        "roi_side_width_fraction": side / detection_width,
        "roi_side_height_fraction": side / detection_height,
        "roi_requires_padding": bool(
            left < 0
            or top < 0
            or right > detection_width
            or bottom > detection_height
        ),
    }


def protocol_config(args, manifest_sha256, device):
    return {
        "name": "face_roi_mtcnn_1p5_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": manifest_sha256,
        "data_root": str(args.data_root),
        "rows_are_dropped": False,
        "bbox_source": "MTCNN detection on FF++ raw PNG only",
        "target_bbox_policy": (
            "reuse the frozen raw-image normalized ROI; never redetect on "
            "Instagram or proxy images"
        ),
        "face_selection": (
            "highest-confidence detection passing the extraction filters"
        ),
        "expansion": args.expansion,
        "roi_shape": "square centered on face bbox",
        "out_of_frame_policy": "neutral RGB 128 padding",
        "detection_resize": (
            "preserve aspect ratio; cap longest side at "
            f"{args.max_detection_side} pixels"
        ),
        "fallback": "full-frame letterbox when no valid detection exists",
        "detector_filters": {
            "confidence_min": CONFIDENCE_THRESHOLD,
            "face_area_ratio_min": MIN_FACE_AREA_RATIO,
            "face_area_ratio_max": MAX_FACE_AREA_RATIO,
            "bbox_aspect_ratio_min": MIN_ASPECT_RATIO,
            "bbox_aspect_ratio_max": MAX_ASPECT_RATIO,
            "face_center_y_ratio_max": MAX_CENTER_Y_RATIO,
        },
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every,
        "requested_device": args.device,
        "resolved_device": device,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "tensorflow": tf.__version__,
            "mtcnn": importlib.metadata.version("mtcnn"),
        },
    }


def validate_resume_config(config, expected):
    keys = [
        "source_manifest_sha256",
        "data_root",
        "expansion",
        "detection_resize",
        "face_selection",
        "detector_filters",
        "batch_size",
        "resolved_device",
        "versions",
    ]
    mismatches = [key for key in keys if config.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(f"ROI manifest resume mismatch: {mismatches}")


def summarize(output):
    status_counts = output["roi_status"].value_counts().to_dict()
    detected = output[output["roi_status"].eq("detected")]
    return {
        "images": int(len(output)),
        "status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "detection_rate": float(len(detected) / len(output)),
        "fallback_rate": float(1.0 - len(detected) / len(output)),
        "padding_required_count": int(
            detected["roi_requires_padding"].astype(bool).sum()
        ),
        "face_confidence": {
            "mean": float(detected["roi_face_confidence"].mean()),
            "min": float(detected["roi_face_confidence"].min()),
            "max": float(detected["roi_face_confidence"].max()),
        },
        "by_split": {
            str(key): {
                "images": int(len(group)),
                "detected": int(group["roi_status"].eq("detected").sum()),
                "fallback": int(
                    group["roi_status"].eq("fallback_letterbox").sum()
                ),
            }
            for key, group in output.groupby("split", sort=True)
        },
        "by_method": {
            str(key): {
                "images": int(len(group)),
                "detected": int(group["roi_status"].eq("detected").sum()),
                "fallback": int(
                    group["roi_status"].eq("fallback_letterbox").sum()
                ),
            }
            for key, group in output.groupby("method", sort=True)
        },
    }


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.expansion <= 1.0:
        raise ValueError("--expansion must be greater than 1.0.")
    if (
        args.batch_size < 1
        or args.checkpoint_every < 1
        or args.max_detection_side < 299
    ):
        raise ValueError("Invalid batch size or detection size.")
    if not args.data_root.is_dir() or not args.manifest.is_file():
        raise FileNotFoundError("Data root or source manifest is missing.")

    manifest_sha256 = sha256_file(args.manifest)
    source = pd.read_csv(
        args.manifest,
        dtype={
            "group_id": str,
            "video_id": str,
            "source_ids": str,
            "source_path": str,
        },
    )
    if missing := REQUIRED_COLUMNS - set(source.columns):
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    if source["source_path"].duplicated().any():
        raise ValueError("Source manifest paths are not unique.")
    missing_paths = [
        path
        for path in source["source_path"]
        if not (args.data_root / path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Manifest images are missing. First entries: "
            + ", ".join(missing_paths[:10])
        )

    device = select_device(args.device)
    expected_config = protocol_config(
        args, manifest_sha256=manifest_sha256, device=device
    )
    config_path = args.output_dir / "config.json"
    partial_path = args.output_dir / "detections.partial.csv"
    output_path = args.output_dir / "manifest.csv"
    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_config(saved_config, expected_config)
    else:
        if any(args.output_dir.iterdir()):
            raise RuntimeError(
                "ROI output directory is non-empty but has no config.json."
            )
        write_json(config_path, expected_config)

    if output_path.is_file():
        completed = pd.read_csv(output_path)
        if len(completed) != len(source):
            raise RuntimeError("Existing complete ROI manifest has wrong length.")
        print(f"Complete ROI manifest already exists: {output_path}")
        return

    if partial_path.is_file():
        partial = pd.read_csv(partial_path)
        if partial["manifest_index"].duplicated().any():
            raise RuntimeError("Partial detections contain duplicate indices.")
        completed_indices = set(partial["manifest_index"].astype(int))
        print(f"Resuming after {len(completed_indices)}/{len(source)} images.")
    else:
        completed_indices = set()

    detector = MTCNN(device=device)
    remaining = [
        index for index in range(len(source)) if index not in completed_indices
    ]
    pending_rows = []
    for start in range(0, len(remaining), args.batch_size):
        indices = remaining[start : start + args.batch_size]
        arrays = []
        metadata = []
        for index in indices:
            image_path = args.data_root / source.iloc[index]["source_path"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_size = image.size
                detection_image = resize_for_detection(
                    image, args.max_detection_side
                )
            arrays.append(np.asarray(detection_image, dtype=np.uint8))
            metadata.append((index, image_size, detection_image.size))

        detections = detector.detect_faces(
            arrays,
            fit_to_image=True,
            box_format="xywh",
            output_type="json",
        )
        if len(detections) != len(indices):
            raise RuntimeError("MTCNN batch output length mismatch.")
        pending_rows.extend(
            [
                create_detection_row(
                    index,
                    image_size,
                    detection_size,
                    detected,
                    args.expansion,
                )
                for (index, image_size, detection_size), detected in zip(
                    metadata, detections
                )
            ]
        )
        completed = len(completed_indices) + min(
            start + len(indices), len(remaining)
        )
        should_checkpoint = (
            len(pending_rows) >= args.checkpoint_every
            or completed == len(source)
        )
        if should_checkpoint:
            pd.DataFrame(pending_rows).to_csv(
                partial_path,
                mode="a",
                header=not partial_path.exists(),
                index=False,
            )
            pending_rows.clear()
            print(f"Face detections: {completed}/{len(source)}")

    detections = pd.read_csv(partial_path).sort_values("manifest_index")
    expected_indices = list(range(len(source)))
    if detections["manifest_index"].astype(int).tolist() != expected_indices:
        raise RuntimeError("ROI detections do not cover the source manifest once.")
    output = source.copy()
    for column in detections.columns:
        if column != "manifest_index":
            output[column] = detections[column].to_numpy()
    temporary = output_path.with_suffix(".csv.tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(output_path)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": manifest_sha256,
        "roi_manifest": str(output_path),
        "roi_manifest_sha256": sha256_file(output_path),
        **summarize(output),
    }
    write_json(summary_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Face-ROI manifest: {output_path}")


if __name__ == "__main__":
    main()
