#!/usr/bin/env python3
"""Preprocess FF++ c23 videos with the DF40/DeepfakeBench face pipeline.

The image operation intentionally mirrors DeepfakeBench's published
``preprocessing/preprocess.py`` implementation:

* uniformly choose 32 frame indices with ``numpy.linspace``;
* detect the largest dlib frontal face after one upsampling pass;
* derive five alignment points from the 81-point landmark predictor;
* estimate the same similarity transform with a 1.3 scale margin;
* write a 256x256 aligned PNG and aligned 81-point landmark array.

The surrounding orchestration adds resumability and provenance records but does
not alter the source videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_DETECTOR = None
_PREDICTOR = None
_SETTINGS: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepfakes-dir", type=Path, required=True)
    parser.add_argument("--face2face-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--method",
        choices=("deepfakes", "face2face", "both"),
        default="both",
        help="Process one method or both methods.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-method video limit for validation runs.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_latest_video_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            latest[(record["method"], record["video_id"])] = record
    return latest


def shape_to_np(shape: Any) -> np.ndarray:
    coordinates = np.zeros((shape.num_parts, 2), dtype=np.int32)
    for index in range(shape.num_parts):
        coordinates[index] = (shape.part(index).x, shape.part(index).y)
    return coordinates


def get_keypoints(rgb: np.ndarray, face: Any, predictor: Any) -> np.ndarray:
    shape = predictor(rgb, face)
    indices = (37, 44, 30, 49, 55)
    return np.asarray(
        [[shape.part(index).x, shape.part(index).y] for index in indices],
        dtype=np.float32,
    )


def align_and_crop(
    rgb: np.ndarray,
    landmarks: np.ndarray,
    resolution: int,
    scale: float,
) -> np.ndarray:
    # Import inside workers so a missing optional dependency is reported clearly.
    from skimage import transform as trans

    target_size = [112, 112]
    destination = np.asarray(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    destination[:, 0] += 8.0
    destination[:, 0] *= resolution / target_size[0]
    destination[:, 1] *= resolution / target_size[1]

    margin_rate = scale - 1.0
    x_margin = resolution * margin_rate / 2.0
    y_margin = resolution * margin_rate / 2.0
    destination[:, 0] += x_margin
    destination[:, 1] += y_margin
    destination[:, 0] *= resolution / (resolution + 2.0 * x_margin)
    destination[:, 1] *= resolution / (resolution + 2.0 * y_margin)

    transform = trans.SimilarityTransform()
    if not transform.estimate(landmarks.astype(np.float32), destination):
        raise RuntimeError("Similarity transform estimation failed")
    matrix = transform.params[:2, :]
    aligned = cv2.warpAffine(rgb, matrix, (resolution, resolution))
    return cv2.resize(aligned, (resolution, resolution))


def extract_aligned_face(frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = _DETECTOR(rgb, 1)
    if not faces:
        return None, None

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    keypoints = get_keypoints(rgb, face, _PREDICTOR)
    cropped_rgb = align_and_crop(
        rgb,
        keypoints,
        resolution=int(_SETTINGS["resolution"]),
        scale=float(_SETTINGS["scale"]),
    )
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

    # This deliberately matches the reference implementation, including its
    # second detector call on the BGR crop and selection of the first face.
    aligned_faces = _DETECTOR(cropped_bgr, 1)
    if not aligned_faces:
        return None, None
    aligned_shape = _PREDICTOR(cropped_bgr, aligned_faces[0])
    return cropped_bgr, shape_to_np(aligned_shape)


def init_worker(predictor_path: str, resolution: int, scale: float) -> None:
    global _DETECTOR, _PREDICTOR, _SETTINGS
    import dlib

    cv2.setNumThreads(1)
    _DETECTOR = dlib.get_frontal_face_detector()
    _PREDICTOR = dlib.shape_predictor(predictor_path)
    _SETTINGS = {"resolution": resolution, "scale": scale}


def process_video(task: tuple[str, str, str, int]) -> dict[str, Any]:
    method, video_path_text, output_root_text, num_frames = task
    video_path = Path(video_path_text)
    output_root = Path(output_root_text)
    video_id = video_path.stem
    frame_dir = output_root / method / "ff" / "frames" / video_id
    landmark_dir = output_root / method / "ff" / "landmarks" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "method": method,
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "open_failed",
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "frame_records": [],
        }

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = set(
        np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int).tolist()
    ) if frame_count > 0 else set()
    frame_records: list[dict[str, Any]] = []

    current = 0
    while current < frame_count:
        ok, frame = capture.read()
        if not ok:
            frame_records.append({"frame_index": current, "status": "decode_failed"})
            break
        if current not in selected:
            current += 1
            continue

        try:
            cropped, landmarks = extract_aligned_face(frame)
            if cropped is None or landmarks is None:
                frame_records.append({"frame_index": current, "status": "face_failed"})
            else:
                image_path = frame_dir / f"{current:03d}.png"
                landmark_path = landmark_dir / f"{current:03d}.npy"
                if not image_path.exists():
                    if not cv2.imwrite(str(image_path), cropped):
                        raise OSError(f"Could not write {image_path}")
                if not landmark_path.exists():
                    np.save(str(landmark_path), landmarks)
                frame_records.append(
                    {
                        "frame_index": current,
                        "status": "saved",
                        "image_path": str(image_path),
                        "landmark_path": str(landmark_path),
                    }
                )
        except Exception as error:
            frame_records.append(
                {
                    "frame_index": current,
                    "status": "processing_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        current += 1

    capture.release()
    saved = sum(record["status"] == "saved" for record in frame_records)
    failed = sum(record["status"] != "saved" for record in frame_records)
    return {
        "method": method,
        "video_id": video_id,
        "video_path": str(video_path),
        "video_size_bytes": video_path.stat().st_size,
        "frame_count": frame_count,
        "selected_indices": sorted(selected),
        "selected_count": len(selected),
        "saved_count": saved,
        "failed_count": failed,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "frame_records": frame_records,
    }


def write_summary_csv(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    fields = [
        "method",
        "video_id",
        "video_path",
        "video_size_bytes",
        "frame_count",
        "selected_count",
        "saved_count",
        "failed_count",
        "status",
        "elapsed_seconds",
        "completed_at",
    ]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(records):
            record = records[key]
            writer.writerow({field: record.get(field) for field in fields})
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> dict[str, Path]:
    if args.num_frames <= 0 or args.resolution <= 0 or args.workers <= 0:
        raise ValueError("num-frames, resolution, and workers must be positive")
    if args.scale <= 0:
        raise ValueError("scale must be positive")
    for path in (args.deepfakes_dir, args.face2face_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.predictor.is_file():
        raise FileNotFoundError(args.predictor)
    return {
        "deepfakes": args.deepfakes_dir.resolve(),
        "face2face": args.face2face_dir.resolve(),
    }


def main() -> None:
    args = parse_args()
    input_dirs = validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root.resolve()

    selected_methods = (
        ("deepfakes", "face2face") if args.method == "both" else (args.method,)
    )
    videos_by_method: dict[str, list[Path]] = {}
    for method in selected_methods:
        videos = sorted(input_dirs[method].glob("*.mp4"))
        if args.limit is not None:
            videos = videos[: args.limit]
        if not videos:
            raise RuntimeError(f"No MP4 files found for {method}: {input_dirs[method]}")
        videos_by_method[method] = videos

    predictor_hash = sha256(args.predictor)
    config_path = output_root / "preprocessing_config.json"
    config = {
        "protocol": "df40_deepfakebench_face_preprocessing",
        "reference": "SCLBD/DeepfakeBench preprocessing/preprocess.py",
        "frame_sampling": "np.linspace(0, frame_count - 1, 32, endpoint=True, dtype=int)",
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "scale": args.scale,
        "face_detector": "dlib frontal detector, upsample=1, largest face",
        "landmark_predictor": str(args.predictor.resolve()),
        "landmark_predictor_sha256": predictor_hash,
        "alignment_points": [37, 44, 30, 49, 55],
        "inputs": {method: str(input_dirs[method]) for method in selected_methods},
        "output_root": str(output_root),
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "protocol",
            "num_frames",
            "resolution",
            "scale",
            "landmark_predictor_sha256",
            "alignment_points",
        )
        mismatches = [key for key in comparable_keys if existing.get(key) != config.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Output root contains a different preprocessing configuration: {mismatches}"
            )
    else:
        config["created_at"] = utc_now()
        atomic_write_json(config_path, config)

    journal_path = output_root / "video_processing.jsonl"
    latest = load_latest_video_records(journal_path)
    tasks: list[tuple[str, str, str, int]] = []
    skipped = 0
    for method, videos in videos_by_method.items():
        for video in videos:
            prior = latest.get((method, video.stem))
            if prior and prior.get("status") == "complete":
                skipped += 1
                continue
            tasks.append((method, str(video), str(output_root), args.num_frames))

    total_discovered = sum(len(videos) for videos in videos_by_method.values())
    print(f"Discovered: {total_discovered} videos")
    print(f"Already complete: {skipped}")
    print(f"To process: {len(tasks)}")
    print(f"Output: {output_root}")
    if not tasks:
        write_summary_csv(output_root / "video_summary.csv", latest)
        print("Nothing to process.")
        return

    started = time.monotonic()
    processed = 0
    saved_total = 0
    failed_total = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(str(args.predictor.resolve()), args.resolution, args.scale),
    ) as executor:
        futures = {executor.submit(process_video, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:
                method, video_path, _, _ = task
                record = {
                    "method": method,
                    "video_id": Path(video_path).stem,
                    "video_path": video_path,
                    "status": "worker_error",
                    "error": f"{type(error).__name__}: {error}",
                    "completed_at": utc_now(),
                }
            append_jsonl(journal_path, record)
            latest[(record["method"], record["video_id"])] = record
            processed += 1
            saved_total += int(record.get("saved_count", 0))
            failed_total += int(record.get("failed_count", 0))
            if processed == 1 or processed % 10 == 0 or processed == len(tasks):
                elapsed = time.monotonic() - started
                rate = processed / elapsed if elapsed else 0.0
                remaining = (len(tasks) - processed) / rate if rate else 0.0
                print(
                    f"Processed {processed}/{len(tasks)} | "
                    f"saved={saved_total} failed={failed_total} | "
                    f"ETA={remaining / 60:.1f} min",
                    flush=True,
                )

    write_summary_csv(output_root / "video_summary.csv", latest)
    final = {
        "finished_at": utc_now(),
        "videos_discovered": total_discovered,
        "videos_previously_complete": skipped,
        "videos_processed_this_run": processed,
        "images_saved_this_run": saved_total,
        "sample_failures_this_run": failed_total,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(output_root / "run_summary.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
