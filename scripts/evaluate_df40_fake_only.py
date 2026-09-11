import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import timm
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from xception_preprocessing import (
    LETTERBOX_NAME,
    evaluation_transform_from_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Letterbox checkpoint on a fake-only DF40 manifest."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-name", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FakeDataset(Dataset):
    def __init__(self, frame, data_root, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row.source_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "index": index}


def probability_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def fake_metrics(frame, threshold):
    probabilities = frame["fake_probability"].to_numpy(dtype=np.float64)
    predictions = probabilities >= threshold
    return {
        "images": len(frame),
        "fake_recall": float(predictions.mean()),
        "missed_fakes": int((~predictions).sum()),
        "fake_probability": probability_stats(probabilities),
    }


def video_metrics(frame, threshold):
    videos = (
        frame.groupby(["method", "family", "group_id", "video_id"], as_index=False)
        .agg(
            frames=("fake_probability", "size"),
            mean_fake_probability=("fake_probability", "mean"),
            median_fake_probability=("fake_probability", "median"),
        )
    )
    videos["predicted_fake"] = videos["mean_fake_probability"] >= threshold
    summary = {
        "videos": len(videos),
        "fake_recall": float(videos["predicted_fake"].mean()),
        "missed_fakes": int((~videos["predicted_fake"]).sum()),
        "mean_fake_probability": probability_stats(
            videos["mean_fake_probability"].to_numpy()
        ),
    }
    return videos, summary


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for path in (data_root, manifest_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("Threshold must be between zero and one.")

    frame = pd.read_csv(
        manifest_path,
        dtype={"group_id": str, "video_id": str, "source_ids": str},
    )
    required = {
        "split", "label", "method", "family", "group_id", "video_id",
        "source_ids", "source_path",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.empty or set(frame["label"]) != {"fake"}:
        raise ValueError("This evaluator requires a non-empty fake-only manifest.")
    missing_paths = [
        path for path in frame.source_path if not (data_root / path).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Evaluation images are missing. First entries: "
            + ", ".join(missing_paths[:10])
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    preprocessing_name = checkpoint["config"].get("preprocessing_name")
    if preprocessing_name != LETTERBOX_NAME:
        raise ValueError(
            f"Expected {LETTERBOX_NAME!r}, got {preprocessing_name!r}."
        )
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    transform = evaluation_transform_from_checkpoint(checkpoint)
    dataset = FakeDataset(frame, data_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    probabilities = np.empty(len(frame), dtype=np.float32)
    observed_indices = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc="DF40 fake-only evaluation"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
            batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
            indices = batch["index"].tolist()
            probabilities[indices] = batch_probabilities.float().cpu().numpy()
            observed_indices.extend(indices)
    if observed_indices != list(range(len(frame))):
        raise RuntimeError("Evaluation order changed unexpectedly.")

    predictions = frame.copy()
    predictions["fake_probability"] = probabilities
    predictions["predicted_fake"] = probabilities >= args.threshold
    predictions["correct"] = predictions["predicted_fake"]
    videos, overall_video = video_metrics(predictions, args.threshold)
    by_method = {}
    for method, subset in predictions.groupby("method", sort=True):
        method_videos, method_video_summary = video_metrics(
            subset, args.threshold
        )
        by_method[method] = {
            "family": subset["family"].iloc[0],
            "image_level": fake_metrics(subset, args.threshold),
            "video_level": method_video_summary,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    videos.to_csv(output_dir / "video_predictions.csv", index=False)
    result = {
        "evaluation_name": args.evaluation_name,
        "scope": "DF40 FF-domain fake-only test",
        "threshold": args.threshold,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "data_root": str(data_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_auc": checkpoint["best_val_auc"],
        "model": checkpoint["model_name"],
        "preprocessing_name": preprocessing_name,
        "image_level": fake_metrics(predictions, args.threshold),
        "video_level": overall_video,
        "by_method": by_method,
        "not_reported": [
            "ROC-AUC, specificity, precision, and binary F1 require real negatives."
        ],
        "interpretation_limit": (
            "DF40 inputs are 256x256 processed face crops whereas this checkpoint "
            "was trained on FF++ raw full frames; results are a pilot combining "
            "manipulation and preprocessing shifts."
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torchvision": torchvision.__version__,
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Predictions: {output_dir / 'predictions.csv'}")
    print(f"Video predictions: {output_dir / 'video_predictions.csv'}")
    print(f"Metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
