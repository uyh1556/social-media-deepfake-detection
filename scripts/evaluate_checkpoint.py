import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from train_baseline import (
    LABEL_MAP,
    classification_metrics,
    manipulation_metrics,
    sha256_file,
    write_json,
)
from xception_preprocessing import (
    FACE_ROI_NAME,
    crop_face_roi,
    evaluation_transform_from_checkpoint,
    validate_face_roi_columns,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved baseline checkpoint on one frozen split."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--strip-source-prefix",
        default=None,
        help="Remove one explicit leading path component at load time.",
    )
    parser.add_argument(
        "--allow-unverified-lineage",
        action="store_true",
        help="Allow evaluation when manifest lineage cannot be tied to training.",
    )
    return parser.parse_args()


def strip_path_prefix(path, prefix):
    relative = Path(path)
    if prefix is None:
        return relative.as_posix()
    if not relative.parts or relative.parts[0] != prefix:
        raise ValueError(f"Path does not start with {prefix!r}: {relative}")
    return Path(*relative.parts[1:]).as_posix()


class EvaluationDataset(Dataset):
    def __init__(
        self,
        frame,
        data_root,
        transform,
        preprocessing_name,
        input_size,
    ):
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.preprocessing_name = preprocessing_name
        self.input_size = int(input_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["resolved_path"]) as image:
            image = image.convert("RGB")
            if self.preprocessing_name == FACE_ROI_NAME:
                image = crop_face_roi(
                    image, row, fallback_size=self.input_size
                )
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(LABEL_MAP[row["label"]]),
            "index": index,
        }


@torch.no_grad()
def evaluate_with_predictions(model, loader, frame, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    labels = []
    predictions = []
    probabilities = []
    indices = []

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        batch_labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, batch_labels)
        batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
        batch_predictions = logits.argmax(dim=1)
        total_loss += loss.item() * batch_labels.size(0)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
        probabilities.extend(batch_probabilities.float().cpu().tolist())
        indices.extend(batch["index"].tolist())

    if indices != list(range(len(frame))):
        raise RuntimeError("Evaluation order changed unexpectedly.")
    methods = frame["method"].tolist()
    metrics = classification_metrics(labels, predictions, probabilities)
    metrics["loss"] = total_loss / len(frame)
    metrics["by_manipulation"] = manipulation_metrics(
        labels, predictions, probabilities, methods
    )
    return metrics, labels, predictions, probabilities


def write_predictions(path, frame, labels, predictions, probabilities):
    fieldnames = [
        "source_path", "source_reference_path", "split", "label", "method",
        "group_id", "video_id", "true_class", "predicted_class",
        "fake_probability", "correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in frame.iterrows():
            prediction = predictions[index]
            writer.writerow(
                {
                    "source_path": row.source_path,
                    "source_reference_path": row.source_reference_path,
                    "split": row.split,
                    "label": row.label,
                    "method": row.method,
                    "group_id": row.group_id,
                    "video_id": row.video_id,
                    "true_class": labels[index],
                    "predicted_class": prediction,
                    "fake_probability": probabilities[index],
                    "correct": prediction == labels[index],
                }
            )


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for path in [data_root, manifest_path, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"group_id": str, "video_id": str, "source_path": str},
    )
    required = {
        "split", "label", "method", "group_id", "video_id", "source_path"
    }
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame = manifest[manifest["split"] == args.split].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for split={args.split!r}")
    if set(frame.label) - set(LABEL_MAP):
        raise ValueError("Manifest contains unexpected labels.")
    if "source_reference_path" not in frame:
        frame["source_reference_path"] = frame["source_path"]
    frame["resolved_path"] = frame["source_path"].map(
        lambda path: strip_path_prefix(path, args.strip_source_prefix)
    )
    missing_paths = [
        path for path in frame.resolved_path if not (data_root / path).is_file()
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
    manifest_sha256 = sha256_file(manifest_path)
    training_manifest_sha256 = checkpoint["config"].get("manifest_sha256")
    if manifest_sha256 == training_manifest_sha256:
        manifest_lineage = "training_manifest_exact"
    else:
        derived_config_path = manifest_path.parent / "config.json"
        derived_config = (
            json.loads(derived_config_path.read_text(encoding="utf-8"))
            if derived_config_path.is_file()
            else {}
        )
        if (
            derived_config.get("source_manifest_sha256")
            == training_manifest_sha256
        ):
            manifest_lineage = "derived_from_training_manifest"
        elif args.allow_unverified_lineage:
            manifest_lineage = "unverified_override"
        else:
            raise RuntimeError(
                "Evaluation manifest lineage does not match the checkpoint's "
                "training manifest. Use --allow-unverified-lineage only after "
                "documenting the reason."
            )
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    preprocessing_name = checkpoint["config"].get(
        "preprocessing_name", "standard_center_crop_299"
    )
    if preprocessing_name == FACE_ROI_NAME:
        validate_face_roi_columns(frame)
    transform = evaluation_transform_from_checkpoint(checkpoint)

    saved_weights = checkpoint["config"].get("class_weights")
    weight_tensor = None
    if saved_weights is not None:
        weight_tensor = torch.tensor(
            [saved_weights["real"], saved_weights["fake"]],
            dtype=torch.float32,
            device=device,
        )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    dataset = EvaluationDataset(
        frame,
        data_root,
        transform,
        preprocessing_name,
        checkpoint["config"]["data_config"]["input_size"][1],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    metrics, labels, predictions, probabilities = evaluate_with_predictions(
        model, loader, frame, criterion, device
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    write_predictions(
        predictions_path, frame, labels, predictions, probabilities
    )
    result = {
        "domain_name": args.domain_name,
        "split": args.split,
        "images": len(frame),
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_lineage": manifest_lineage,
        "training_manifest_sha256": training_manifest_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_auc": checkpoint["best_val_auc"],
        "model": checkpoint["model_name"],
        "preprocessing_name": preprocessing_name,
        "preprocessing": checkpoint["config"].get(
            "preprocessing", checkpoint["config"]["data_config"]
        ),
        "loss_class_weights": saved_weights,
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "torchvision": torchvision.__version__,
        },
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
    }
    write_json(metrics_path, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
