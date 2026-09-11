import argparse
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


LABEL_MAP = {"real": 0, "fake": 1}
REQUIRED_COLUMNS = {
    "split",
    "label",
    "method",
    "group_id",
    "video_id",
    "source_ids",
    "source_path",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a reproducible FF++ source-domain baseline."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-protocol",
        required=True,
        help="Recorded protocol name, e.g. custom_source_aware_v1.",
    )
    parser.add_argument("--model", default="xception")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint created by this script.",
    )
    parser.add_argument(
        "--unweighted-loss",
        action="store_true",
        help="Disable inverse-frequency class weights.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(data), file, ensure_ascii=False, indent=2)


def write_history_csv(path, history):
    if not history:
        return

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "learning_rate": row["learning_rate"],
                    "train_loss": row["train"]["loss"],
                    "train_accuracy": row["train"]["accuracy"],
                    "val_loss": row["val"]["loss"],
                    "val_accuracy": row["val"]["accuracy"],
                    "val_precision": row["val"]["precision"],
                    "val_recall": row["val"]["recall"],
                    "val_f1": row["val"]["f1"],
                    "val_roc_auc": row["val"]["roc_auc"],
                }
            )


def load_and_validate_manifest(manifest_path, data_root):
    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "group_id": str,
            "video_id": str,
            "source_ids": str,
            "source_path": str,
        },
    )

    missing_columns = REQUIRED_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing_columns)}"
        )

    unexpected_splits = set(manifest["split"].unique()) - {
        "train",
        "val",
        "test",
    }
    if unexpected_splits:
        raise ValueError(f"Unexpected split values: {sorted(unexpected_splits)}")

    unexpected_labels = set(manifest["label"].unique()) - set(LABEL_MAP)
    if unexpected_labels:
        raise ValueError(f"Unexpected labels: {sorted(unexpected_labels)}")

    group_leakage = (
        manifest.groupby("group_id")["split"].nunique().gt(1).sum()
    )
    video_leakage = (
        manifest.groupby(["method", "video_id"])["split"]
        .nunique()
        .gt(1)
        .sum()
    )
    if group_leakage or video_leakage:
        raise RuntimeError(
            "Split leakage detected: "
            f"groups={group_leakage}, videos={video_leakage}"
        )

    missing_paths = []
    for relative_path in manifest["source_path"]:
        if not (data_root / relative_path).is_file():
            missing_paths.append(relative_path)
            if len(missing_paths) == 10:
                break
    if missing_paths:
        raise FileNotFoundError(
            "Manifest images are missing. First entries: "
            + ", ".join(missing_paths)
        )

    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Train and validation splits must both be non-empty.")

    return manifest, train, val


class FFPPDataset(Dataset):
    def __init__(self, frame, data_root, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self.data_root / row["source_path"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(LABEL_MAP[row["label"]], dtype=torch.long),
            "method": row["method"],
            "video_id": row["video_id"],
            "group_id": row["group_id"],
        }


def make_loaders(train_frame, val_frame, data_root, data_config, args):
    evaluation_transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    # Baseline training uses only a horizontal flip in addition to the
    # pretrained model's deterministic resize/crop/normalization. Compression,
    # blur, noise, and platform-like transforms belong to later experiments.
    training_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), evaluation_transform]
    )

    train_dataset = FFPPDataset(train_frame, data_root, training_transform)
    val_dataset = FFPPDataset(val_frame, data_root, evaluation_transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def calculate_class_weights(train_frame):
    numeric_labels = train_frame["label"].map(LABEL_MAP).to_numpy()
    counts = np.bincount(numeric_labels, minlength=len(LABEL_MAP))
    if np.any(counts == 0):
        raise ValueError(f"A training class is empty: counts={counts.tolist()}")
    weights = len(numeric_labels) / (len(LABEL_MAP) * counts)
    return counts, weights


def classification_metrics(labels, predictions, probabilities):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def manipulation_metrics(labels, predictions, probabilities, methods):
    results = {}
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    probabilities_array = np.asarray(probabilities)
    methods_array = np.asarray(methods)

    for fake_method in ("Deepfakes", "Face2Face"):
        mask = np.isin(methods_array, ["original", fake_method])
        subset_labels = labels_array[mask]
        if len(np.unique(subset_labels)) < 2:
            continue
        results[f"original_vs_{fake_method}"] = classification_metrics(
            subset_labels,
            predictions_array[mask],
            probabilities_array[mask],
        )
    return results


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_samples = 0
    labels_all = []
    predictions_all = []
    probabilities_all = []
    methods_all = []

    progress = tqdm(loader, desc="Validation", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.float().cpu().tolist())
        methods_all.extend(batch["method"])

    metrics = classification_metrics(
        labels_all, predictions_all, probabilities_all
    )
    metrics["loss"] = total_loss / total_samples
    metrics["by_manipulation"] = manipulation_metrics(
        labels_all,
        predictions_all,
        probabilities_all,
        methods_all,
    )
    return metrics


def atomic_torch_save(data, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(destination)


def checkpoint_payload(
    epoch,
    model,
    optimizer,
    scaler,
    train_metrics,
    val_metrics,
    history,
    best_auc,
    epochs_without_improvement,
    config,
):
    return {
        "epoch": epoch,
        "model_name": config["model"],
        "num_classes": len(LABEL_MAP),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "best_val_auc": best_auc,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
        "label_map": LABEL_MAP,
    }


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Choose a new run directory or pass --resume."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for full baseline training.")

    manifest, train_frame, val_frame = load_and_validate_manifest(
        args.manifest, args.data_root
    )
    class_counts, class_weights = calculate_class_weights(train_frame)
    weight_tensor = None
    if not args.unweighted_loss:
        weight_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )

    model = timm.create_model(
        args.model,
        pretrained=args.resume is None,
        num_classes=len(LABEL_MAP),
    )
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    train_loader, val_loader = make_loaders(
        train_frame, val_frame, args.data_root, data_config, args
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "resume": str(args.resume) if args.resume else None,
        "manifest_sha256": sha256_file(args.manifest),
        "label_map": LABEL_MAP,
        "class_counts": {
            "real": int(class_counts[LABEL_MAP["real"]]),
            "fake": int(class_counts[LABEL_MAP["fake"]]),
        },
        "class_weights": (
            None
            if args.unweighted_loss
            else {
                "real": float(class_weights[LABEL_MAP["real"]]),
                "fake": float(class_weights[LABEL_MAP["fake"]]),
            }
        ),
        "train_transform": (
            "model eval resize/crop/normalization + horizontal flip p=0.5"
        ),
        "validation_transform": "model eval resize/crop/normalization",
        "data_config": data_config,
        "train_images": len(train_frame),
        "val_images": len(val_frame),
        "test_images_held_out": int((manifest["split"] == "test").sum()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(args.output_dir / "config.json", config)

    history = []
    best_auc = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                "Resume model mismatch: "
                f"{checkpoint['model_name']} != {args.model}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_auc = checkpoint.get("best_val_auc", float("-inf"))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        start_epoch = checkpoint["epoch"] + 1

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]))
    print("Validation distribution:", Counter(val_frame["label"]))

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        learning_rate = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_result)

        improved = val_metrics["roc_auc"] > best_auc + args.min_delta
        if improved:
            best_auc = val_metrics["roc_auc"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            epoch,
            model,
            optimizer,
            scaler,
            train_metrics,
            val_metrics,
            history,
            best_auc,
            epochs_without_improvement,
            config,
        )
        atomic_torch_save(payload, args.output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, args.output_dir / "best.pt")

        write_json(args.output_dir / "history.json", history)
        write_history_csv(args.output_dir / "history.csv", history)

        print(
            f"Train loss={train_metrics['loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}"
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['roc_auc']:.4f}"
        )
        print("Confusion matrix:", val_metrics["confusion_matrix"])
        print(
            "Early stopping: "
            f"{epochs_without_improvement}/{args.patience}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    print("\nTraining complete.")
    print("Best validation AUC:", best_auc)
    print("Run directory:", args.output_dir)


if __name__ == "__main__":
    main()
