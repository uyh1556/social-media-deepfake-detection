#!/usr/bin/env python3
"""Train the M7 paired-JPEG consistency pilot without changing baselines."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
from PIL import Image, features
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as functional
from tqdm.auto import tqdm

from train_baseline import (
    LABEL_MAP,
    FFPPDataset,
    atomic_torch_save,
    calculate_class_weights,
    checkpoint_payload,
    evaluate,
    json_ready,
    load_and_validate_manifest,
    seed_worker,
    set_seed,
    sha256_file,
    write_json,
)
from train_xception_family_coverage import (
    FIXED_BUDGETS,
    load_condition,
    validate_manifest,
)
from xception_preprocessing import (
    JpegRoundTrip,
    Letterbox,
    canonical_reencode_transform,
    interpolation_mode,
)


EXPERIMENT_FAMILY = "family_coverage_jpeg_consistency_pilot_v1"
PREPROCESSING_NAME = "canonical256_jpegpair_consistency_letterbox299"
CONDITION_ID = "M7"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=(
            project_root
            / "configs/family_coverage_jpeg_consistency_pilot_v1/"
            "protocol.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, choices=[42], default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--persistent-progress", action="store_true")
    return parser.parse_args()


def load_protocol(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != EXPERIMENT_FAMILY:
        raise ValueError("Unexpected JPEG-consistency pilot protocol.")
    if protocol.get("models") != [CONDITION_ID]:
        raise ValueError("The pilot must be restricted to M7.")
    if protocol.get("training_seeds") != [42]:
        raise ValueError("The pilot must be restricted to seed 42.")
    preprocessing = protocol["preprocessing"]
    expected = {
        "name": PREPROCESSING_NAME,
        "canonical_size": 256,
        "train_jpeg_qualities": [75, 80, 85, 90, 95],
        "paired_quality_sampling": (
            "uniform ordered pair without replacement"
        ),
        "validation_jpeg_quality": 95,
        "jpeg_subsampling": 2,
        "jpeg_optimize": False,
        "jpeg_progressive": False,
        "model_input_size": 299,
    }
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(
                f"Pilot preprocessing mismatch for {key}: "
                f"expected={value}, actual={preprocessing.get(key)}"
            )
    expected_objective = {
        "classification": (
            "cross-entropy on one uniformly sampled anchor JPEG view"
        ),
        "second_view_role": (
            "consistency regularization only, without an additional "
            "supervised cross-entropy term"
        ),
        "consistency": (
            "Jensen-Shannon divergence between the two "
            "class-probability distributions"
        ),
        "consistency_weight": 0.1,
    }
    if protocol.get("objective") != expected_objective:
        raise ValueError("Unexpected frozen pilot objective.")
    return protocol


def run_name() -> str:
    return (
        "xception_m7_fs_fr_efs_"
        "canonical256_jpegpair_consistency_js0p1_letterbox299_"
        "pilot_v1_seed42"
    )


class PairedJpegDataset(Dataset):
    """Return two JPEG qualities of one shared geometric image view."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        data_config: dict,
        *,
        canonical_size: int,
        qualities: list[int],
        jpeg_subsampling: int,
        jpeg_optimize: bool,
        jpeg_progressive: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.qualities = tuple(int(value) for value in qualities)
        if len(self.qualities) < 2:
            raise ValueError("Paired training requires at least two qualities.")
        interpolation = interpolation_mode(data_config["interpolation"])
        self.canonical = Letterbox(
            size=canonical_size,
            interpolation=interpolation,
        )
        self.model_letterbox = Letterbox(
            size=data_config["input_size"][1],
            interpolation=interpolation,
        )
        self.jpeg_subsampling = int(jpeg_subsampling)
        self.jpeg_optimize = bool(jpeg_optimize)
        self.jpeg_progressive = bool(jpeg_progressive)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=data_config["mean"],
            std=data_config["std"],
        )

    def __len__(self) -> int:
        return len(self.frame)

    def make_view(self, canonical: Image.Image, quality: int) -> torch.Tensor:
        image = JpegRoundTrip(
            quality=quality,
            subsampling=self.jpeg_subsampling,
            optimize=self.jpeg_optimize,
            progressive=self.jpeg_progressive,
        )(canonical)
        image = self.model_letterbox(image)
        return self.normalize(self.to_tensor(image))

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = self.data_root / row["source_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if random.random() < 0.5:
                image = functional.hflip(image)
            canonical = self.canonical(image)
            quality_a, quality_b = random.sample(self.qualities, 2)
            view_a = self.make_view(canonical, quality_a)
            view_b = self.make_view(canonical, quality_b)
        return {
            "image_a": view_a,
            "image_b": view_b,
            "quality_a": quality_a,
            "quality_b": quality_b,
            "label": torch.tensor(
                LABEL_MAP[row["label"]], dtype=torch.long
            ),
        }


def jensen_shannon_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """Symmetric JS divergence in probability space, in natural-log units."""
    probabilities_a = torch.softmax(logits_a.float(), dim=1).clamp_min(1e-7)
    probabilities_b = torch.softmax(logits_b.float(), dim=1).clamp_min(1e-7)
    midpoint = (0.5 * (probabilities_a + probabilities_b)).clamp_min(1e-7)
    divergence_a = (
        probabilities_a * (probabilities_a.log() - midpoint.log())
    ).sum(dim=1)
    divergence_b = (
        probabilities_b * (probabilities_b.log() - midpoint.log())
    ).sum(dim=1)
    return 0.5 * (divergence_a + divergence_b).mean()


def train_one_epoch_paired(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    consistency_weight: float,
    accumulation_steps: int,
    persistent_progress: bool,
) -> dict:
    model.train()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_classification_loss = 0.0
    total_consistency_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        desc="Train paired JPEG consistency",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for step, batch in enumerate(progress, start=1):
        view_a = batch["image_a"].to(device, non_blocking=True)
        view_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        paired_images = torch.cat([view_a, view_b], dim=0)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            paired_logits = model(paired_images)
            logits_a, logits_b = paired_logits.chunk(2, dim=0)
            classification_loss = criterion(logits_a, labels)
        consistency_loss = jensen_shannon_divergence(logits_a, logits_b)
        loss = classification_loss.float() + (
            consistency_weight * consistency_loss
        )
        scaler.scale(loss / accumulation_steps).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_classification_loss += (
            float(classification_loss.detach()) * batch_size
        )
        total_consistency_loss += (
            float(consistency_loss.detach()) * batch_size
        )
        total_correct += (logits_a.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(
            total=f"{float(loss.detach()):.4f}",
            ce=f"{float(classification_loss.detach()):.4f}",
            js=f"{float(consistency_loss.detach()):.4f}",
        )
    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "consistency_loss": total_consistency_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def write_consistency_history(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_classification_loss",
        "train_consistency_loss",
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
                    "train_classification_loss": row["train"][
                        "classification_loss"
                    ],
                    "train_consistency_loss": row["train"][
                        "consistency_loss"
                    ],
                    "train_accuracy": row["train"]["accuracy"],
                    "val_loss": row["val"]["loss"],
                    "val_accuracy": row["val"]["accuracy"],
                    "val_precision": row["val"]["precision"],
                    "val_recall": row["val"]["recall"],
                    "val_f1": row["val"]["f1"],
                    "val_roc_auc": row["val"]["roc_auc"],
                }
            )


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    args.conditions_config = args.conditions_config.resolve()
    args.pilot_config = args.pilot_config.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if args.batch_size != 8 or args.gradient_accumulation_steps != 2:
        raise ValueError(
            "The frozen pilot requires batch-size 8 and accumulation 2."
        )
    protocol = load_protocol(args.pilot_config)
    family_protocol, condition = load_condition(
        args.conditions_config, CONDITION_ID
    )
    validate_manifest(
        args.manifest,
        condition["seen_fake_methods"],
        family_protocol.get("budgets", FIXED_BUDGETS),
    )
    actual_manifest_hash = sha256_file(args.manifest)
    expected_manifest_hash = protocol["manifest_sha256"][CONDITION_ID]
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "Training manifest hash mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    output_dir = args.output_root / run_name()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        last_checkpoint = output_dir / "last.pt"
        if last_checkpoint.is_file():
            args.resume = last_checkpoint
            print(f"Auto-resuming pilot: {last_checkpoint}", flush=True)
        else:
            raise FileExistsError(
                f"Non-empty pilot directory has no last.pt: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for pilot training.")
    manifest, train_frame, val_frame = load_and_validate_manifest(
        args.manifest, args.data_root
    )
    class_counts, class_weights = calculate_class_weights(train_frame)
    weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    model = timm.create_model(
        "xception",
        pretrained=args.resume is None,
        num_classes=len(LABEL_MAP),
    )
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    if tuple(data_config["input_size"]) != (3, 299, 299):
        raise RuntimeError(
            f"Unexpected Xception input: {data_config['input_size']}"
        )
    preprocessing = protocol["preprocessing"]
    train_dataset = PairedJpegDataset(
        train_frame,
        args.data_root,
        data_config,
        canonical_size=preprocessing["canonical_size"],
        qualities=preprocessing["train_jpeg_qualities"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    validation_transform = canonical_reencode_transform(
        data_config,
        canonical_size=preprocessing["canonical_size"],
        jpeg_quality=preprocessing["validation_jpeg_quality"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    val_dataset = FFPPDataset(
        val_frame, args.data_root, validation_transform
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        **common,
    )
    if len(train_loader) % args.gradient_accumulation_steps:
        raise RuntimeError(
            "Frozen pilot expects complete gradient-accumulation groups."
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    consistency_weight = float(
        protocol["objective"]["consistency_weight"]
    )
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "output_dir": str(output_dir),
        "resume": str(args.resume) if args.resume else None,
        "model": "xception",
        "split_protocol": EXPERIMENT_FAMILY,
        "experiment_family": EXPERIMENT_FAMILY,
        "condition_name": "m7_fs_fr_efs_jpeg_consistency",
        "preprocessing_name": PREPROCESSING_NAME,
        "preprocessing": preprocessing,
        "objective": protocol["objective"],
        "optimization": protocol["optimization"],
        "manifest_sha256": actual_manifest_hash,
        "label_map": LABEL_MAP,
        "class_counts": {
            "real": int(class_counts[LABEL_MAP["real"]]),
            "fake": int(class_counts[LABEL_MAP["fake"]]),
        },
        "class_weights": {
            "real": float(class_weights[LABEL_MAP["real"]]),
            "fake": float(class_weights[LABEL_MAP["fake"]]),
        },
        "train_transform": (
            "shared horizontal flip p=0.5 -> Letterbox256 -> two "
            "distinct random JPEG qualities -> Letterbox299 -> "
            "ToTensor -> Normalize; supervised CE on anchor view only"
        ),
        "validation_transform": repr(validation_transform),
        "data_config": data_config,
        "train_images": len(train_frame),
        "train_views_per_image": 2,
        "val_images": len(val_frame),
        "test_images_held_out": int((manifest["split"] == "test").sum()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "pillow": PIL.__version__,
            "libjpeg": features.version_codec("jpg"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(output_dir / "config.json", config)

    history: list[dict] = []
    best_auc = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        saved = checkpoint["config"]
        checks = {
            "experiment_family": EXPERIMENT_FAMILY,
            "manifest_sha256": actual_manifest_hash,
            "seed": args.seed,
            "preprocessing": preprocessing,
            "objective": protocol["objective"],
            "optimization": protocol["optimization"],
        }
        for key, expected in checks.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch for {key}.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_auc = checkpoint.get("best_val_auc", float("-inf"))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        if epochs_without_improvement >= args.patience:
            print(
                f"Pilot already stopped at epoch {checkpoint['epoch']}.",
                flush=True,
            )
            return

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]))
    print("Validation distribution:", Counter(val_frame["label"]))
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch_paired(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            consistency_weight=consistency_weight,
            accumulation_steps=args.gradient_accumulation_steps,
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            persistent_progress=args.persistent_progress,
        )
        epoch_result = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        write_consistency_history(output_dir / "history.csv", history)
        print(
            f"Train total={train_metrics['loss']:.4f}, "
            f"CE={train_metrics['classification_loss']:.4f}, "
            f"JS={train_metrics['consistency_loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}",
            flush=True,
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        print("Confusion matrix:", val_metrics["confusion_matrix"])
        print(
            f"Early stopping: {epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break
    print("\nPilot training complete.", flush=True)
    print("Best validation AUC:", best_auc, flush=True)
    print("Run directory:", output_dir, flush=True)


if __name__ == "__main__":
    main()
