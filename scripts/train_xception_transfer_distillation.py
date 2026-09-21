#!/usr/bin/env python3
"""Train one M7 student with uniform or transfer-aware expert distillation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
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
from torch.nn import functional as nn_functional
from torch.utils.data import DataLoader, Dataset
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
from xception_preprocessing import canonical_mixed_reencode_transforms


EXPERIMENT_FAMILY = "transfer_aware_distillation_pilot_v1"
CONDITION_ID = "M7"
STRATEGIES = ("uniform_kd", "transfer_kd")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
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
        raise ValueError("Unexpected distillation pilot protocol")
    if protocol.get("student_condition") != CONDITION_ID:
        raise ValueError("The pilot must remain restricted to M7")
    if protocol.get("training_seeds") != [42]:
        raise ValueError("The pilot must remain restricted to seed 42")
    return protocol


def run_name(strategy: str) -> str:
    return (
        f"xception_m7_fs_fr_efs_{strategy}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_"
        "pilot_v1_seed42"
    )


class DistillationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        transform,
        teacher_cache: pd.DataFrame,
        teacher_column: str,
    ) -> None:
        required = {
            "sample_id",
            "source_path",
            "method",
            teacher_column,
        }
        missing = required - set(teacher_cache.columns)
        if missing:
            raise ValueError(f"Teacher cache columns missing: {sorted(missing)}")
        if teacher_cache["sample_id"].duplicated().any():
            raise RuntimeError("Teacher cache contains duplicate sample IDs")
        expected_ids = set(frame.loc[frame["label"] == "fake", "sample_id"])
        cached_ids = set(teacher_cache["sample_id"])
        if cached_ids != expected_ids:
            raise RuntimeError(
                "Teacher cache and fake training sample IDs differ: "
                f"missing={len(expected_ids - cached_ids)}, "
                f"extra={len(cached_ids - expected_ids)}"
            )
        teacher = teacher_cache[
            ["sample_id", "source_path", "method", teacher_column]
        ].rename(
            columns={
                "source_path": "teacher_source_path",
                "method": "teacher_method",
                teacher_column: "teacher_fake_probability",
            }
        )
        merged = frame.merge(
            teacher, on="sample_id", how="left", validate="one_to_one"
        )
        fake = merged["label"] == "fake"
        if merged.loc[fake, "teacher_fake_probability"].isna().any():
            raise RuntimeError("Some fake training samples lack teacher targets")
        if merged.loc[~fake, "teacher_fake_probability"].notna().any():
            raise RuntimeError("Real samples must not receive KD targets")
        if not (
            merged.loc[fake, "source_path"]
            == merged.loc[fake, "teacher_source_path"]
        ).all():
            raise RuntimeError("Teacher cache source paths do not match")
        if not (
            merged.loc[fake, "method"] == merged.loc[fake, "teacher_method"]
        ).all():
            raise RuntimeError("Teacher cache methods do not match")
        self.frame = merged.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["source_path"]) as image:
            image = self.transform(image.convert("RGB"))
        has_teacher = row["label"] == "fake"
        target = float(row["teacher_fake_probability"]) if has_teacher else 0.0
        return {
            "image": image,
            "label": torch.tensor(LABEL_MAP[row["label"]], dtype=torch.long),
            "teacher_fake_probability": torch.tensor(target, dtype=torch.float32),
            "has_teacher": torch.tensor(has_teacher, dtype=torch.bool),
            "method": str(row["method"]),
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    distillation_weight: float,
    persistent_progress: bool,
) -> dict:
    model.train()
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_classification_loss = 0.0
    total_distillation_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_teacher_samples = 0
    progress = tqdm(
        loader,
        desc="Train transfer distillation",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        teacher_targets = batch["teacher_fake_probability"].to(
            device, non_blocking=True
        )
        teacher_mask = batch["has_teacher"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            classification_loss = criterion(logits, labels)
        fake_logit = (logits[:, 1] - logits[:, 0]).float()
        if teacher_mask.any():
            distillation_loss = nn_functional.binary_cross_entropy_with_logits(
                fake_logit[teacher_mask], teacher_targets[teacher_mask].float()
            )
        else:
            distillation_loss = fake_logit.sum() * 0.0
        loss = classification_loss.float() + (
            distillation_weight * distillation_loss
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        teacher_count = int(teacher_mask.sum().item())
        total_loss += float(loss.detach()) * batch_size
        total_classification_loss += (
            float(classification_loss.detach()) * batch_size
        )
        total_distillation_loss += (
            float(distillation_loss.detach()) * teacher_count
        )
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size
        total_teacher_samples += teacher_count
        progress.set_postfix(
            total=f"{float(loss.detach()):.4f}",
            ce=f"{float(classification_loss.detach()):.4f}",
            kd=f"{float(distillation_loss.detach()):.4f}",
        )
    return {
        "loss": total_loss / total_samples,
        "classification_loss": total_classification_loss / total_samples,
        "distillation_loss": total_distillation_loss / total_teacher_samples,
        "accuracy": total_correct / total_samples,
        "teacher_samples": total_teacher_samples,
    }


def write_history(path: Path, history: list[dict]) -> None:
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_classification_loss",
        "train_distillation_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_roc_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
                    "train_distillation_loss": row["train"][
                        "distillation_loss"
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
    args.teacher_cache = args.teacher_cache.resolve()
    args.output_root = args.output_root.resolve()
    args.conditions_config = args.conditions_config.resolve()
    args.pilot_config = args.pilot_config.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    for path in [
        args.data_root,
        args.manifest,
        args.teacher_cache,
        args.conditions_config,
        args.pilot_config,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = load_protocol(args.pilot_config)
    family_protocol, condition = load_condition(
        args.conditions_config, CONDITION_ID
    )
    validate_manifest(
        args.manifest,
        condition["seen_fake_methods"],
        family_protocol.get("budgets", FIXED_BUDGETS),
    )
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != protocol["manifest_sha256"]:
        raise ValueError("M7 training manifest hash mismatch")
    teacher_cache_hash = sha256_file(args.teacher_cache)
    teacher_summary_path = args.teacher_cache.with_name("cache_summary.json")
    if not teacher_summary_path.is_file():
        raise FileNotFoundError(teacher_summary_path)
    teacher_summary = json.loads(
        teacher_summary_path.read_text(encoding="utf-8")
    )
    if teacher_summary.get("protocol") != EXPERIMENT_FAMILY:
        raise ValueError("Teacher cache protocol mismatch")
    if teacher_summary.get("teacher_targets_sha256") != teacher_cache_hash:
        raise ValueError("Teacher cache file hash mismatch")
    if teacher_summary.get("student_manifest_sha256") != manifest_hash:
        raise ValueError("Teacher cache/student manifest mismatch")
    if teacher_summary.get("protected_unseen_used") or teacher_summary.get(
        "wilddeepfake_used"
    ):
        raise ValueError("Final evaluation data leaked into teacher cache")
    teacher_cache = pd.read_csv(args.teacher_cache)
    teacher_column = protocol["strategies"][args.strategy]["teacher_target"]

    output_dir = args.output_root / run_name(args.strategy)
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        last = output_dir / "last.pt"
        if not last.is_file():
            raise FileExistsError(f"Non-empty run has no last.pt: {output_dir}")
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if (
            int(checkpoint["epoch"]) >= args.epochs
            or int(checkpoint.get("epochs_without_improvement", 0))
            >= args.patience
        ):
            print("Training already complete:", output_dir, flush=True)
            return
        args.resume = last
        print("Auto-resuming:", last, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required")
    manifest, train_frame, val_frame = load_and_validate_manifest(
        args.manifest, args.data_root
    )
    class_counts, class_weights = calculate_class_weights(train_frame)
    model = timm.create_model(
        "xception", pretrained=args.resume is None, num_classes=len(LABEL_MAP)
    )
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    preprocessing = protocol["preprocessing"]
    train_transform, val_transform = canonical_mixed_reencode_transforms(
        data_config,
        canonical_size=preprocessing["canonical_size"],
        train_jpeg_qualities=preprocessing["train_jpeg_qualities"],
        validation_jpeg_quality=preprocessing["validation_jpeg_quality"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    train_dataset = DistillationDataset(
        train_frame,
        args.data_root,
        train_transform,
        teacher_cache,
        teacher_column,
    )
    val_dataset = FFPPDataset(val_frame, args.data_root, val_transform)
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    kd_weight = float(protocol["objective"]["distillation_weight"])
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "teacher_cache": str(args.teacher_cache),
        "output_root": str(args.output_root),
        "output_dir": str(output_dir),
        "resume": str(args.resume) if args.resume else None,
        "model": "xception",
        "split_protocol": EXPERIMENT_FAMILY,
        "experiment_family": EXPERIMENT_FAMILY,
        "condition_name": f"m7_fs_fr_efs_{args.strategy}",
        "preprocessing_name": preprocessing["name"],
        "preprocessing": preprocessing,
        "objective": protocol["objective"],
        "distillation_strategy": args.strategy,
        "teacher_target_column": teacher_column,
        "manifest_sha256": manifest_hash,
        "teacher_cache_sha256": teacher_cache_hash,
        "teacher_cache_summary_sha256": sha256_file(teacher_summary_path),
        "pilot_config_sha256": sha256_file(args.pilot_config),
        "label_map": LABEL_MAP,
        "class_counts": {
            "real": int(class_counts[0]),
            "fake": int(class_counts[1]),
        },
        "class_weights": {
            "real": float(class_weights[0]),
            "fake": float(class_weights[1]),
        },
        "train_transform": repr(train_transform),
        "validation_transform": repr(val_transform),
        "data_config": data_config,
        "train_images": len(train_frame),
        "val_images": len(val_frame),
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

    history = []
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
            "distillation_strategy": args.strategy,
            "manifest_sha256": manifest_hash,
            "teacher_cache_sha256": teacher_cache_hash,
            "seed": args.seed,
            "objective": protocol["objective"],
            "preprocessing": preprocessing,
        }
        for key, expected in checks.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch for {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_auc = checkpoint.get("best_val_auc", float("-inf"))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        start_epoch = int(checkpoint["epoch"]) + 1

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]))
    print("Validation distribution:", Counter(val_frame["label"]))
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            distillation_weight=kd_weight,
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            persistent_progress=args.persistent_progress,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
        )
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
        write_history(output_dir / "history.csv", history)
        print(
            f"Train total={train_metrics['loss']:.4f}, "
            f"CE={train_metrics['classification_loss']:.4f}, "
            f"KD={train_metrics['distillation_loss']:.4f}, "
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
    print("\nDistillation training complete.", flush=True)
    print("Best validation AUC:", best_auc, flush=True)
    print("Run directory:", output_dir, flush=True)


if __name__ == "__main__":
    main()
