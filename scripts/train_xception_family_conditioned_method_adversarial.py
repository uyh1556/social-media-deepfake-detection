#!/usr/bin/env python3
"""Train the M7 family-conditioned method-adversarial pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from PIL import features
from torch import nn
from torch.utils.data import DataLoader
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
from train_xception_method_quality_adaptive import MethodQualityDataset
from xception_preprocessing import canonical_reencode_transform


EXPERIMENT_FAMILY = "family_conditioned_method_adversarial_pilot_v1"
PREPROCESSING_NAME = (
    "canonical256_jpegmixed_familyconditioned_"
    "methodadversarial_letterbox299"
)
CONDITION_ID = "M7"
CONDITION_NAME = "m7_fs_fr_efs_family_conditioned_method_adversarial"
FAMILY_METHODS = {
    "FS": ("SimSwap", "BlendFace"),
    "FR": ("Wav2Lip", "FOMM"),
    "EFS": ("StyleGAN3", "DiT"),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
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
        raise ValueError("Unexpected method-adversarial protocol.")
    if protocol.get("models") != [CONDITION_ID]:
        raise ValueError("The pilot must be restricted to M7.")
    if protocol.get("training_seeds") != [42]:
        raise ValueError("The pilot must be restricted to seed 42.")

    preprocessing = protocol["preprocessing"]
    expected_preprocessing = {
        "name": PREPROCESSING_NAME,
        "canonical_size": 256,
        "train_jpeg_qualities": [75, 80, 85, 90, 95],
        "train_jpeg_sampling": "uniform",
        "validation_jpeg_quality": 95,
        "jpeg_subsampling": 2,
        "jpeg_optimize": False,
        "jpeg_progressive": False,
        "model_input_size": 299,
    }
    for key, value in expected_preprocessing.items():
        if preprocessing.get(key) != value:
            raise ValueError(
                f"Preprocessing mismatch for {key}: "
                f"expected={value}, actual={preprocessing.get(key)}"
            )

    objective = protocol["objective"]
    actual_families = {
        family: tuple(methods)
        for family, methods in objective.get("family_methods", {}).items()
    }
    if actual_families != FAMILY_METHODS:
        raise ValueError("Unexpected family-method mapping.")
    expected_objective = {
        "gradient_reversal_gamma": 10.0,
        "maximum_gradient_reversal_strength": 0.1,
        "discriminator_hidden_features": 256,
        "discriminator_dropout": 0.2,
    }
    for key, value in expected_objective.items():
        if objective.get(key) != value:
            raise ValueError(
                f"Objective mismatch for {key}: "
                f"expected={value}, actual={objective.get(key)}"
            )
    optimization = protocol["optimization"]
    if optimization.get("images_per_loader_batch") != 16:
        raise ValueError("The pilot requires loader batch size 16.")
    if optimization.get("views_per_image") != 1:
        raise ValueError("The pilot requires one image view.")
    return protocol


def run_name() -> str:
    return (
        "xception_m7_fs_fr_efs_canonical256_jpegmix_"
        "familyconditioned_methodadversarial_"
        "letterbox299_pilot_v1_seed42"
    )


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        return -ctx.strength * gradients, None


def gradient_reverse(inputs: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(inputs, strength)


class FamilyMethodDiscriminators(nn.Module):
    """One binary seen-method discriminator for each forgery family."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                family: nn.Sequential(
                    nn.Linear(input_features, hidden_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_features, 2),
                )
                for family in FAMILY_METHODS
            }
        )

    def forward(
        self,
        features: torch.Tensor,
        methods: list[str],
        strength: float,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        output = {}
        for family, family_methods in FAMILY_METHODS.items():
            indices = [
                index
                for index, method in enumerate(methods)
                if method in family_methods
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(
                indices, dtype=torch.long, device=features.device
            )
            selected = features.index_select(0, index_tensor)
            targets = torch.tensor(
                [family_methods.index(methods[index]) for index in indices],
                dtype=torch.long,
                device=features.device,
            )
            logits = self.heads[family](
                gradient_reverse(selected, strength)
            )
            output[family] = (logits, targets)
        return output


def grl_strength(
    global_step: int,
    total_steps: int,
    *,
    gamma: float,
    maximum: float,
) -> float:
    if total_steps <= 1:
        return float(maximum)
    progress = min(max(global_step / (total_steps - 1), 0.0), 1.0)
    return float(maximum * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0))


def forward_with_features(
    model: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_map = model.forward_features(images)
    pooled = model.forward_head(feature_map, pre_logits=True)
    logits = model.get_classifier()(pooled)
    return logits, pooled


def train_one_epoch(
    model: nn.Module,
    discriminators: FamilyMethodDiscriminators,
    loader: DataLoader,
    classification_criterion: nn.Module,
    method_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    epochs: int,
    gamma: float,
    maximum_grl_strength: float,
    persistent_progress: bool,
) -> dict:
    model.train()
    discriminators.train()
    use_amp = device.type == "cuda"
    total_steps = epochs * len(loader)
    epoch_start_step = (epoch - 1) * len(loader)
    totals = {
        "classification_loss": 0.0,
        "method_loss": 0.0,
        "correct": 0,
        "samples": 0,
    }
    family_loss_sum = Counter()
    family_correct = Counter()
    family_samples = Counter()
    progress = tqdm(
        loader,
        desc="Train family-conditioned adversarial",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    last_strength = 0.0
    for batch_index, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        methods = [str(value) for value in batch["method"]]
        last_strength = grl_strength(
            epoch_start_step + batch_index,
            total_steps,
            gamma=gamma,
            maximum=maximum_grl_strength,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits, pooled = forward_with_features(model, images)
            classification_loss = classification_criterion(logits, labels)
            family_outputs = discriminators(
                pooled, methods, last_strength
            )
            method_losses = {
                family: method_criterion(method_logits, targets)
                for family, (method_logits, targets) in family_outputs.items()
            }
            if method_losses:
                method_loss = torch.stack(list(method_losses.values())).mean()
            else:
                # A fully Real random batch is rare but valid. Keep the
                # binary update and give the method heads no update.
                method_loss = pooled.sum() * 0.0
            loss = classification_loss.float() + method_loss.float()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        totals["classification_loss"] += (
            float(classification_loss.detach()) * batch_size
        )
        totals["method_loss"] += float(method_loss.detach()) * batch_size
        totals["correct"] += (logits.argmax(dim=1) == labels).sum().item()
        totals["samples"] += batch_size
        for family, (method_logits, targets) in family_outputs.items():
            count = targets.size(0)
            family_loss_sum[family] += (
                float(method_losses[family].detach()) * count
            )
            family_correct[family] += (
                (method_logits.argmax(dim=1) == targets).sum().item()
            )
            family_samples[family] += count
        progress.set_postfix(
            cls=f"{float(classification_loss.detach()):.4f}",
            method=f"{float(method_loss.detach()):.4f}",
            grl=f"{last_strength:.3f}",
        )

    if totals["samples"] != 39_600:
        raise RuntimeError(f"Unexpected epoch sample count: {totals['samples']}")
    family_metrics = {
        family: {
            "loss": family_loss_sum[family] / family_samples[family],
            "accuracy": family_correct[family] / family_samples[family],
            "samples": int(family_samples[family]),
        }
        for family in FAMILY_METHODS
    }
    return {
        "loss": (
            totals["classification_loss"] + totals["method_loss"]
        ) / totals["samples"],
        "classification_loss": (
            totals["classification_loss"] / totals["samples"]
        ),
        "method_loss": totals["method_loss"] / totals["samples"],
        "accuracy": totals["correct"] / totals["samples"],
        "grl_strength_end": last_strength,
        "family_method_metrics": family_metrics,
    }


def write_history_csv(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_classification_loss",
        "train_method_loss",
        "train_accuracy",
        "grl_strength_end",
        "fs_method_accuracy",
        "fr_method_accuracy",
        "efs_method_accuracy",
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
                    "train_method_loss": row["train"]["method_loss"],
                    "train_accuracy": row["train"]["accuracy"],
                    "grl_strength_end": row["train"][
                        "grl_strength_end"
                    ],
                    "fs_method_accuracy": row["train"][
                        "family_method_metrics"
                    ]["FS"]["accuracy"],
                    "fr_method_accuracy": row["train"][
                        "family_method_metrics"
                    ]["FR"]["accuracy"],
                    "efs_method_accuracy": row["train"][
                        "family_method_metrics"
                    ]["EFS"]["accuracy"],
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
    protocol = load_protocol(args.pilot_config)
    if args.batch_size != protocol["optimization"]["images_per_loader_batch"]:
        raise ValueError("The frozen pilot requires batch size 16.")

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
    fake_train = train_frame[train_frame["label"] == "fake"]
    expected_method_counts = {
        method: 3_300
        for methods in FAMILY_METHODS.values()
        for method in methods
    }
    actual_method_counts = fake_train["method"].value_counts().to_dict()
    if actual_method_counts != expected_method_counts:
        raise ValueError(
            "Unexpected M7 method counts: "
            f"expected={expected_method_counts}, actual={actual_method_counts}"
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
    objective = protocol["objective"]
    discriminators = FamilyMethodDiscriminators(
        input_features=int(model.num_features),
        hidden_features=int(objective["discriminator_hidden_features"]),
        dropout=float(objective["discriminator_dropout"]),
    )

    preprocessing = protocol["preprocessing"]
    train_dataset = MethodQualityDataset(
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
    val_dataset = FFPPDataset(val_frame, args.data_root, validation_transform)
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
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )

    model = model.to(device)
    discriminators = discriminators.to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    method_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminators.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

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
        "condition_name": CONDITION_NAME,
        "preprocessing_name": PREPROCESSING_NAME,
        "preprocessing": preprocessing,
        "objective": objective,
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
            "horizontal flip p=0.5 -> Letterbox256 -> one uniform random "
            "JPEG Q75/Q80/Q85/Q90/Q95 -> Letterbox299 -> Normalize; "
            "family-conditioned gradient-reversal method heads"
        ),
        "validation_transform": repr(validation_transform),
        "data_config": data_config,
        "train_images": len(train_frame),
        "train_views_per_image": 1,
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
            "objective": objective,
            "optimization": protocol["optimization"],
        }
        for key, expected in checks.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch for {key}.")
        model.load_state_dict(checkpoint["model_state_dict"])
        discriminators.load_state_dict(
            checkpoint["method_discriminator_state_dict"]
        )
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
    print("Train methods:", Counter(train_frame["method"]))
    print("Validation distribution:", Counter(val_frame["label"]))
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            discriminators,
            train_loader,
            classification_criterion,
            method_criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            epochs=args.epochs,
            gamma=float(objective["gradient_reversal_gamma"]),
            maximum_grl_strength=float(
                objective["maximum_gradient_reversal_strength"]
            ),
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            classification_criterion,
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
        payload["method_discriminator_state_dict"] = (
            discriminators.state_dict()
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)

        family_accuracy = train_metrics["family_method_metrics"]
        print(
            f"Train binary loss={train_metrics['classification_loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}, "
            f"method loss={train_metrics['method_loss']:.4f}"
        )
        print(
            "Method-head accuracy: "
            + ", ".join(
                f"{family}={family_accuracy[family]['accuracy']:.4f}"
                for family in FAMILY_METHODS
            )
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, "
            f"AUC={val_metrics['roc_auc']:.4f}"
        )
        print("Confusion matrix:", val_metrics["confusion_matrix"])
        print(
            f"Early stopping: {epochs_without_improvement}/"
            f"{args.patience}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    print("\nTraining complete.")
    print("Best validation AUC:", best_auc)
    print("Run directory:", output_dir)


if __name__ == "__main__":
    main()
