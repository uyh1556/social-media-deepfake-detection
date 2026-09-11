import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    train_one_epoch,
    write_history_csv,
    write_json,
)
from xception_preprocessing import (
    LETTERBOX_FILL_RGB,
    LETTERBOX_NAME,
    letterbox_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a separate full-frame letterbox Xception ablation without "
            "modifying the Standard Xception baseline."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-protocol", required=True)
    parser.add_argument("--model", default="xception")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--unweighted-loss", action="store_true")
    return parser.parse_args()


def make_loaders(train_frame, val_frame, data_root, data_config, args):
    training_transform, evaluation_transform = letterbox_transforms(
        data_config
    )
    train_dataset = FFPPDataset(
        train_frame, data_root, training_transform
    )
    val_dataset = FFPPDataset(
        val_frame, data_root, evaluation_transform
    )
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
        train_dataset, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, training_transform, evaluation_transform


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if args.model not in {"xception", "legacy_xception"}:
        raise ValueError("This ablation is restricted to Xception.")
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
        raise RuntimeError("A CUDA GPU is required for full training.")
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
    if tuple(data_config["input_size"]) != (3, 299, 299):
        raise RuntimeError(
            f"Unexpected Xception input: {data_config['input_size']}"
        )
    train_loader, val_loader, train_transform, val_transform = make_loaders(
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
        "experiment_family": "xception_preprocessing_ablation_v1",
        "preprocessing_name": LETTERBOX_NAME,
        "preprocessing": {
            "policy": "preserve full frame and aspect ratio",
            "resize": "fit long edge inside 299 with bicubic antialiasing",
            "padding": "symmetric constant padding to 299x299",
            "padding_fill_rgb": LETTERBOX_FILL_RGB,
            "distortion": "none",
        },
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
        "train_transform": repr(train_transform),
        "validation_transform": repr(val_transform),
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
            raise ValueError("Resume model does not match.")
        if checkpoint["config"].get("preprocessing_name") != LETTERBOX_NAME:
            raise ValueError("Resume checkpoint is not the letterbox experiment.")
        if checkpoint["config"].get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("Resume manifest does not match.")
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
