import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
import torch.nn.functional as nn_functional
from PIL import features
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
    CONTROLLED_REENCODE_NAME,
    LETTERBOX_FILL_RGB,
    LETTERBOX_NAME,
    MIXED_JPEG_REENCODE_NAME,
    RandomGaussianNoise,
    canonical_mixed_reencode_transforms,
    canonical_reencode_transforms,
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
    parser.add_argument(
        "--experiment-family",
        default="xception_preprocessing_ablation_v1",
    )
    parser.add_argument("--condition-name", default=None)
    parser.add_argument(
        "--persistent-progress",
        action="store_true",
        help=(
            "Keep the Train/Validation tqdm bars visible in notebook output. "
            "Recommended for Colab."
        ),
    )
    parser.add_argument(
        "--canonical-size",
        type=int,
        default=None,
        help=(
            "Enable the controlled re-encoding pipeline by first "
            "letterboxing every image to this common size."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="Fixed JPEG quality for controlled in-memory re-encoding.",
    )
    parser.add_argument(
        "--train-jpeg-qualities",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Uniform JPEG quality choices used only during training. "
            "Validation remains fixed at --jpeg-quality."
        ),
    )
    parser.add_argument("--jpeg-subsampling", type=int, default=2)
    parser.add_argument("--jpeg-optimize", action="store_true")
    parser.add_argument("--jpeg-progressive", action="store_true")
    parser.add_argument("--train-noise-probability", type=float, default=0.0)
    parser.add_argument("--train-noise-max-std", type=float, default=2.0)
    parser.add_argument(
        "--method-loss-weights-json", type=Path, default=None,
        help="Optional JSON mapping fake method names to positive loss weights.",
    )
    parser.add_argument("--adaptive-strategy", choices=["uniform", "difficulty", "complementarity"], default=None)
    parser.add_argument("--adaptive-protocol-config", type=Path, default=None)
    parser.add_argument("--adaptive-validation-roles", type=Path, default=None)
    return parser.parse_args()


def train_one_epoch_method_weighted(
    model, loader, class_weights, method_weights, optimizer, scaler, device,
    *, persistent_progress=False,
):
    from tqdm.auto import tqdm
    model.train()
    total_loss = total_correct = total_samples = 0
    progress = tqdm(loader, desc="Train", leave=persistent_progress, dynamic_ncols=True)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        sample_weights = torch.tensor(
            [method_weights.get(method, 1.0) for method in batch["method"]],
            dtype=torch.float32, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            logits = model(images)
            losses = nn_functional.cross_entropy(
                logits, labels, weight=class_weights, reduction="none"
            )
            loss = (losses * sample_weights).sum() / sample_weights.sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        size = labels.size(0)
        total_loss += loss.item() * size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += size
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def make_loaders(train_frame, val_frame, data_root, data_config, args):
    if args.canonical_size is None:
        training_transform, evaluation_transform = letterbox_transforms(
            data_config
        )
    elif args.train_jpeg_qualities is None:
        training_transform, evaluation_transform = (
            canonical_reencode_transforms(
                data_config,
                canonical_size=args.canonical_size,
                jpeg_quality=args.jpeg_quality,
                jpeg_subsampling=args.jpeg_subsampling,
                jpeg_optimize=args.jpeg_optimize,
                jpeg_progressive=args.jpeg_progressive,
            )
        )
    else:
        training_transform, evaluation_transform = (
            canonical_mixed_reencode_transforms(
                data_config,
                canonical_size=args.canonical_size,
                train_jpeg_qualities=args.train_jpeg_qualities,
                validation_jpeg_quality=args.jpeg_quality,
                jpeg_subsampling=args.jpeg_subsampling,
                jpeg_optimize=args.jpeg_optimize,
                jpeg_progressive=args.jpeg_progressive,
            )
        )
        if args.train_noise_probability > 0:
            training_transform.transforms.insert(
                2, RandomGaussianNoise(args.train_noise_probability, args.train_noise_max_std)
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
        # Adaptive runs recreate seeded workers each epoch for resumable augmentation.
        "persistent_workers": args.workers > 0 and not bool(
            getattr(args, "adaptive_strategy", None)
        ),
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
    if args.method_loss_weights_json is not None:
        args.method_loss_weights_json = args.method_loss_weights_json.resolve()
    adaptive = args.adaptive_strategy is not None
    if adaptive != (args.adaptive_protocol_config is not None and args.adaptive_validation_roles is not None):
        raise ValueError("Adaptive strategy requires both protocol config and validation roles")
    if not adaptive and (args.adaptive_protocol_config or args.adaptive_validation_roles):
        raise ValueError("Specify --adaptive-strategy with adaptive paths")
    if adaptive and (args.canonical_size != 256 or args.jpeg_quality != 95 or args.jpeg_subsampling != 2
                     or args.jpeg_optimize or args.jpeg_progressive or args.train_jpeg_qualities is not None
                     or args.train_noise_probability != 0 or args.method_loss_weights_json is not None
                     or args.unweighted_loss):
        raise ValueError("Adaptive pilot requires standard fixed-Q95 M7 with no static weighting/noise")
    if args.model not in {"xception", "legacy_xception"}:
        raise ValueError("This ablation is restricted to Xception.")
    controlled_reencode = args.canonical_size is not None
    if not 0 <= args.train_noise_probability <= 1 or args.train_noise_max_std <= 0:
        raise ValueError("Invalid train noise probability or strength")
    if args.train_noise_probability > 0 and args.train_jpeg_qualities is None:
        raise ValueError("Pilot noise requires the Mixed-JPEG training pipeline")
    if controlled_reencode != (args.jpeg_quality is not None):
        raise ValueError(
            "--canonical-size and --jpeg-quality must be provided together."
        )
    if controlled_reencode:
        if args.canonical_size < 1:
            raise ValueError("--canonical-size must be positive.")
        if not 1 <= args.jpeg_quality <= 100:
            raise ValueError("--jpeg-quality must be between 1 and 100.")
        if args.jpeg_subsampling not in {0, 1, 2}:
            raise ValueError("--jpeg-subsampling must be 0, 1, or 2.")
        if args.train_jpeg_qualities is not None:
            if not args.train_jpeg_qualities:
                raise ValueError("--train-jpeg-qualities cannot be empty.")
            if len(set(args.train_jpeg_qualities)) != len(
                args.train_jpeg_qualities
            ):
                raise ValueError(
                    "--train-jpeg-qualities values must be unique."
                )
            if any(
                not 1 <= quality <= 100
                for quality in args.train_jpeg_qualities
            ):
                raise ValueError(
                    "--train-jpeg-qualities must be between 1 and 100."
                )
    elif args.train_jpeg_qualities is not None:
        raise ValueError(
            "--train-jpeg-qualities requires --canonical-size and "
            "--jpeg-quality."
        )
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        existing_names = {path.name for path in args.output_dir.iterdir()}
        config_path = args.output_dir / "config.json"
        recoverable_precheckpoint = existing_names == {"config.json"} or (
            adaptive and "config.json" in existing_names and existing_names <= {
                "config.json", "adaptive_probe_samples.csv", "adaptive_weight_history.csv"})
        if recoverable_precheckpoint:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            expected_manifest_hash = sha256_file(args.manifest)
            identity_matches = (
                existing_config.get("manifest_sha256") == expected_manifest_hash
                and int(existing_config.get("seed", -1)) == args.seed
                and existing_config.get("split_protocol") == args.split_protocol
                and existing_config.get("experiment_family") == args.experiment_family
                and existing_config.get("condition_name") == args.condition_name
            )
            if not identity_matches:
                raise RuntimeError(
                    f"Pre-checkpoint config does not match this run: {config_path}"
                )
            print(
                "Recovering run interrupted before the first checkpoint:",
                args.output_dir,
                flush=True,
            )
        else:
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
    adaptive_protocol = adaptive_roles = None
    if adaptive:
        from prepare_family_rotation_adaptive_learning import load_roles
        adaptive_protocol = json.loads(args.adaptive_protocol_config.read_text())
        if adaptive_protocol.get("protocol") != "family_rotation_adaptive_learning_v1":
            raise ValueError("Unexpected adaptive protocol")
        adaptive_roles = load_roles(args.manifest, args.adaptive_validation_roles, adaptive_protocol)
        selection_ids = set(adaptive_roles.loc[adaptive_roles.validation_role == "selection", "sample_id"])
        val_frame = val_frame[val_frame.sample_id.isin(selection_ids)].reset_index(drop=True)
    method_loss_weights = None
    if args.method_loss_weights_json is not None:
        if not args.method_loss_weights_json.is_file():
            raise FileNotFoundError(args.method_loss_weights_json)
        payload = json.loads(args.method_loss_weights_json.read_text(encoding="utf-8"))
        method_loss_weights = payload.get("weights", payload)
        fake_methods = set(train_frame.loc[train_frame["label"] == "fake", "method"])
        if set(method_loss_weights) != fake_methods:
            raise ValueError(
                "Method-loss weight membership mismatch: "
                f"expected={sorted(fake_methods)}, got={sorted(method_loss_weights)}"
            )
        method_loss_weights = {k: float(v) for k, v in method_loss_weights.items()}
        if any(not np.isfinite(v) or v <= 0 for v in method_loss_weights.values()):
            raise ValueError("Method-loss weights must be finite and positive")
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

    mixed_jpeg = args.train_jpeg_qualities is not None
    preprocessing_name = (
        MIXED_JPEG_REENCODE_NAME
        if mixed_jpeg
        else CONTROLLED_REENCODE_NAME
        if controlled_reencode
        else LETTERBOX_NAME
    )
    if controlled_reencode:
        preprocessing = {
            "policy": (
                "common aspect-preserving canonical canvas, uniformly "
                "sampled in-memory JPEG round trip, then Xception letterbox"
                if mixed_jpeg
                else "common aspect-preserving canonical canvas, fixed "
                "in-memory JPEG round trip, then Xception letterbox"
            ),
            "canonical_size": int(args.canonical_size),
            "canonical_resize": (
                "fit long edge inside canonical size with bicubic "
                "antialiasing"
            ),
            "canonical_padding": (
                "symmetric constant padding to canonical square"
            ),
            "canonical_padding_fill_rgb": LETTERBOX_FILL_RGB,
            "jpeg_subsampling": int(args.jpeg_subsampling),
            "jpeg_optimize": bool(args.jpeg_optimize),
            "jpeg_progressive": bool(args.jpeg_progressive),
            "model_input_resize": (
                "canonical image to 299x299 with bicubic antialiasing"
            ),
            "source_files_modified": False,
        }
        if mixed_jpeg:
            preprocessing.update(
                {
                    "train_jpeg_qualities": [
                        int(value) for value in args.train_jpeg_qualities
                    ],
                    "train_jpeg_sampling": "uniform",
                    "validation_jpeg_quality": int(args.jpeg_quality),
                }
            )
            preprocessing["train_noise"] = {
                "probability": args.train_noise_probability,
                "max_std_0_to_255": args.train_noise_max_std,
                "strength_sampling": "uniform(0,max_std) when applied",
                "position": "after Letterbox256, before JPEG",
                "label_independent": True,
                "validation_noise": False,
            }
        else:
            preprocessing["jpeg_quality"] = int(args.jpeg_quality)
    else:
        preprocessing = {
            "policy": "preserve full frame and aspect ratio",
            "resize": "fit long edge inside 299 with bicubic antialiasing",
            "padding": "symmetric constant padding to 299x299",
            "padding_fill_rgb": LETTERBOX_FILL_RGB,
            "distortion": "none",
        }
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "resume": str(args.resume) if args.resume else None,
        "experiment_family": args.experiment_family,
        "condition_name": args.condition_name,
        "preprocessing_name": preprocessing_name,
        "preprocessing": preprocessing,
        "manifest_sha256": sha256_file(args.manifest),
        "method_loss_weights": method_loss_weights,
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
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "pillow": PIL.__version__,
            "libjpeg": features.version_codec("jpg"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    controller = None
    if adaptive:
        from family_rotation_adaptive_learning import AdaptiveController
        config["adaptive_learning"] = {
            "strategy": args.adaptive_strategy, "protocol": adaptive_protocol,
            "protocol_sha256": sha256_file(args.adaptive_protocol_config),
            "validation_roles_sha256": sha256_file(args.adaptive_validation_roles),
            "implementation_sha256": sha256_file(Path(__file__).with_name("family_rotation_adaptive_learning.py")),
            "trainer_implementation_sha256": sha256_file(Path(__file__)),
            "meta_validation_images": int(adaptive_roles.validation_role.eq("meta").sum()),
            "checkpoint_validation_images": len(val_frame),
        }
        old_config = args.output_dir / "config.json"
        if old_config.exists():
            old = json.loads(old_config.read_text())
            for key in ("manifest_sha256", "seed", "adaptive_learning", "epochs", "batch_size",
                        "learning_rate", "weight_decay", "patience", "min_delta", "workers"):
                if json_ready(old.get(key)) != json_ready(config.get(key)):
                    raise RuntimeError(f"Adaptive resume configuration differs: {key}")
    if adaptive:
        from prepare_family_rotation_adaptive_learning import atomic_json
        atomic_json(args.output_dir / "config.json", json_ready(config))
        controller = AdaptiveController(args.adaptive_strategy, adaptive_protocol, train_frame,
            adaptive_roles, args.data_root, val_transform, args.output_dir, workers=args.workers)
    else:
        write_json(args.output_dir / "config.json", config)

    history = []
    best_auc = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 1
    adaptive_elapsed_seconds = 0.0
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if checkpoint["model_name"] != args.model:
            raise ValueError("Resume model does not match.")
        if checkpoint["config"].get("preprocessing_name") != preprocessing_name:
            raise ValueError(
                "Resume checkpoint preprocessing does not match this run."
            )
        if checkpoint["config"].get("preprocessing") != preprocessing:
            raise ValueError(
                "Resume checkpoint preprocessing parameters do not match."
            )
        if checkpoint["config"].get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("Resume manifest does not match.")
        if adaptive:
            if json_ready(checkpoint["config"].get("adaptive_learning")) != json_ready(config["adaptive_learning"]):
                raise ValueError("Resume adaptive strategy/protocol/validation roles differ")
            controller.load_state_dict(checkpoint["adaptive_state"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_auc = checkpoint.get("best_val_auc", float("-inf"))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        start_epoch = checkpoint["epoch"] + 1
        if adaptive:
            import random
            rng = checkpoint["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
            train_loader.generator.set_state(checkpoint["train_loader_generator_state"].cpu())
            adaptive_elapsed_seconds = checkpoint.get("adaptive_train_validation_seconds", 0.0)
        if epochs_without_improvement >= args.patience:
            print(
                "Run already reached early stopping at epoch "
                f"{checkpoint['epoch']}; nothing to resume."
            )
            return

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]))
    print("Validation distribution:", Counter(val_frame["label"]))
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        print(f"\nEpoch {epoch}/{args.epochs}")
        if controller is not None:
            train_metrics = controller.train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        elif method_loss_weights is None:
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                persistent_progress=args.persistent_progress,
            )
        else:
            train_metrics = train_one_epoch_method_weighted(
                model, train_loader, weight_tensor, method_loss_weights,
                optimizer, scaler, device,
                persistent_progress=args.persistent_progress,
            )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            persistent_progress=args.persistent_progress,
        )
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
        if controller is not None:
            import random
            payload["adaptive_state"] = controller.state_dict()
            payload["rng_state"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
            payload["train_loader_generator_state"] = train_loader.generator.get_state()
            adaptive_elapsed_seconds += time.monotonic() - epoch_started
            payload["adaptive_train_validation_seconds"] = adaptive_elapsed_seconds
            payload["ordinary_train_images_seen"] = epoch * len(train_frame)
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
