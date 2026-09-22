#!/usr/bin/env python3
"""Train the M7 method-by-quality adaptive reweighting pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
from collections import Counter, defaultdict
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


EXPERIMENT_FAMILY = "method_quality_adaptive_pilot_v1"
PREPROCESSING_NAME = (
    "canonical256_jpegmixed_methodquality_adaptive_letterbox299"
)
CONDITION_ID = "M7"
REAL_METHOD = "original"


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
        default=project_root / f"configs/{EXPERIMENT_FAMILY}/protocol.json",
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
        raise ValueError("Unexpected method-quality adaptive protocol.")
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
    expected_objective = {
        "fake_methods": [
            "SimSwap",
            "BlendFace",
            "Wav2Lip",
            "FOMM",
            "StyleGAN3",
            "DiT",
        ],
        "reference_quality": 95,
        "ema_beta": 0.9,
        "minimum_group_observations": 16,
        "softmax_temperature": 0.25,
        "uniform_objective_fraction": 0.5,
        "maximum_group_weight_multiplier": 3.0,
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
        raise ValueError("The pilot must use exactly one view per image.")
    return protocol


def run_name() -> str:
    return (
        "xception_m7_fs_fr_efs_"
        "canonical256_jpegmix_methodquality_adaptive_"
        "letterbox299_pilot_v1_seed42"
    )


class MethodQualityDataset(Dataset):
    """Return one uniformly sampled JPEG view and its observed group."""

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

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = self.data_root / row["source_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if random.random() < 0.5:
                image = functional.hflip(image)
            image = self.canonical(image)
            quality = random.choice(self.qualities)
            image = JpegRoundTrip(
                quality=quality,
                subsampling=self.jpeg_subsampling,
                optimize=self.jpeg_optimize,
                progressive=self.jpeg_progressive,
            )(image)
            image = self.model_letterbox(image)
            image = self.normalize(self.to_tensor(image))
        return {
            "image": image,
            "label": torch.tensor(
                LABEL_MAP[str(row["label"])], dtype=torch.long
            ),
            "method": str(row["method"]),
            "quality": int(quality),
        }


def _capped_distribution(target: np.ndarray, cap: float) -> np.ndarray:
    """Project positive weights to a simplex with a shared upper bound."""
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()
    count = len(target)
    if cap * count < 1.0 - 1e-12:
        raise ValueError("Group-weight cap cannot sum to one.")
    output = np.zeros(count, dtype=np.float64)
    free = np.ones(count, dtype=bool)
    remaining = 1.0
    while free.any():
        proposal = target[free]
        proposal = proposal / proposal.sum() * remaining
        over = proposal > cap + 1e-12
        free_indices = np.flatnonzero(free)
        if not over.any():
            output[free_indices] = proposal
            break
        capped_indices = free_indices[over]
        output[capped_indices] = cap
        free[capped_indices] = False
        remaining = 1.0 - float(output.sum())
    output = output / output.sum()
    return output


class MethodQualityWeights:
    """Online excess-loss weights, normalized separately for Real/Fake."""

    def __init__(self, objective: dict) -> None:
        self.fake_methods = tuple(objective["fake_methods"])
        self.methods = (REAL_METHOD, *self.fake_methods)
        self.qualities = (75, 80, 85, 90, 95)
        self.reference_quality = int(objective["reference_quality"])
        self.beta = float(objective["ema_beta"])
        self.minimum_observations = int(
            objective["minimum_group_observations"]
        )
        self.temperature = float(objective["softmax_temperature"])
        self.uniform_fraction = float(
            objective["uniform_objective_fraction"]
        )
        self.maximum_multiplier = float(
            objective["maximum_group_weight_multiplier"]
        )
        self.group_keys = tuple(
            (method, quality)
            for method in self.methods
            for quality in self.qualities
        )
        self.ema_loss = {key: None for key in self.group_keys}
        self.observations = {key: 0 for key in self.group_keys}
        self.weights = self._uniform_weights()

    def _keys_for_label(self, label: int) -> tuple[tuple[str, int], ...]:
        methods = (REAL_METHOD,) if label == LABEL_MAP["real"] else self.fake_methods
        return tuple(
            (method, quality)
            for method in methods
            for quality in self.qualities
        )

    def _uniform_weights(self) -> dict[tuple[str, int], float]:
        output: dict[tuple[str, int], float] = {}
        for label in (LABEL_MAP["real"], LABEL_MAP["fake"]):
            keys = self._keys_for_label(label)
            for key in keys:
                output[key] = 1.0 / len(keys)
        return output

    def multiplier(self, method: str, quality: int, label: int) -> float:
        key = (method, int(quality))
        keys = self._keys_for_label(int(label))
        if key not in keys:
            raise ValueError(f"Unexpected method-quality group: {key}")
        return len(keys) * self.weights[key]

    def observe(
        self,
        methods: list[str],
        qualities: list[int],
        losses: list[float],
    ) -> None:
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for method, quality, loss in zip(methods, qualities, losses):
            key = (str(method), int(quality))
            if key not in self.ema_loss:
                raise ValueError(f"Unexpected observed group: {key}")
            grouped[key].append(float(loss))
        for key, values in grouped.items():
            batch_mean = float(np.mean(values))
            previous = self.ema_loss[key]
            self.ema_loss[key] = (
                batch_mean
                if previous is None
                else self.beta * previous + (1.0 - self.beta) * batch_mean
            )
            self.observations[key] += len(values)
        self._refresh_weights()

    def _refresh_weights(self) -> None:
        if any(
            self.observations[key] < self.minimum_observations
            for key in self.group_keys
        ):
            return
        output: dict[tuple[str, int], float] = {}
        for label in (LABEL_MAP["real"], LABEL_MAP["fake"]):
            keys = self._keys_for_label(label)
            excess = []
            for method, quality in keys:
                reference = self.ema_loss[(method, self.reference_quality)]
                current = self.ema_loss[(method, quality)]
                excess.append(max(float(current) - float(reference), 0.0))
            logits = np.asarray(excess, dtype=np.float64) / self.temperature
            logits -= logits.max()
            softmax = np.exp(logits)
            softmax /= softmax.sum()
            uniform = np.full(len(keys), 1.0 / len(keys))
            target = (
                self.uniform_fraction * uniform
                + (1.0 - self.uniform_fraction) * softmax
            )
            bounded = _capped_distribution(
                target,
                cap=self.maximum_multiplier / len(keys),
            )
            output.update(dict(zip(keys, bounded.tolist())))
        self.weights = output

    def summary_rows(
        self,
        *,
        epoch: int,
        epoch_loss_sum: dict[tuple[str, int], float],
        epoch_counts: dict[tuple[str, int], int],
    ) -> list[dict]:
        rows = []
        for method, quality in self.group_keys:
            count = int(epoch_counts.get((method, quality), 0))
            mean_loss = (
                float(epoch_loss_sum[(method, quality)] / count)
                if count
                else math.nan
            )
            reference = self.ema_loss[(method, self.reference_quality)]
            current = self.ema_loss[(method, quality)]
            excess = (
                max(float(current) - float(reference), 0.0)
                if current is not None and reference is not None
                else math.nan
            )
            label = "real" if method == REAL_METHOD else "fake"
            label_id = LABEL_MAP[label]
            rows.append(
                {
                    "epoch": epoch,
                    "label": label,
                    "method": method,
                    "jpeg_quality": quality,
                    "epoch_samples": count,
                    "epoch_mean_loss": mean_loss,
                    "ema_loss": current,
                    "q95_reference_ema_loss": reference,
                    "ema_excess_loss": excess,
                    "objective_weight": self.weights[(method, quality)],
                    "loss_multiplier": self.multiplier(
                        method, quality, label_id
                    ),
                    "observations": self.observations[(method, quality)],
                }
            )
        return rows

    def state_dict(self) -> dict:
        def encode(mapping: dict) -> dict:
            return {
                f"{method}|{quality}": value
                for (method, quality), value in mapping.items()
            }

        return {
            "fake_methods": list(self.fake_methods),
            "qualities": list(self.qualities),
            "reference_quality": self.reference_quality,
            "beta": self.beta,
            "minimum_observations": self.minimum_observations,
            "temperature": self.temperature,
            "uniform_fraction": self.uniform_fraction,
            "maximum_multiplier": self.maximum_multiplier,
            "ema_loss": encode(self.ema_loss),
            "observations": encode(self.observations),
            "weights": encode(self.weights),
        }

    def load_state_dict(self, state: dict) -> None:
        checks = {
            "fake_methods": list(self.fake_methods),
            "qualities": list(self.qualities),
            "reference_quality": self.reference_quality,
            "beta": self.beta,
            "minimum_observations": self.minimum_observations,
            "temperature": self.temperature,
            "uniform_fraction": self.uniform_fraction,
            "maximum_multiplier": self.maximum_multiplier,
        }
        for key, expected in checks.items():
            if state.get(key) != expected:
                raise ValueError(f"Adaptive-state mismatch for {key}.")

        def decode(mapping: dict) -> dict:
            output = {}
            for encoded, value in mapping.items():
                method, quality = encoded.rsplit("|", 1)
                output[(method, int(quality))] = value
            return output

        self.ema_loss = decode(state["ema_loss"])
        self.observations = {
            key: int(value)
            for key, value in decode(state["observations"]).items()
        }
        self.weights = {
            key: float(value)
            for key, value in decode(state["weights"]).items()
        }
        if set(self.weights) != set(self.group_keys):
            raise ValueError("Adaptive-state groups changed.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    adaptive: MethodQualityWeights,
    device: torch.device,
    *,
    epoch: int,
    persistent_progress: bool,
) -> dict:
    model.train()
    use_amp = device.type == "cuda"
    total_objective = 0.0
    total_unweighted_ce = 0.0
    total_correct = 0
    total_samples = 0
    epoch_loss_sum: dict[tuple[str, int], float] = defaultdict(float)
    epoch_counts: dict[tuple[str, int], int] = defaultdict(int)
    progress = tqdm(
        loader,
        desc="Train method-quality adaptive",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        methods = [str(value) for value in batch["method"]]
        qualities = [int(value) for value in batch["quality"].tolist()]
        multipliers = torch.tensor(
            [
                adaptive.multiplier(method, quality, int(label))
                for method, quality, label in zip(
                    methods, qualities, labels.detach().cpu().tolist()
                )
            ],
            dtype=torch.float32,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            per_sample_ce = criterion(logits, labels)
            objective_loss = (per_sample_ce.float() * multipliers).mean()
        scaler.scale(objective_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses = per_sample_ce.detach().float().cpu().tolist()
        adaptive.observe(methods, qualities, losses)
        for method, quality, loss in zip(methods, qualities, losses):
            key = (method, quality)
            epoch_loss_sum[key] += float(loss)
            epoch_counts[key] += 1

        batch_size = labels.size(0)
        total_objective += float(objective_loss.detach()) * batch_size
        total_unweighted_ce += float(per_sample_ce.detach().sum())
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        largest = max(
            adaptive.weights,
            key=lambda key: adaptive.multiplier(
                key[0], key[1],
                LABEL_MAP["real"] if key[0] == REAL_METHOD else LABEL_MAP["fake"],
            ),
        )
        progress.set_postfix(
            loss=f"{float(objective_loss.detach()):.4f}",
            focus=f"{largest[0]}@Q{largest[1]}",
        )
    if total_samples != 39_600:
        raise RuntimeError(f"Unexpected epoch sample count: {total_samples}")
    return {
        "loss": total_objective / total_samples,
        "unweighted_classification_loss": total_unweighted_ce / total_samples,
        "accuracy": total_correct / total_samples,
        "groups": adaptive.summary_rows(
            epoch=epoch,
            epoch_loss_sum=epoch_loss_sum,
            epoch_counts=epoch_counts,
        ),
    }


def write_history_csv(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_unweighted_classification_loss",
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
                    "train_unweighted_classification_loss": row["train"][
                        "unweighted_classification_loss"
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


def write_group_history_csv(path: Path, history: list[dict]) -> None:
    rows = [group for epoch in history for group in epoch["train"]["groups"]]
    pd.DataFrame(rows).to_csv(path, index=False)


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
        raise RuntimeError(f"Unexpected Xception input: {data_config['input_size']}")
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
    train_criterion = nn.CrossEntropyLoss(
        weight=weight_tensor, reduction="none"
    )
    val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    adaptive = MethodQualityWeights(protocol["objective"])
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
        "condition_name": "m7_fs_fr_efs_method_quality_adaptive",
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
            "horizontal flip p=0.5 -> Letterbox256 -> one uniform random "
            "JPEG Q75/Q80/Q85/Q90/Q95 -> Letterbox299 -> Normalize; "
            "online method-quality excess-loss reweighting"
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
            "objective": protocol["objective"],
            "optimization": protocol["optimization"],
        }
        for key, expected in checks.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch for {key}.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        adaptive.load_state_dict(checkpoint["adaptive_weight_state"])
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
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            adaptive,
            device,
            epoch=epoch,
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            val_criterion,
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
        payload["adaptive_weight_state"] = adaptive.state_dict()
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)
        write_group_history_csv(output_dir / "group_history.csv", history)
        focus = max(
            adaptive.weights,
            key=lambda key: adaptive.multiplier(
                key[0], key[1],
                LABEL_MAP["real"] if key[0] == REAL_METHOD else LABEL_MAP["fake"],
            ),
        )
        print(
            f"Train objective={train_metrics['loss']:.4f}, "
            f"raw CE={train_metrics['unweighted_classification_loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}",
            flush=True,
        )
        print(
            f"Adaptive focus: {focus[0]}@Q{focus[1]} "
            f"(multiplier={adaptive.multiplier(focus[0], focus[1], 0 if focus[0] == REAL_METHOD else 1):.3f})",
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
    print("\nMethod-quality adaptive pilot complete.", flush=True)
    print("Best validation AUC:", best_auc, flush=True)
    print("Run directory:", output_dir, flush=True)


if __name__ == "__main__":
    main()
