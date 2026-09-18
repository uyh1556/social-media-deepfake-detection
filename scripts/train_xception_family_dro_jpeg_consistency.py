#!/usr/bin/env python3
"""Train M7 with family Group DRO and paired-JPEG consistency."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
from PIL import features
from torch import nn
from torch.utils.data import DataLoader, Sampler
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
from train_xception_jpeg_consistency import PairedJpegDataset
from xception_preprocessing import canonical_reencode_transform


EXPERIMENT_FAMILY = "family_dro_jpeg_consistency_pilot_v1"
PREPROCESSING_NAME = "canonical256_jpegpair_familydro_letterbox299"
CONDITION_ID = "M7"
ALL_GROUPS = ("real", "FS", "FR", "EFS")
DRO_GROUPS = ("FS", "FR", "EFS")
GROUP_TO_INDEX = {name: index for index, name in enumerate(ALL_GROUPS)}


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
        raise ValueError("Unexpected family-DRO pilot protocol.")
    if protocol.get("models") != [CONDITION_ID]:
        raise ValueError("The pilot must be restricted to M7.")
    if protocol.get("training_seeds") != [42]:
        raise ValueError("The pilot must be restricted to seed 42.")
    preprocessing = protocol["preprocessing"]
    expected_preprocessing = {
        "name": PREPROCESSING_NAME,
        "canonical_size": 256,
        "train_jpeg_qualities": [75, 80, 85, 90, 95],
        "paired_quality_sampling": "uniform ordered pair without replacement",
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
    if objective.get("family_dro_groups") != list(DRO_GROUPS):
        raise ValueError("Unexpected fake-family Group DRO groups.")
    if objective.get("real_group") != "real":
        raise ValueError("The Real risk must remain a separate fixed class term.")
    if float(objective.get("consistency_weight")) != 0.1:
        raise ValueError("The frozen consistency weight must be 0.1.")
    if float(objective.get("group_dro_step_size")) != 0.01:
        raise ValueError("The frozen Group DRO step size must be 0.01.")
    return protocol


def run_name() -> str:
    return (
        "xception_m7_fs_fr_efs_"
        "canonical256_jpegpair_familydro_eta0p01_js0p1_"
        "letterbox299_pilot_v1_seed42"
    )


class FamilyBalancedBatchSampler(Sampler[list[int]]):
    """Use every item once while guaranteeing all four groups per batch."""

    def __init__(self, frame: pd.DataFrame, *, seed: int) -> None:
        if len(frame) != 39_600:
            raise ValueError(f"Unexpected M7 training size: {len(frame)}")
        self.seed = int(seed)
        self.epoch = 0
        self.indices: dict[str, np.ndarray] = {}
        for group in ALL_GROUPS:
            values = frame.index[frame["family"] == group].to_numpy()
            self.indices[group] = values.astype(np.int64, copy=False)
        expected = {"real": 19_800, "FS": 6_600, "FR": 6_600, "EFS": 6_600}
        actual = {key: len(value) for key, value in self.indices.items()}
        if actual != expected:
            raise ValueError(
                f"Unexpected M7 family counts: expected={expected}, actual={actual}"
            )
        self.num_batches = 4_950

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {
            group: rng.permutation(indices).tolist()
            for group, indices in self.indices.items()
        }
        positions = {group: 0 for group in ALL_GROUPS}
        extras = np.repeat(np.array(DRO_GROUPS), 1_650)
        rng.shuffle(extras)

        def take(group: str, count: int) -> list[int]:
            start = positions[group]
            stop = start + count
            positions[group] = stop
            return pools[group][start:stop]

        for extra_group in extras.tolist():
            batch = take("real", 4)
            for group in DRO_GROUPS:
                batch.extend(take(group, 2 if group == extra_group else 1))
            rng.shuffle(batch)
            yield batch
        final = {group: positions[group] for group in ALL_GROUPS}
        expected = {group: len(self.indices[group]) for group in ALL_GROUPS}
        if final != expected:
            raise RuntimeError(
                f"Sampler did not consume the epoch exactly: {final} != {expected}"
            )


class GroupDRO:
    """Exponentiated-gradient weights over the frozen manifest groups."""

    def __init__(self, *, step_size: float, device: torch.device) -> None:
        self.step_size = float(step_size)
        self.log_weights = torch.zeros(
            len(DRO_GROUPS), dtype=torch.float32, device=device
        )

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def loss(self, group_losses: torch.Tensor) -> torch.Tensor:
        if group_losses.shape != (len(DRO_GROUPS),):
            raise ValueError("One loss is required for every fake family.")
        with torch.no_grad():
            self.log_weights.add_(self.step_size * group_losses.detach())
            self.log_weights.sub_(torch.logsumexp(self.log_weights, dim=0))
        return torch.sum(self.weights.detach() * group_losses)

    def state_dict(self) -> dict:
        return {
            "groups": list(DRO_GROUPS),
            "step_size": self.step_size,
            "log_weights": self.log_weights.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("groups") != list(DRO_GROUPS):
            raise ValueError("Group DRO checkpoint groups changed.")
        if float(state.get("step_size")) != self.step_size:
            raise ValueError("Group DRO checkpoint step size changed.")
        self.log_weights.copy_(state["log_weights"].to(self.log_weights.device))


def per_image_js(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probabilities_a = torch.softmax(logits_a.float(), dim=1).clamp_min(1e-7)
    probabilities_b = torch.softmax(logits_b.float(), dim=1).clamp_min(1e-7)
    midpoint = (0.5 * (probabilities_a + probabilities_b)).clamp_min(1e-7)
    divergence_a = (
        probabilities_a * (probabilities_a.log() - midpoint.log())
    ).sum(dim=1)
    divergence_b = (
        probabilities_b * (probabilities_b.log() - midpoint.log())
    ).sum(dim=1)
    return 0.5 * (divergence_a + divergence_b)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    batch_sampler: FamilyBalancedBatchSampler,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    group_dro: GroupDRO,
    device: torch.device,
    *,
    epoch: int,
    consistency_weight: float,
    accumulation_steps: int,
    persistent_progress: bool,
) -> dict:
    model.train()
    batch_sampler.set_epoch(epoch)
    use_amp = device.type == "cuda"
    totals = {
        "samples": 0,
        "correct": 0,
        "ce": 0.0,
        "js": 0.0,
        "robust": 0.0,
    }
    group_ce_sum = torch.zeros(len(ALL_GROUPS), dtype=torch.float64)
    group_js_sum = torch.zeros(len(ALL_GROUPS), dtype=torch.float64)
    group_counts = torch.zeros(len(ALL_GROUPS), dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        desc="Train family DRO + JPEG consistency",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for step, batch in enumerate(progress, start=1):
        view_a = batch["image_a"].to(device, non_blocking=True)
        view_b = batch["image_b"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        group_ids = torch.tensor(
            [GROUP_TO_INDEX[name] for name in batch["family"]],
            dtype=torch.long,
            device=device,
        )
        paired_images = torch.cat([view_a, view_b], dim=0)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            paired_logits = model(paired_images)
            logits_a, logits_b = paired_logits.chunk(2, dim=0)
            per_sample_ce = criterion(logits_a, labels)
        per_sample_consistency = per_image_js(logits_a, logits_b)
        group_ce = torch.stack(
            [
                per_sample_ce[group_ids == index].mean()
                for index in range(len(ALL_GROUPS))
            ]
        ).float()
        group_js = torch.stack(
            [
                per_sample_consistency[group_ids == index].mean()
                for index in range(len(ALL_GROUPS))
            ]
        )
        group_total = group_ce + consistency_weight * group_js
        fake_family_robust_loss = group_dro.loss(group_total[1:])
        robust_loss = 0.5 * group_total[0] + 0.5 * fake_family_robust_loss
        scaler.scale(robust_loss / accumulation_steps).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        totals["samples"] += batch_size
        totals["correct"] += (logits_a.argmax(dim=1) == labels).sum().item()
        totals["ce"] += float(per_sample_ce.detach().sum())
        totals["js"] += float(per_sample_consistency.detach().sum())
        totals["robust"] += float(robust_loss.detach()) * batch_size
        for index in range(len(ALL_GROUPS)):
            mask = group_ids == index
            count = int(mask.sum())
            group_counts[index] += count
            group_ce_sum[index] += float(per_sample_ce[mask].detach().sum())
            group_js_sum[index] += float(
                per_sample_consistency[mask].detach().sum()
            )
        weights = group_dro.weights.detach().cpu().tolist()
        progress.set_postfix(
            robust=f"{float(robust_loss.detach()):.4f}",
            ce=f"{float(per_sample_ce.detach().mean()):.4f}",
            js=f"{float(per_sample_consistency.detach().mean()):.4f}",
            worst=DRO_GROUPS[int(np.argmax(weights))],
        )
    if totals["samples"] != 39_600:
        raise RuntimeError(f"Unexpected samples in epoch: {totals['samples']}")
    counts = group_counts.numpy()
    group_metrics = {}
    final_weights = group_dro.weights.detach().cpu().numpy()
    effective_weights = np.concatenate(
        [np.array([0.5]), 0.5 * final_weights]
    )
    for index, group in enumerate(ALL_GROUPS):
        group_metrics[group] = {
            "samples": int(counts[index]),
            "classification_loss": float(group_ce_sum[index] / counts[index]),
            "consistency_loss": float(group_js_sum[index] / counts[index]),
            "objective_weight": float(effective_weights[index]),
        }
    return {
        "loss": totals["robust"] / totals["samples"],
        "classification_loss": totals["ce"] / totals["samples"],
        "consistency_loss": totals["js"] / totals["samples"],
        "accuracy": totals["correct"] / totals["samples"],
        "groups": group_metrics,
    }


def write_history_csv(path: Path, history: list[dict]) -> None:
    fieldnames = [
        "epoch", "learning_rate", "train_loss",
        "train_classification_loss", "train_consistency_loss",
        "train_accuracy", "val_loss", "val_accuracy", "val_precision",
        "val_recall", "val_f1", "val_roc_auc",
    ]
    for group in ALL_GROUPS:
        fieldnames.extend(
            [f"{group}_ce", f"{group}_js", f"{group}_objective_weight"]
        )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            output = {
                "epoch": row["epoch"],
                "learning_rate": row["learning_rate"],
                "train_loss": row["train"]["loss"],
                "train_classification_loss": row["train"]["classification_loss"],
                "train_consistency_loss": row["train"]["consistency_loss"],
                "train_accuracy": row["train"]["accuracy"],
                "val_loss": row["val"]["loss"],
                "val_accuracy": row["val"]["accuracy"],
                "val_precision": row["val"]["precision"],
                "val_recall": row["val"]["recall"],
                "val_f1": row["val"]["f1"],
                "val_roc_auc": row["val"]["roc_auc"],
            }
            for group in ALL_GROUPS:
                metrics = row["train"]["groups"][group]
                output[f"{group}_ce"] = metrics["classification_loss"]
                output[f"{group}_js"] = metrics["consistency_loss"]
                output[f"{group}_objective_weight"] = metrics[
                    "objective_weight"
                ]
            writer.writerow(output)


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
        raise ValueError("The pilot requires batch-size 8 and accumulation 2.")
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
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = timm.create_model(
        "xception", pretrained=args.resume is None, num_classes=len(LABEL_MAP)
    )
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    if tuple(data_config["input_size"]) != (3, 299, 299):
        raise RuntimeError(f"Unexpected Xception input: {data_config['input_size']}")
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
    batch_sampler = FamilyBalancedBatchSampler(train_frame, seed=args.seed)
    validation_transform = canonical_reencode_transform(
        data_config,
        canonical_size=preprocessing["canonical_size"],
        jpeg_quality=preprocessing["validation_jpeg_quality"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    val_dataset = FFPPDataset(val_frame, args.data_root, validation_transform)
    common = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, **common
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        **common,
    )
    if len(train_loader) % args.gradient_accumulation_steps:
        raise RuntimeError("Epoch batches must fit complete accumulation groups.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")
    validation_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    objective = protocol["objective"]
    group_dro = GroupDRO(
        step_size=objective["group_dro_step_size"], device=device
    )
    consistency_weight = float(objective["consistency_weight"])
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
        "condition_name": "m7_fs_fr_efs_family_dro_jpeg_consistency",
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
            "shared flip -> Letterbox256 -> two distinct random JPEG "
            "qualities -> Letterbox299; fixed 0.5 Real risk plus 0.5 "
            "Group DRO over FS/FR/EFS risks, each containing anchor "
            "CE + 0.1 paired-view JS"
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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        group_dro.load_state_dict(checkpoint["group_dro_state"])
        history = checkpoint.get("history", [])
        best_auc = checkpoint.get("best_val_auc", float("-inf"))
        epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        if epochs_without_improvement >= args.patience:
            print(f"Pilot already stopped at epoch {checkpoint['epoch']}.", flush=True)
            return

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]), flush=True)
    print("Train family distribution:", Counter(train_frame["family"]), flush=True)
    print("Validation distribution:", Counter(val_frame["label"]), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            batch_sampler,
            criterion,
            optimizer,
            scaler,
            group_dro,
            device,
            epoch=epoch,
            consistency_weight=consistency_weight,
            accumulation_steps=args.gradient_accumulation_steps,
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            validation_criterion,
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
        payload["group_dro_state"] = group_dro.state_dict()
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        write_history_csv(output_dir / "history.csv", history)
        weights = train_metrics["groups"]
        print(
            f"Train robust={train_metrics['loss']:.4f}, "
            f"CE={train_metrics['classification_loss']:.4f}, "
            f"JS={train_metrics['consistency_loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}",
            flush=True,
        )
        print(
            "DRO weights: "
            + ", ".join(
                f"{group}={weights[group]['objective_weight']:.4f}"
                for group in ALL_GROUPS
            ),
            flush=True,
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, AUC={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        print("Confusion matrix:", val_metrics["confusion_matrix"], flush=True)
        print(
            f"Early stopping: {epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break
    print("\nFamily-DRO + consistency pilot complete.", flush=True)
    print("Best validation AUC:", best_auc, flush=True)
    print("Run directory:", output_dir, flush=True)


if __name__ == "__main__":
    main()
