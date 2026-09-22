#!/usr/bin/env python3
"""Train one M7 model with method-episodic and quality-invariant learning."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from collections import Counter, OrderedDict
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
from torch.func import functional_call
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


EXPERIMENT_FAMILY = "method_episodic_quality_invariant_pilot_v1"
PREPROCESSING_NAME = "canonical256_jpegpair_method_episodic_letterbox299"
CONDITION_ID = "M7"


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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=16)
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
        raise ValueError("Unexpected method-episodic protocol.")
    if protocol.get("models") != [CONDITION_ID]:
        raise ValueError("The pilot must be restricted to M7.")
    if protocol.get("training_seeds") != [42]:
        raise ValueError("The pilot must be restricted to seed 42.")
    episode = protocol["episode"]
    expected_episode = {
        "unique_images": 8,
        "support_real": 3,
        "support_fake_methods": 3,
        "query_real": 1,
        "query_fake_methods": 1,
        "episodes_per_epoch": 4950,
        "query_episodes_per_method": 825,
        "all_training_images_once_per_epoch": True,
    }
    for key, value in expected_episode.items():
        if episode.get(key) != value:
            raise ValueError(
                f"Episode mismatch for {key}: {episode.get(key)} != {value}"
            )
    objective = protocol["objective"]
    if float(objective.get("inner_step_size")) != 0.001:
        raise ValueError("Unexpected inner step size.")
    if float(objective.get("meta_query_weight")) != 1.0:
        raise ValueError("Unexpected meta-query weight.")
    if float(objective.get("consistency_weight")) != 0.1:
        raise ValueError("Unexpected consistency weight.")
    if protocol["optimization"].get("gradient_accumulation_steps") != 2:
        raise ValueError("Unexpected gradient accumulation.")
    return protocol


def run_name() -> str:
    return (
        "xception_m7_fs_fr_efs_"
        "method_episodic_quality_invariant_letterbox299_"
        "pilot_v1_seed42"
    )


class EpisodicPairedJpegDataset(PairedJpegDataset):
    """Accept sampler tuples and attach the frozen episode role."""

    def __getitem__(self, key) -> dict:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("Episodic sampler must yield (index, role, method).")
        index, episode_role, query_method = key
        item = super().__getitem__(int(index))
        item["episode_role"] = str(episode_role)
        item["query_method"] = str(query_method)
        item["method"] = str(self.frame.iloc[int(index)]["method"])
        return item


class MethodEpisodicBatchSampler(Sampler[list[tuple[int, str, str]]]):
    """Consume M7 exactly once while rotating every fake method as query."""

    def __init__(self, frame: pd.DataFrame, methods: list[str], seed: int):
        self.frame = frame.reset_index(drop=True)
        self.methods = tuple(methods)
        self.seed = int(seed)
        self.epoch = 0
        self.real_indices = self.frame.index[
            self.frame["label"] == "real"
        ].to_numpy(dtype=np.int64)
        self.fake_indices = {
            method: self.frame.index[
                (self.frame["label"] == "fake")
                & (self.frame["method"] == method)
            ].to_numpy(dtype=np.int64)
            for method in self.methods
        }
        if len(self.real_indices) != 19_800:
            raise ValueError("M7 must contain 19,800 Real train images.")
        counts = {key: len(value) for key, value in self.fake_indices.items()}
        if set(counts.values()) != {3_300}:
            raise ValueError(f"Every M7 fake method must have 3,300 images: {counts}")
        self.num_batches = 4_950
        self.query_per_method = 825

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[tuple[int, str, str]]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        method_order = list(self.methods)
        rng.shuffle(method_order)
        method_position = {method: i for i, method in enumerate(method_order)}

        real = rng.permutation(self.real_indices)
        real_query = real[: self.num_batches].tolist()
        real_support = real[self.num_batches :].tolist()
        real_support_position = 0

        fake_query: dict[str, list[int]] = {}
        fake_support: dict[str, list[int]] = {}
        support_position = {method: 0 for method in self.methods}
        for method, indices in self.fake_indices.items():
            shuffled = rng.permutation(indices)
            fake_query[method] = shuffled[: self.query_per_method].tolist()
            fake_support[method] = shuffled[self.query_per_method :].tolist()

        query_methods = np.repeat(
            np.array(self.methods, dtype=object), self.query_per_method
        )
        rng.shuffle(query_methods)
        query_position = {method: 0 for method in self.methods}

        for episode_index, query_method_value in enumerate(query_methods.tolist()):
            query_method = str(query_method_value)
            qpos = query_position[query_method]
            query_position[query_method] += 1
            batch: list[tuple[int, str, str]] = [
                (real_query[episode_index], "query", query_method),
                (fake_query[query_method][qpos], "query", query_method),
            ]
            batch.extend(
                (index, "support", query_method)
                for index in real_support[
                    real_support_position : real_support_position + 3
                ]
            )
            real_support_position += 3

            start = method_position[query_method]
            support_methods = [
                method_order[(start + offset) % len(method_order)]
                for offset in (1, 2, 3)
            ]
            for support_method in support_methods:
                position = support_position[support_method]
                batch.append(
                    (
                        fake_support[support_method][position],
                        "support",
                        query_method,
                    )
                )
                support_position[support_method] += 1
            rng.shuffle(batch)
            yield batch

        if real_support_position != len(real_support):
            raise RuntimeError("Real support pool was not consumed exactly.")
        if any(value != 825 for value in query_position.values()):
            raise RuntimeError(f"Query rotation is not balanced: {query_position}")
        expected_support = {method: 2_475 for method in self.methods}
        if support_position != expected_support:
            raise RuntimeError(
                f"Support rotation is not balanced: {support_position}"
            )


class FeatureCapture:
    """Capture the global-pooled representation without changing Xception."""

    def __init__(self, model: nn.Module):
        if not hasattr(model, "global_pool"):
            raise AttributeError("Xception model has no global_pool module.")
        self.values: list[torch.Tensor] = []
        self.handle = model.global_pool.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        self.values.append(output.flatten(1))

    def pop(self) -> torch.Tensor:
        if len(self.values) != 1:
            raise RuntimeError(f"Expected one captured feature, got {len(self.values)}")
        return self.values.pop()

    def close(self) -> None:
        self.handle.remove()


def feature_consistency(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - nn.functional.cosine_similarity(features_a.float(), features_b.float(), dim=1)).mean()


def paired_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    consistency_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_a, logits_b = logits.chunk(2, dim=0)
    features_a, features_b = features.chunk(2, dim=0)
    classification = criterion(logits_a, labels)
    consistency = feature_consistency(features_a, features_b)
    total = classification.float() + consistency_weight * consistency
    return total, classification, consistency, logits_a


def forward_with_features(
    model: nn.Module,
    capture: FeatureCapture,
    images: torch.Tensor,
    parameters: OrderedDict[str, torch.Tensor] | None = None,
    buffers: OrderedDict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parameters is None:
        logits = model(images)
    else:
        logits = functional_call(model, (parameters, buffers), (images,))
    return logits, capture.pop()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    sampler: MethodEpisodicBatchSampler,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    inner_step_size: float,
    meta_query_weight: float,
    consistency_weight: float,
    accumulation_steps: int,
    persistent_progress: bool,
) -> dict:
    model.train()
    sampler.set_epoch(epoch)
    use_amp = device.type == "cuda"
    totals = Counter()
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)
    capture = FeatureCapture(model)
    progress = tqdm(
        loader,
        desc="Train method-episodic",
        leave=persistent_progress,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    try:
        for step, batch in enumerate(progress, start=1):
            view_a = batch["image_a"].to(device, non_blocking=True)
            view_b = batch["image_b"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            support_mask = torch.tensor(
                [value == "support" for value in batch["episode_role"]],
                dtype=torch.bool,
                device=device,
            )
            query_mask = ~support_mask
            if support_mask.sum().item() != 6 or query_mask.sum().item() != 2:
                raise RuntimeError("Every episode must contain 6 support and 2 query images.")

            support_images = torch.cat(
                [view_a[support_mask], view_b[support_mask]], dim=0
            )
            support_labels = labels[support_mask]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                support_logits, support_features = forward_with_features(
                    model, capture, support_images
                )
                support_loss, support_ce, support_consistency, support_anchor = paired_loss(
                    support_logits,
                    support_features,
                    support_labels,
                    criterion,
                    consistency_weight,
                )

            named_parameters = OrderedDict(model.named_parameters())
            inner_gradients = torch.autograd.grad(
                support_loss,
                tuple(named_parameters.values()),
                create_graph=False,
                retain_graph=True,
            )
            fast_parameters = OrderedDict(
                (name, parameter - inner_step_size * gradient.detach())
                for (name, parameter), gradient in zip(
                    named_parameters.items(), inner_gradients
                )
            )
            # Query-time BatchNorm statistics are episode-local. Support
            # updates the persistent running statistics, while functional
            # query evaluation mutates only these clones.
            buffers = OrderedDict(
                (name, value.clone()) for name, value in model.named_buffers()
            )
            query_images = torch.cat(
                [view_a[query_mask], view_b[query_mask]], dim=0
            )
            query_labels = labels[query_mask]
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                query_logits, query_features = forward_with_features(
                    model,
                    capture,
                    query_images,
                    fast_parameters,
                    buffers,
                )
                query_loss, query_ce, query_consistency, query_anchor = paired_loss(
                    query_logits,
                    query_features,
                    query_labels,
                    criterion,
                    consistency_weight,
                )
            outer_loss = support_loss.float() + meta_query_weight * query_loss.float()
            scaler.scale(outer_loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            totals["outer_loss"] += float(outer_loss.detach()) * batch_size
            totals["support_loss"] += float(support_loss.detach()) * 6
            totals["support_ce"] += float(support_ce.detach()) * 6
            totals["support_consistency"] += float(support_consistency.detach()) * 6
            totals["query_loss"] += float(query_loss.detach()) * 2
            totals["query_ce"] += float(query_ce.detach()) * 2
            totals["query_consistency"] += float(query_consistency.detach()) * 2
            total_correct += (
                (support_anchor.argmax(dim=1) == support_labels).sum().item()
                + (query_anchor.argmax(dim=1) == query_labels).sum().item()
            )
            total_samples += batch_size
            progress.set_postfix(
                outer=f"{float(outer_loss.detach()):.4f}",
                support=f"{float(support_loss.detach()):.4f}",
                query=f"{float(query_loss.detach()):.4f}",
            )
    finally:
        capture.close()
    return {
        "loss": totals["outer_loss"] / total_samples,
        "support_loss": totals["support_loss"] / (len(loader) * 6),
        "support_classification_loss": totals["support_ce"] / (len(loader) * 6),
        "support_consistency_loss": totals["support_consistency"] / (len(loader) * 6),
        "query_loss": totals["query_loss"] / (len(loader) * 2),
        "query_classification_loss": totals["query_ce"] / (len(loader) * 2),
        "query_consistency_loss": totals["query_consistency"] / (len(loader) * 2),
        "accuracy": total_correct / total_samples,
    }


def write_history(path: Path, history: list[dict]) -> None:
    fields = [
        "epoch", "learning_rate", "train_loss", "train_accuracy",
        "support_loss", "support_classification_loss", "support_consistency_loss",
        "query_loss", "query_classification_loss", "query_consistency_loss",
        "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in history:
            train = row["train"]
            val = row["val"]
            writer.writerow({
                "epoch": row["epoch"],
                "learning_rate": row["learning_rate"],
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "support_loss": train["support_loss"],
                "support_classification_loss": train["support_classification_loss"],
                "support_consistency_loss": train["support_consistency_loss"],
                "query_loss": train["query_loss"],
                "query_classification_loss": train["query_classification_loss"],
                "query_consistency_loss": train["query_consistency_loss"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_f1": val["f1"],
                "val_roc_auc": val["roc_auc"],
            })


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
    family_protocol, condition = load_condition(args.conditions_config, CONDITION_ID)
    validate_manifest(
        args.manifest,
        condition["seen_fake_methods"],
        family_protocol.get("budgets", FIXED_BUDGETS),
    )
    manifest_hash = sha256_file(args.manifest)
    if manifest_hash != protocol["manifest_sha256"][CONDITION_ID]:
        raise ValueError("M7 manifest hash mismatch.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)

    output_dir = args.output_root / run_name()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        last = output_dir / "last.pt"
        if not last.is_file():
            raise FileExistsError(f"Non-empty run has no last.pt: {output_dir}")
        args.resume = last
        print(f"Auto-resuming: {last}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    device = torch.device("cuda")
    manifest, train_frame, val_frame = load_and_validate_manifest(
        args.manifest, args.data_root
    )
    _, class_weights = calculate_class_weights(train_frame)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = timm.create_model(
        "xception", pretrained=args.resume is None, num_classes=len(LABEL_MAP)
    )
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    if tuple(data_config["input_size"]) != (3, 299, 299):
        raise RuntimeError(f"Unexpected Xception input: {data_config['input_size']}")
    preprocessing = protocol["preprocessing"]
    train_dataset = EpisodicPairedJpegDataset(
        train_frame,
        args.data_root,
        data_config,
        canonical_size=preprocessing["canonical_size"],
        qualities=preprocessing["train_jpeg_qualities"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    episode_sampler = MethodEpisodicBatchSampler(
        train_frame, protocol["seen_methods"], args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=episode_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    validation_transform = canonical_reencode_transform(
        data_config,
        canonical_size=preprocessing["canonical_size"],
        jpeg_quality=preprocessing["validation_jpeg_quality"],
        jpeg_subsampling=preprocessing["jpeg_subsampling"],
        jpeg_optimize=preprocessing["jpeg_optimize"],
        jpeg_progressive=preprocessing["jpeg_progressive"],
    )
    val_loader = DataLoader(
        FFPPDataset(val_frame, args.data_root, validation_transform),
        batch_size=args.validation_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    objective = protocol["objective"]
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
        "condition_name": "m7_fs_fr_efs_method_episodic_quality_invariant",
        "preprocessing_name": PREPROCESSING_NAME,
        "preprocessing": preprocessing,
        "episode": protocol["episode"],
        "objective": objective,
        "optimization": protocol["optimization"],
        "manifest_sha256": manifest_hash,
        "label_map": LABEL_MAP,
        "class_weights": {"real": float(class_weights[0]), "fake": float(class_weights[1])},
        "train_images": len(train_frame),
        "train_views_per_image": 2,
        "val_images": len(val_frame),
        "test_images_held_out": int((manifest["split"] == "test").sum()),
        "data_config": data_config,
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
    no_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved = checkpoint["config"]
        for key, expected in {
            "experiment_family": EXPERIMENT_FAMILY,
            "manifest_sha256": manifest_hash,
            "seed": args.seed,
            "preprocessing": preprocessing,
            "episode": protocol["episode"],
            "objective": objective,
        }.items():
            if saved.get(key) != expected:
                raise ValueError(f"Resume mismatch for {key}.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_auc = float(checkpoint.get("best_val_auc", float("-inf")))
        no_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        if no_improvement >= args.patience:
            print(f"Pilot already stopped at epoch {checkpoint['epoch']}.", flush=True)
            return

    print(json.dumps(json_ready(config), ensure_ascii=False, indent=2))
    print("Train distribution:", Counter(train_frame["label"]), flush=True)
    print("Validation distribution:", Counter(val_frame["label"]), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            episode_sampler,
            criterion,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            inner_step_size=float(objective["inner_step_size"]),
            meta_query_weight=float(objective["meta_query_weight"]),
            consistency_weight=float(objective["consistency_weight"]),
            accumulation_steps=int(protocol["optimization"]["gradient_accumulation_steps"]),
            persistent_progress=args.persistent_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            persistent_progress=args.persistent_progress,
        )
        result = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(result)
        improved = val_metrics["roc_auc"] > best_auc + args.min_delta
        if improved:
            best_auc = val_metrics["roc_auc"]
            no_improvement = 0
        else:
            no_improvement += 1
        payload = checkpoint_payload(
            epoch,
            model,
            optimizer,
            scaler,
            train_metrics,
            val_metrics,
            history,
            best_auc,
            no_improvement,
            config,
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, output_dir / "best.pt")
        write_json(output_dir / "history.json", history)
        write_history(output_dir / "history.csv", history)
        print(
            f"Train total={train_metrics['loss']:.4f}, "
            f"support={train_metrics['support_loss']:.4f}, "
            f"query={train_metrics['query_loss']:.4f}, "
            f"accuracy={train_metrics['accuracy']:.4f}",
            flush=True,
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}, "
            f"accuracy={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, AUC={val_metrics['roc_auc']:.4f}",
            flush=True,
        )
        print("Confusion matrix:", val_metrics["confusion_matrix"], flush=True)
        print(f"Early stopping: {no_improvement}/{args.patience}", flush=True)
        if no_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break
    print("\nMethod-episodic pilot complete.", flush=True)
    print("Best validation AUC:", best_auc, flush=True)
    print("Run directory:", output_dir, flush=True)


if __name__ == "__main__":
    main()
