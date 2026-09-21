#!/usr/bin/env python3
"""Cache non-self expert targets for the M7 transfer-aware KD pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluate_checkpoint import EvaluationDataset
from evaluate_xception_method_transfer import run_name, validate_checkpoint
from train_baseline import LABEL_MAP, seed_worker, sha256_file, write_json
from xception_preprocessing import (
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
)


PROTOCOL = "transfer_aware_distillation_pilot_v1"
PARENT_PROTOCOL = "method_transfer_matrix_v1"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--student-manifest", type=Path, required=True)
    parser.add_argument("--expert-manifest-dir", type=Path, required=True)
    parser.add_argument("--expert-runs-root", type=Path, required=True)
    parser.add_argument("--transfer-evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pilot-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument(
        "--expert-config",
        type=Path,
        default=root / f"configs/{PARENT_PROTOCOL}/protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path, expected_protocol: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != expected_protocol:
        raise ValueError(f"Unexpected protocol in {path}")
    return value


def load_transfer_weights(
    path: Path, pilot: dict
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    methods = list(pilot["methods"])
    conditions = pilot["transfer_weighting"]["conditions"]
    expected = {
        (source, target, condition)
        for source in methods
        for target in methods
        for condition in conditions
    }
    actual = set(
        zip(
            frame["training_method"],
            frame["target_method"],
            frame["evaluation_condition"],
        )
    )
    if actual != expected or len(frame) != len(expected):
        raise RuntimeError("Transfer matrix is incomplete or incompatible")
    result = {}
    for target in methods:
        utilities = {}
        for source in methods:
            if source == target:
                continue
            rows = frame[
                (frame["training_method"] == source)
                & (frame["target_method"] == target)
                & (frame["evaluation_condition"].isin(conditions))
            ]
            if len(rows) != len(conditions):
                raise RuntimeError(f"Missing transfer rows: {source}->{target}")
            utilities[source] = float(
                np.maximum(rows["roc_auc"].to_numpy(float) - 0.5, 0.0).mean()
            )
        total = sum(utilities.values())
        if total <= 0:
            raise RuntimeError(f"No positive transfer utility for {target}")
        result[target] = {
            source: utility / total for source, utility in utilities.items()
        }
    return frame, result


@torch.no_grad()
def predict_fake_probability(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> np.ndarray:
    model.eval()
    values = []
    indices = []
    for batch in tqdm(loader, desc=description, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        values.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        indices.extend(batch["index"].tolist())
    if indices != list(range(len(loader.dataset))):
        raise RuntimeError("Teacher inference order changed")
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "student_manifest": args.student_manifest.resolve(),
        "expert_manifest_dir": args.expert_manifest_dir.resolve(),
        "expert_runs_root": args.expert_runs_root.resolve(),
        "transfer_evaluation_dir": args.transfer_evaluation_dir.resolve(),
        "output_dir": args.output_dir.resolve(),
        "pilot_config": args.pilot_config.resolve(),
        "expert_config": args.expert_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)
    pilot = load_json(paths["pilot_config"], PROTOCOL)
    expert_protocol = load_json(paths["expert_config"], PARENT_PROTOCOL)
    if sha256_file(paths["student_manifest"]) != pilot["manifest_sha256"]:
        raise ValueError("M7 manifest hash mismatch")
    if list(pilot["methods"]) != list(expert_protocol["methods"]):
        raise ValueError("Pilot and expert method order differs")

    parent_summary = paths["transfer_evaluation_dir"] / "evaluation_summary.json"
    transfer_path = paths["transfer_evaluation_dir"] / "transfer_matrix_long.csv"
    parent = load_json(parent_summary, PARENT_PROTOCOL)
    if parent.get("protected_unseen_used") or parent.get("wilddeepfake_used"):
        raise ValueError("Final evaluation data leaked into transfer matrix")
    _, transfer_weights = load_transfer_weights(transfer_path, pilot)

    manifest = pd.read_csv(
        paths["student_manifest"], dtype=str, keep_default_na=False
    )
    fake_train = manifest[
        (manifest["split"] == pilot["teacher_cache"]["source_split"])
        & (manifest["label"] == pilot["teacher_cache"]["source_label"])
    ].copy().reset_index(drop=True)
    if set(fake_train["method"]) != set(pilot["methods"]):
        raise RuntimeError("Teacher cache does not contain all six methods")
    if fake_train["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate fake training sample IDs")
    missing = [
        value
        for value in fake_train["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing training images: " + ", ".join(missing[:5]))
    fake_train["resolved_path"] = fake_train["source_path"]

    checkpoint_info = {}
    checkpoints = {}
    data_config = None
    for method, definition in expert_protocol["methods"].items():
        manifest_path = (
            paths["expert_manifest_dir"] / f"{definition['slug']}_seed42.csv"
        )
        checkpoint_path = (
            paths["expert_runs_root"]
            / run_name(method, definition, 42)
            / "best.pt"
        )
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                manifest_path if not manifest_path.is_file() else checkpoint_path
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        info = validate_checkpoint(
            checkpoint,
            checkpoint_path,
            manifest_path,
            method,
            definition,
            expert_protocol,
        )
        if data_config is None:
            data_config = info["data_config"]
        elif data_config != info["data_config"]:
            raise RuntimeError("Expert input configurations differ")
        checkpoint_info[method] = {
            "checkpoint_sha256": info["sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        }
        checkpoints[method] = checkpoint_path

    identity = {
        "protocol": PROTOCOL,
        "pilot_config_sha256": sha256_file(paths["pilot_config"]),
        "student_manifest_sha256": sha256_file(paths["student_manifest"]),
        "parent_evaluation_summary_sha256": sha256_file(parent_summary),
        "transfer_matrix_sha256": sha256_file(transfer_path),
        "experts": checkpoint_info,
        "fake_train_images": int(len(fake_train)),
        "teacher_condition": pilot["teacher_cache"]["teacher_input_condition"],
    }
    output_dir = paths["output_dir"]
    identity_path = output_dir / "run_identity.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Output has no identity: {output_dir}")
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("Existing teacher cache belongs to another run")
        final_path = output_dir / "teacher_targets.csv"
        summary_path = output_dir / "cache_summary.json"
        if final_path.is_file() and summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("teacher_targets_sha256") != sha256_file(final_path):
                raise RuntimeError("Completed teacher cache hash mismatch")
            print("Teacher cache already complete:", final_path, flush=True)
            return
        if final_path.is_file() != summary_path.is_file():
            raise RuntimeError(
                "Teacher cache completion files are inconsistent; use a new "
                "output directory instead of overwriting the partial result"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    transform = canonical_reencode_transform(
        data_config,
        canonical_size=pilot["preprocessing"]["canonical_size"],
        jpeg_quality=95,
        jpeg_subsampling=pilot["preprocessing"]["jpeg_subsampling"],
        jpeg_optimize=pilot["preprocessing"]["jpeg_optimize"],
        jpeg_progressive=pilot["preprocessing"]["jpeg_progressive"],
    )
    dataset = EvaluationDataset(
        fake_train,
        paths["data_root"],
        transform,
        MIXED_JPEG_REENCODE_NAME,
        data_config["input_size"][1],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required to cache teacher predictions")
    cache = fake_train[["sample_id", "source_path", "method", "family"]].copy()
    for method, definition in pilot["methods"].items():
        partial_path = output_dir / f"expert_{definition['slug']}.csv"
        if partial_path.is_file():
            partial = pd.read_csv(partial_path)
            if partial["sample_id"].tolist() != cache["sample_id"].tolist():
                raise RuntimeError(f"Cached expert order mismatch: {partial_path}")
            probabilities = partial["fake_probability"].to_numpy(float)
            print("Reusing:", partial_path.name, flush=True)
        else:
            checkpoint = torch.load(
                checkpoints[method], map_location=device, weights_only=False
            )
            model = timm.create_model(
                checkpoint["model_name"],
                pretrained=False,
                num_classes=checkpoint["num_classes"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            probabilities = predict_fake_probability(
                model, loader, device, f"Teacher {method}"
            )
            temporary = partial_path.with_suffix(".csv.tmp")
            pd.DataFrame(
                {
                    "sample_id": cache["sample_id"],
                    "fake_probability": probabilities,
                }
            ).to_csv(temporary, index=False)
            temporary.replace(partial_path)
            del model, checkpoint
            torch.cuda.empty_cache()
        cache[f"expert_{definition['slug']}_fake_probability"] = probabilities

    lower, upper = pilot["teacher_cache"]["target_probability_clip"]
    uniform_targets = []
    transfer_targets = []
    for row in cache.itertuples(index=False):
        target = row.method
        experts = [method for method in pilot["methods"] if method != target]
        values = np.asarray(
            [
                getattr(
                    row,
                    f"expert_{pilot['methods'][method]['slug']}_fake_probability",
                )
                for method in experts
            ],
            dtype=float,
        )
        uniform_targets.append(float(values.mean()))
        transfer_targets.append(
            float(
                sum(
                    transfer_weights[target][method] * value
                    for method, value in zip(experts, values)
                )
            )
        )
    cache["uniform_teacher_fake_probability"] = np.clip(
        uniform_targets, lower, upper
    )
    cache["transfer_teacher_fake_probability"] = np.clip(
        transfer_targets, lower, upper
    )
    target_path = output_dir / "teacher_targets.csv"
    temporary = target_path.with_suffix(".csv.tmp")
    cache.to_csv(temporary, index=False)
    temporary.replace(target_path)
    weight_rows = []
    for target, weights in transfer_weights.items():
        for source, weight in weights.items():
            weight_rows.append(
                {
                    "target_method": target,
                    "teacher_method": source,
                    "transfer_weight": weight,
                    "uniform_weight": 0.2,
                }
            )
    pd.DataFrame(weight_rows).to_csv(
        output_dir / "teacher_weights.csv", index=False
    )
    write_json(
        output_dir / "cache_summary.json",
        {
            **identity,
            "teacher_targets_sha256": sha256_file(target_path),
            "uniform_target_mean": float(
                cache["uniform_teacher_fake_probability"].mean()
            ),
            "transfer_target_mean": float(
                cache["transfer_teacher_fake_probability"].mean()
            ),
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
        },
    )
    print("Saved:", target_path, flush=True)


if __name__ == "__main__":
    main()
