#!/usr/bin/env python3
"""Evaluate one selection's fixed-Q95 and Mixed-JPEG M1-M7 on 21 groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions, write_predictions
from evaluate_xception_family_coverage import cross_split_audit
from train_baseline import sha256_file, write_json
from xception_preprocessing import LETTERBOX_NAME, evaluation_transform_from_checkpoint


MODELS = tuple(f"M{i}" for i in range(1, 8))
PROTOCOLS = ("fixed_q95", "mixed_jpeg")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=[f"S{i}" for i in range(1, 7)], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--df40-test-manifest", type=Path, required=True)
    parser.add_argument("--ffpp-test-manifest", type=Path, required=True)
    parser.add_argument("--rotation-manifest-root", type=Path, required=True)
    parser.add_argument("--s1-manifest-root", type=Path, required=True)
    parser.add_argument("--rotation-runs-root", type=Path, required=True)
    parser.add_argument("--s1-fixed-runs-root", type=Path, required=True)
    parser.add_argument("--s1-mixed-runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=root / "configs/family_rotation_v1/selections.json",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def methods_for(config: dict, selection: str, model: str) -> list[str]:
    return [
        config["families"][family][letter]
        for family in config["models"][model]["families"]
        for letter in config["selections"][selection]
    ]


def manifest_path(args: argparse.Namespace, model: str) -> Path:
    if args.selection == "S1":
        return args.s1_manifest_root / f"{model.lower()}_seed42.csv"
    return (
        args.rotation_manifest_root / args.selection.lower()
        / f"{model.lower()}_seed42.csv"
    )


def checkpoint_path(
    args: argparse.Namespace, config: dict, protocol: str, model: str
) -> Path:
    name = config["models"][model]["name"]
    if args.selection == "S1":
        if protocol == "fixed_q95":
            run = (
                f"xception_{model.lower()}_{name}_"
                "canonical256_jpegq95_letterbox299_control_v1_"
                f"seed{args.seed}"
            )
            return args.s1_fixed_runs_root / run / "best.pt"
        run = (
            f"xception_{model.lower()}_{name}_"
            "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
            f"seed{args.seed}"
        )
        return args.s1_mixed_runs_root / run / "best.pt"
    run = (
        f"xception_{args.selection.lower()}_{model.lower()}_{name}_"
        f"{protocol}_family_rotation_v1_seed{args.seed}"
    )
    return args.rotation_runs_root / run / "best.pt"


def method_rows(
    selection: str,
    protocol: str,
    model: str,
    trained: set[str],
    test: pd.DataFrame,
    metrics: dict,
) -> list[dict]:
    rows = []
    for method, group in test[test["label"] == "fake"].groupby("method"):
        pair = metrics["by_manipulation"][f"original_vs_{method}"]
        cm = pair["confusion_matrix"]
        rows.append({
            "selection": selection,
            "protocol": protocol,
            "model": model,
            "method": method,
            "family": group["family"].iloc[0],
            "status": (
                "trained" if method in trained
                else "ffpp_reference" if method in {"Deepfakes", "Face2Face"}
                else "unseen"
            ),
            "roc_auc": float(pair["roc_auc"]),
            "f1": float(pair["f1"]),
            "fake_recall": float(pair["recall"]),
            "real_fpr": float(cm[0][1] / sum(cm[0])),
        })
    return rows


def mean(rows: list[dict], status: str | None = None) -> float:
    values = [r["roc_auc"] for r in rows if status is None or r["status"] == status]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    test = pd.concat(
        [
            pd.read_csv(args.df40_test_manifest, dtype=str, keep_default_na=False),
            pd.read_csv(args.ffpp_test_manifest, dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    )
    counts = test.groupby("method").size()
    if len(test) != 42000 or len(counts) != 21 or set(counts) != {
        "original", "Deepfakes", "Face2Face", *config["families"]["FS"].values(),
        *config["families"]["FR"].values(), *config["families"]["EFS"].values(),
    } or set(counts.tolist()) != {2000}:
        raise RuntimeError(f"Unexpected 21-group test manifest: {counts.to_dict()}")
    test["source_reference_path"] = test["source_path"]
    test["resolved_path"] = test["source_path"]
    missing = [p for p in test["source_path"] if not (args.data_root / p).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_methods, models = [], []
    test_hash = hashlib_pair(args.df40_test_manifest, args.ffpp_test_manifest)

    for protocol in args.protocols:
        for model_id in args.models:
            trained = set(methods_for(config, args.selection, model_id))
            training_manifest = manifest_path(args, model_id)
            checkpoint_file = checkpoint_path(args, config, protocol, model_id)
            for path in (training_manifest, checkpoint_file):
                if not path.is_file():
                    raise FileNotFoundError(path)
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
            if checkpoint["config"].get("manifest_sha256") != sha256_file(training_manifest):
                raise RuntimeError(f"Checkpoint/manifest mismatch: {checkpoint_file}")
            audit = cross_split_audit(
                pd.read_csv(training_manifest, dtype=str, keep_default_na=False), test
            )
            out = args.output_root / protocol / model_id.lower()
            metrics_path, predictions_path = out / "metrics.json", out / "predictions.csv"
            if metrics_path.is_file() and predictions_path.is_file():
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                if result["test_manifest_sha256"] != test_hash:
                    raise RuntimeError(f"Existing result used another test set: {out}")
                metrics = result["metrics"]
            else:
                if out.exists() and any(out.iterdir()):
                    raise FileExistsError(out)
                transform = evaluation_transform_from_checkpoint(checkpoint)
                dataset = EvaluationDataset(
                    test, args.data_root, transform, LETTERBOX_NAME,
                    checkpoint["config"]["data_config"]["input_size"][1],
                )
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0,
                )
                network = timm.create_model(
                    checkpoint["model_name"], pretrained=False,
                    num_classes=checkpoint["num_classes"],
                )
                network.load_state_dict(checkpoint["model_state_dict"])
                network = network.to(device)
                metrics, labels, predictions, probabilities = evaluate_with_predictions(
                    network, loader, test, nn.CrossEntropyLoss(), device,
                    progress_desc=f"{args.selection} {protocol} {model_id}",
                    persistent_progress=True,
                )
                out.mkdir(parents=True, exist_ok=True)
                write_predictions(predictions_path, test, labels, predictions, probabilities)
                result = {
                    "selection": args.selection, "protocol": protocol,
                    "model": model_id, "trained_methods": sorted(trained),
                    "checkpoint": str(checkpoint_file), "cross_split_audit": audit,
                    "test_manifest_sha256": test_hash, "test_images": len(test),
                    "metrics": metrics,
                }
                write_json(metrics_path, result)
                del network, loader, dataset
                torch.cuda.empty_cache()
            rows = method_rows(args.selection, protocol, model_id, trained, test, metrics)
            all_methods.extend(rows)
            cm = metrics["confusion_matrix"]
            models.append({
                "selection": args.selection, "protocol": protocol, "model": model_id,
                "trained_methods": "|".join(sorted(trained)),
                "real_fpr": float(cm[0][1] / sum(cm[0])),
                "df40_all_macro_auc": mean([r for r in rows if r["status"] != "ffpp_reference"]),
                "df40_seen_macro_auc": mean(rows, "trained"),
                "df40_unseen_macro_auc": mean(rows, "unseen"),
                "ffpp_reference_macro_auc": mean(rows, "ffpp_reference"),
            })
    pd.DataFrame(models).to_csv(args.output_root / "model_summary.csv", index=False)
    pd.DataFrame(all_methods).to_csv(args.output_root / "method_summary.csv", index=False)
    write_json(args.output_root / "evaluation_summary.json", {
        "protocol": "family_rotation_21_group_evaluation_v1",
        "selection": args.selection, "seed": args.seed,
        "test_images": len(test), "groups": 21,
        "models": list(args.models), "training_protocols": list(args.protocols),
        "test_manifest_sha256": test_hash,
    })


def hashlib_pair(first: Path, second: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    digest.update(sha256_file(first).encode())
    digest.update(sha256_file(second).encode())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
