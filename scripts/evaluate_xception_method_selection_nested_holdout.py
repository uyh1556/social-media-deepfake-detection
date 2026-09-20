#!/usr/bin/env python3
"""Evaluate nested method-holdout models without touching protected unseen data."""

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

from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
from evaluate_xception_family_coverage import (
    cross_split_audit,
    read_manifest,
    validate_test_manifest,
)
from evaluate_xception_method_transfer import metrics_from_arrays
from train_baseline import sha256_file, write_json
from xception_preprocessing import (
    MIXED_JPEG_REENCODE_NAME,
    canonical_reencode_transform,
)


PROTOCOL = "method_selection_nested_holdout_v1"
EXPECTED_PER_CLASS = 2_000


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=root / f"configs/{PROTOCOL}/protocol.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol in {path}")
    return config


def run_name(outer_slug: str, run_key: str, seed: int) -> str:
    return (
        f"xception_nested_holdout_{outer_slug}_{run_key}_"
        "canonical256_jpegmix75_80_85_90_95_letterbox299_v1_"
        f"seed{seed}"
    )


def validate_checkpoint(
    checkpoint: dict,
    checkpoint_path: Path,
    manifest_path: Path,
    outer_slug: str,
    run_key: str,
    seed: int,
    config: dict,
) -> dict:
    saved = checkpoint["config"]
    manifest_hash = sha256_file(manifest_path)
    if saved.get("experiment_family") != PROTOCOL:
        raise ValueError(f"Unexpected experiment family: {checkpoint_path}")
    if saved.get("split_protocol") != PROTOCOL:
        raise ValueError(f"Unexpected split protocol: {checkpoint_path}")
    expected_condition = f"holdout_{outer_slug}_{run_key}_jpeg_mixed"
    if saved.get("condition_name") != expected_condition:
        raise ValueError(f"Checkpoint condition mismatch: {checkpoint_path}")
    if saved.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Checkpoint/manifest mismatch: {checkpoint_path}")
    if saved.get("preprocessing_name") != MIXED_JPEG_REENCODE_NAME:
        raise ValueError(f"Unexpected preprocessing: {checkpoint_path}")
    if int(saved.get("seed", -1)) != seed:
        raise ValueError(f"Unexpected seed: {checkpoint_path}")
    expected = config["preprocessing"]
    actual = saved.get("preprocessing", {})
    for key in [
        "canonical_size", "train_jpeg_qualities", "train_jpeg_sampling",
        "validation_jpeg_quality", "jpeg_subsampling", "jpeg_optimize",
        "jpeg_progressive",
    ]:
        if actual.get(key) != expected.get(key):
            raise ValueError(f"Preprocessing mismatch for {key}: {checkpoint_path}")
    return {
        "path": checkpoint_path,
        "sha256": sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_auc": float(checkpoint["best_val_auc"]),
        "data_config": saved["data_config"],
    }


def prediction_frame(
    frame: pd.DataFrame,
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
) -> pd.DataFrame:
    columns = [
        "sample_id", "source_path", "label", "method", "family",
        "group_id", "video_id",
    ]
    result = frame[columns].copy()
    result["true_class"] = labels
    result["predicted_class"] = predictions
    result["fake_probability"] = probabilities
    result["correct"] = np.asarray(labels) == np.asarray(predictions)
    return result


def build_report(summary: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "# Nested Method-Holdout Evaluation",
        "",
        (
            "> Each outer method is excluded from transfer-based selection, "
            "training, validation, and checkpoint selection. Protected unseen "
            "methods and WildDeepfake are not used."
        ),
        "",
        "## Strategy summary",
        "",
        "| Condition | Strategy | Mean AUC | Worst AUC | Mean Real FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.strategy} | "
            f"{row.mean_auc:.4f} | {row.worst_auc:.4f} | "
            f"{row.mean_real_fpr:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Proposed transfer3 paired differences",
            "",
            "| Condition | Baseline | Mean AUC difference | Wins / 6 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.evaluation_condition} | {row.baseline_strategy} | "
            f"{row.mean_auc_difference:+.4f} | {row.transfer3_wins} / "
            f"{row.outer_holdouts} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "This is an internal six-method nested-holdout pilot at one "
                "training seed. It can validate the selection rule before that "
                "rule is frozen, but it is not the final protected-unseen result."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "runs_root": args.runs_root.resolve(),
        "output_root": args.output_root.resolve(),
        "protocol_config": args.protocol_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_root" and not path.exists():
            raise FileNotFoundError(path)
    config = load_config(paths["protocol_config"])
    if args.seed not in config["training_seeds"]:
        raise ValueError(f"Seed not enabled: {args.seed}")

    plan_path = paths["training_manifest_dir"] / "selection_plan.csv"
    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False)
    expected_pairs = {
        (outer, strategy)
        for outer in config["methods"]
        for strategy in config["strategies"]
    }
    actual_pairs = set(zip(plan["outer_holdout"], plan["strategy"]))
    if actual_pairs != expected_pairs:
        raise RuntimeError("Nested selection plan is incomplete")

    full_test = read_manifest(paths["test_manifest"])
    validate_test_manifest(full_test)
    test_hash = sha256_file(paths["test_manifest"])
    if test_hash != config["test_manifest_sha256"]:
        raise ValueError("Test manifest hash mismatch")
    missing_test = [
        value for value in full_test["source_path"]
        if not (paths["data_root"] / value).is_file()
    ]
    if missing_test:
        raise FileNotFoundError(
            "Test archive incomplete. First missing paths: "
            + ", ".join(missing_test[:5])
        )

    unique_runs = plan.drop_duplicates(["outer_holdout", "run_key"])
    checkpoints = {}
    identity_checkpoints = {}
    reference_data_config = None
    for row in unique_runs.itertuples(index=False):
        outer_slug = config["methods"][row.outer_holdout]["slug"]
        manifest = paths["training_manifest_dir"] / row.manifest
        if sha256_file(manifest) != row.manifest_sha256:
            raise ValueError(f"Manifest hash mismatch: {manifest}")
        development = read_manifest(manifest)
        if row.outer_holdout in set(
            development.loc[development["label"] == "fake", "method"]
        ):
            raise RuntimeError(f"Outer leakage in {manifest}")
        cross_split_audit(development, full_test)
        checkpoint_path = (
            paths["runs_root"]
            / run_name(outer_slug, row.run_key, args.seed)
            / "best.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        info = validate_checkpoint(
            checkpoint, checkpoint_path, manifest, outer_slug,
            row.run_key, args.seed, config,
        )
        if reference_data_config is None:
            reference_data_config = info["data_config"]
        elif reference_data_config != info["data_config"]:
            raise RuntimeError("Nested checkpoints use different model inputs")
        key = (row.outer_holdout, row.run_key)
        checkpoints[key] = info
        identity_checkpoints[f"{row.outer_holdout}|{row.run_key}"] = {
            "sha256": info["sha256"],
            "epoch": info["epoch"],
            "manifest_sha256": row.manifest_sha256,
        }

    identity = {
        "protocol": PROTOCOL,
        "protocol_config_sha256": sha256_file(paths["protocol_config"]),
        "selection_plan_sha256": sha256_file(plan_path),
        "test_manifest_sha256": test_hash,
        "seed": args.seed,
        "checkpoints": identity_checkpoints,
        "protected_unseen_used": False,
        "wilddeepfake_used": False,
    }
    output_root = paths["output_root"]
    identity_path = output_root / "run_identity.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not identity_path.is_file():
            raise FileExistsError(f"Non-empty output has no identity: {output_root}")
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Existing evaluation belongs to another run")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required")
    preprocessing = config["preprocessing"]
    result_by_run = {}

    for row in unique_runs.itertuples(index=False):
        outer = row.outer_holdout
        key = (outer, row.run_key)
        test_frame = full_test[
            (full_test["method"] == "original") | (full_test["method"] == outer)
        ].copy().reset_index(drop=True)
        counts = test_frame.groupby("method").size().to_dict()
        if counts != {"original": EXPECTED_PER_CLASS, outer: EXPECTED_PER_CLASS}:
            raise RuntimeError(f"Unexpected outer-test counts for {outer}: {counts}")
        test_frame["resolved_path"] = test_frame["source_path"]
        test_frame["source_reference_path"] = test_frame["source_path"]

        info = checkpoints[key]
        checkpoint = torch.load(
            info["path"], map_location=device, weights_only=False
        )
        model = timm.create_model(
            checkpoint["model_name"], pretrained=False,
            num_classes=checkpoint["num_classes"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        saved_weights = checkpoint["config"].get("class_weights")
        weights = None
        if saved_weights is not None:
            weights = torch.tensor(
                [saved_weights["real"], saved_weights["fake"]],
                dtype=torch.float32, device=device,
            )
        criterion = nn.CrossEntropyLoss(weight=weights)

        for condition in config["evaluation_conditions"]:
            condition_name = condition["name"]
            result_dir = output_root / config["methods"][outer]["slug"] / row.run_key / condition_name
            predictions_path = result_dir / "predictions.csv"
            metrics_path = result_dir / "metrics.json"
            if predictions_path.is_file() and metrics_path.is_file():
                prediction = pd.read_csv(predictions_path)
                if prediction["sample_id"].tolist() != test_frame["sample_id"].tolist():
                    raise RuntimeError(f"Prediction order changed: {predictions_path}")
                print(f"Skipping completed: {outer}/{row.run_key}/{condition_name}", flush=True)
            elif result_dir.exists() and any(result_dir.iterdir()):
                raise FileExistsError(f"Incomplete result directory: {result_dir}")
            else:
                transform = canonical_reencode_transform(
                    reference_data_config,
                    canonical_size=preprocessing["canonical_size"],
                    jpeg_quality=condition["jpeg_quality"],
                    jpeg_subsampling=preprocessing["jpeg_subsampling"],
                    jpeg_optimize=preprocessing["jpeg_optimize"],
                    jpeg_progressive=preprocessing["jpeg_progressive"],
                )
                dataset = EvaluationDataset(
                    test_frame, paths["data_root"], transform,
                    MIXED_JPEG_REENCODE_NAME,
                    reference_data_config["input_size"][1],
                )
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=args.workers > 0,
                )
                print(
                    f"\n=== Holdout {outer} / {row.run_key} / {condition_name} ===",
                    flush=True,
                )
                _, labels, predictions, probabilities = evaluate_with_predictions(
                    model, loader, test_frame, criterion, device,
                    progress_desc=f"{outer} {row.run_key} {condition_name}",
                    persistent_progress=True,
                )
                prediction = prediction_frame(
                    test_frame, labels, predictions, probabilities
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                temporary = predictions_path.with_suffix(".csv.tmp")
                prediction.to_csv(temporary, index=False)
                temporary.replace(predictions_path)
                metrics = metrics_from_arrays(
                    np.asarray(labels), np.asarray(predictions),
                    np.asarray(probabilities),
                )
                write_json(
                    metrics_path,
                    {
                        "outer_holdout": outer,
                        "run_key": row.run_key,
                        "evaluation_condition": condition,
                        "checkpoint": str(info["path"]),
                        "checkpoint_sha256": info["sha256"],
                        "test_manifest_sha256": test_hash,
                        "metrics": metrics,
                        "protected_unseen_used": False,
                        "wilddeepfake_used": False,
                    },
                )

            metrics = metrics_from_arrays(
                prediction["true_class"].to_numpy(int),
                prediction["predicted_class"].to_numpy(int),
                prediction["fake_probability"].to_numpy(float),
            )
            result_by_run[(outer, row.run_key, condition_name)] = metrics
        del model, checkpoint
        torch.cuda.empty_cache()

    rows = []
    for row in plan.itertuples(index=False):
        for condition in config["evaluation_conditions"]:
            metrics = result_by_run[(
                row.outer_holdout, row.run_key, condition["name"]
            )]
            rows.append(
                {
                    "outer_holdout": row.outer_holdout,
                    "outer_family": row.outer_family,
                    "strategy": row.strategy,
                    "selected_methods": row.selected_methods,
                    "selected_families": row.selected_families,
                    "run_key": row.run_key,
                    "evaluation_condition": condition["name"],
                    **metrics,
                }
            )
    long = pd.DataFrame(rows).sort_values(
        ["evaluation_condition", "outer_holdout", "strategy"]
    )
    long.to_csv(output_root / "nested_holdout_results.csv", index=False)
    summary = (
        long.groupby(["evaluation_condition", "strategy"])
        .agg(
            mean_auc=("roc_auc", "mean"),
            median_auc=("roc_auc", "median"),
            worst_auc=("roc_auc", "min"),
            mean_auprc=("auprc", "mean"),
            mean_real_fpr=("real_fpr", "mean"),
            mean_fake_recall=("fake_recall", "mean"),
        )
        .reset_index()
        .sort_values(["evaluation_condition", "mean_auc"], ascending=[True, False])
    )
    summary.to_csv(output_root / "strategy_summary.csv", index=False)

    comparisons = []
    for condition in [item["name"] for item in config["evaluation_conditions"]]:
        condition_frame = long[long["evaluation_condition"] == condition]
        pivot = condition_frame.pivot(
            index="outer_holdout", columns="strategy", values="roc_auc"
        )
        for baseline in ["random3", "family_balanced3", "all5"]:
            differences = pivot["transfer3"] - pivot[baseline]
            comparisons.append(
                {
                    "evaluation_condition": condition,
                    "baseline_strategy": baseline,
                    "outer_holdouts": int(len(differences)),
                    "mean_auc_difference": float(differences.mean()),
                    "median_auc_difference": float(differences.median()),
                    "minimum_auc_difference": float(differences.min()),
                    "maximum_auc_difference": float(differences.max()),
                    "transfer3_wins": int((differences > 0).sum()),
                    "ties": int((differences == 0).sum()),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        output_root / "transfer3_comparisons.csv", index=False
    )
    (output_root / "REPORT.md").write_text(
        build_report(summary, comparison_frame), encoding="utf-8"
    )
    write_json(
        output_root / "evaluation_summary.json",
        {
            "protocol": PROTOCOL,
            "seed": args.seed,
            "outer_holdouts": list(config["methods"]),
            "strategies": list(config["strategies"]),
            "evaluation_conditions": [
                item["name"] for item in config["evaluation_conditions"]
            ],
            "protected_unseen_used": False,
            "wilddeepfake_used": False,
            "outputs": {
                "long": "nested_holdout_results.csv",
                "strategy_summary": "strategy_summary.csv",
                "comparisons": "transfer3_comparisons.csv",
                "report": "REPORT.md",
            },
        },
    )
    print(summary.to_string(index=False), flush=True)
    print("Saved:", output_root, flush=True)


if __name__ == "__main__":
    main()
