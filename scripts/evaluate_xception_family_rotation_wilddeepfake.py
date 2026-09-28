#!/usr/bin/env python3
"""Evaluate a rotation selection on native WildDeepfake, reusing sequence logic."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", choices=[f"S{i}" for i in range(1, 7)], required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--protocols", nargs="+", choices=["fixed_q95", "mixed_jpeg"], default=["fixed_q95", "mixed_jpeg"])
    p.add_argument("--models", nargs="+", choices=[f"M{i}" for i in range(1, 8)], default=[f"M{i}" for i in range(1, 8)])
    for name in ("data-root", "manifest", "rotation-manifest-root", "s1-manifest-root", "rotation-runs-root", "s1-fixed-runs-root", "s1-mixed-runs-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--reuse-s1-root", type=Path)
    p.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    p.add_argument("--wild-config", type=Path, default=root / "configs/wilddeepfake_evaluation_v1/protocol.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    for values in (args.seeds, args.protocols, args.models):
        if len(values) != len(set(values)):
            p.error("Seeds, protocols and models must not contain duplicates")
    return args


def reuse_s1(args, protocol, model, seed, test, checkpoint_hash, manifest_hash, result_checker):
    if args.selection != "S1" or args.reuse_s1_root is None:
        return
    target = args.output_root / protocol / model.lower() / f"seed{seed}"
    if target.exists() and any(target.iterdir()):
        return
    source = args.reuse_s1_root / protocol / model.lower() / f"seed{seed}"
    metrics_path = source / "metrics.json"
    if not metrics_path.is_file():
        return
    metadata = json.loads(metrics_path.read_text())
    if (
        metadata.get("checkpoint_sha256") != checkpoint_hash
        or metadata.get("test_manifest_sha256") != manifest_hash
        or metadata.get("evaluation_preprocessing") != "native_letterbox299"
    ):
        print(f"Existing S1 result differs; evaluating current checkpoint: {protocol}/{model}/{seed}", flush=True)
        return
    complete, _, _ = result_checker(source, test.sample_id.tolist(), test.sequence_uid.nunique(), checkpoint_hash, manifest_hash)
    if complete:
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("frame_predictions.csv", "sequence_predictions.csv", "metrics.json"):
            shutil.copy2(source / filename, target / filename)
        print(f"Reused verified S1 Wild result: {protocol}/{model}/seed{seed}", flush=True)


def main():
    args = parse_args()
    import pandas as pd
    import torch
    from evaluate_xception_family_rotation import checkpoint_path, manifest_path, methods_for
    from evaluate_xception_wilddeepfake import (
        aggregate_seeds, atomic_csv, build_loaders, evaluate_one, result_is_complete,
        summary_row, training_overlap_audit, validate_manifest,
    )
    from train_baseline import sha256_file, write_json
    config = json.loads(args.config.read_text())
    wild_config = json.loads(args.wild_config.read_text())
    test = validate_manifest(args.manifest, args.data_root, wild_config)
    test_hash = sha256_file(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, loaders, reference_config = [], None, None
    audits = {}
    for model in args.models:
        manifest = manifest_path(args, model)
        training = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        actual_methods = set(training.loc[training.label == "fake", "method"])
        if actual_methods != set(methods_for(config, args.selection, model)):
            raise RuntimeError(f"Wrong selection/model training methods: {manifest}")
        audits[model] = training_overlap_audit(test, training)
        manifest_hash = sha256_file(manifest)
        for protocol in args.protocols:
            for seed in args.seeds:
                args.seed = seed
                path = checkpoint_path(args, config, protocol, model)
                print(f"Preparing WildDeepfake: {args.selection} {protocol} {model} seed{seed}", flush=True)
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                saved = checkpoint["config"]
                if saved.get("manifest_sha256") != manifest_hash or int(saved.get("seed", -1)) != seed:
                    raise RuntimeError(f"Checkpoint/manifest/seed mismatch: {path}")
                if checkpoint.get("label_map") != {"real": 0, "fake": 1}:
                    raise RuntimeError(f"Unexpected label map: {path}")
                data_config = saved["data_config"]
                if reference_config is None:
                    reference_config = data_config
                    loaders = build_loaders(test, args.data_root, data_config, args.batch_size, args.workers)
                elif data_config != reference_config:
                    raise RuntimeError(f"Xception input configuration differs: {path}")
                info = {
                    "path": str(path), "sha256": sha256_file(path),
                    "epoch": int(checkpoint["epoch"]),
                    "best_validation_auc": float(checkpoint["best_val_auc"]),
                }
                reuse_s1(args, protocol, model, seed, test, info["sha256"], test_hash, result_is_complete)
                frames, sequences = evaluate_one(
                    protocol, model, seed, {"preprocessing": "native_letterbox299"},
                    checkpoint, info, loaders["native_letterbox299"], test,
                    args.manifest, args.output_root, device,
                )
                row = summary_row(protocol, model, config["models"][model], seed, "native_letterbox299", frames, sequences)
                row["selection"] = args.selection
                rows.append(row)
                atomic_csv(pd.DataFrame(rows), args.output_root / "model_seed_summary.csv")
                del checkpoint
    summaries = pd.DataFrame(rows)
    aggregated = aggregate_seeds(summaries)
    aggregated.insert(0, "selection", args.selection)
    atomic_csv(aggregated, args.output_root / "seed_aggregate_summary.csv")
    write_json(args.output_root / "evaluation_summary.json", {
        "selection": args.selection, "training_seeds": args.seeds,
        "models": args.models, "training_protocols": args.protocols,
        "test_manifest_sha256": test_hash, "frames": len(test),
        "sequences": int(test.sequence_uid.nunique()),
        "evaluation_preprocessing": "native_letterbox299",
        "sequence_score": "mean of 16 frame fake probabilities",
        "decision_threshold": 0.5, "external_exploratory_evaluation": True,
        "test_used_for_training_or_tuning": False, "training_overlap_audits": audits,
    })
    print(aggregated[["selection", "protocol", "model", "training_seeds", "sequence_roc_auc_mean", "sequence_roc_auc_std"]].to_string(index=False), flush=True)
    print("Saved WildDeepfake results:", args.output_root, flush=True)


if __name__ == "__main__":
    main()
