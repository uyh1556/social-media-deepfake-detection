#!/usr/bin/env python3
"""Evaluate the frozen S2-S4 M7 weak-noise pilots on native WildDeepfake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", nargs="+", choices=["S2", "S3", "S4"], default=["S2", "S3", "S4"])
    for name in ("data-root", "manifest", "training-manifest-root", "pilot-runs-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    parser.add_argument("--wild-config", type=Path, default=root / "configs/wilddeepfake_evaluation_v1/protocol.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--noise-max-std", type=int, choices=[2, 5], default=2)
    args = parser.parse_args()
    if len(args.selections) != len(set(args.selections)):
        parser.error("Selections must not contain duplicates")

    from evaluate_xception_family_rotation import methods_for
    from evaluate_xception_wilddeepfake import (
        atomic_csv, build_loaders, evaluate_one, summary_row,
        training_overlap_audit, validate_manifest,
    )
    from train_baseline import sha256_file, write_json

    config = json.loads(args.config.read_text())
    wild_config = json.loads(args.wild_config.read_text())
    test = validate_manifest(args.manifest, args.data_root, wild_config)
    test_hash = sha256_file(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required")
    rows = []
    reference_data_config = None
    loaders = None
    for selection in args.selections:
        training_manifest = args.training_manifest_root / selection.lower() / "m7_seed42.csv"
        if not training_manifest.is_file():
            raise FileNotFoundError(training_manifest)
        training = pd.read_csv(training_manifest, dtype=str, keep_default_na=False)
        expected_methods = set(methods_for(config, selection, "M7"))
        if set(training.loc[training.label == "fake", "method"]) != expected_methods:
            raise RuntimeError(f"Wrong M7 training methods: {training_manifest}")
        audit = training_overlap_audit(test, training)
        training_hash = sha256_file(training_manifest)
        checkpoint_path = (
            args.pilot_runs_root
            / f"xception_{selection.lower()}_m7_mixed_jpeg_noise_p25_std{args.noise_max_std}_seed42"
            / "best.pt"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        if saved.get("manifest_sha256") != training_hash or int(saved.get("seed", -1)) != 42:
            raise RuntimeError(f"Checkpoint/manifest/seed mismatch: {checkpoint_path}")
        noise = saved.get("preprocessing", {}).get("train_noise", {})
        if noise.get("probability") != 0.25 or noise.get("max_std_0_to_255") != args.noise_max_std:
            raise RuntimeError(f"Not the frozen weak-noise pilot: {checkpoint_path}")
        if checkpoint.get("label_map") != {"real": 0, "fake": 1}:
            raise RuntimeError(f"Unexpected label map: {checkpoint_path}")
        data_config = saved["data_config"]
        if reference_data_config is None:
            reference_data_config = data_config
            loaders = build_loaders(test, args.data_root, data_config, args.batch_size, args.workers)
        elif data_config != reference_data_config:
            raise RuntimeError(f"Different Xception input configuration: {checkpoint_path}")
        info = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "best_validation_auc": float(checkpoint["best_val_auc"]),
        }
        selection_output = args.output_root / selection.lower()
        print(
            f"\n===== WildDeepfake {selection} / M7 noise pilot / seed42 "
            f"({len(test)} frames, {test.sequence_uid.nunique()} sequences) =====",
            flush=True,
        )
        frames, sequences = evaluate_one(
            "mixed_jpeg_noise_pilot", "M7", 42,
            {"preprocessing": "native_letterbox299"}, checkpoint, info,
            loaders["native_letterbox299"], test, args.manifest,
            selection_output, device,
        )
        row = summary_row(
            "mixed_jpeg_noise_pilot", "M7",
            {"name": f"{selection.lower()}_m7_mixed_jpeg_noise_p25_std{args.noise_max_std}"},
            42, "native_letterbox299", frames, sequences,
        )
        row["selection"] = selection
        rows.append(row)
        atomic_csv(pd.DataFrame([row]), selection_output / "model_seed_summary.csv")
        write_json(selection_output / "evaluation_summary.json", {
            "selection": selection,
            "model": "M7",
            "training_seed": 42,
            "training_manifest_sha256": training_hash,
            "checkpoint_sha256": info["sha256"],
            "test_manifest_sha256": test_hash,
            "frames": int(len(test)),
            "sequences": int(test.sequence_uid.nunique()),
            "evaluation_preprocessing": "native_letterbox299",
            "sequence_score": "mean of 16 frame fake probabilities",
            "decision_threshold": 0.5,
            "external_exploratory_evaluation": True,
            "test_used_for_training_or_tuning": False,
            "training_overlap_audit": audit,
        })
        print(
            f"{selection}: sequence AUC={row['sequence_roc_auc']:.4f}, "
            f"Real FPR={row['sequence_real_fpr']:.2%}, "
            f"Fake recall={row['sequence_fake_recall']:.2%}",
            flush=True,
        )
        del checkpoint
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(rows), args.output_root / "model_seed_summary.csv")
    print("Saved WildDeepfake pilot results:", args.output_root, flush=True)


if __name__ == "__main__":
    main()
