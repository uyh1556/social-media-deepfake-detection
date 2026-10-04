#!/usr/bin/env python3
"""Matched uniform/complementarity M7 with Mixed-JPEG and optional std5 noise."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from family_rotation_adaptive_augmentation import (
    AUGMENTATIONS, CONFIG, EXPERIMENT, ROOT, STRATEGIES, adaptive_config_path,
    load_experiment, run_directory, validate_saved,
)
from prepare_family_rotation_adaptive_learning import load_roles, sha


def command_for(args, settings, selection, strategy, augmentation, manifest, roles):
    run = run_directory(args.output_root, selection, strategy, augmentation, args.seed)
    command = [sys.executable, "-u", str(ROOT / "scripts/train_xception_letterbox.py"),
        "--data-root", str(args.data_root), "--manifest", str(manifest), "--output-dir", str(run),
        "--split-protocol", EXPERIMENT, "--experiment-family", EXPERIMENT,
        "--condition-name", f"{selection.lower()}_m7_{augmentation}_{strategy}", "--model", "xception",
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--learning-rate", str(args.learning_rate), "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience), "--seed", str(args.seed),
        "--canonical-size", "256", "--jpeg-quality", "95", "--jpeg-subsampling", "2",
        "--train-jpeg-qualities", *map(str, settings["train_jpeg_qualities"]),
        "--train-noise-probability", str(settings["noise_probability"] if augmentation == "mixed_jpeg_noise" else 0),
        "--train-noise-max-std", str(settings["noise_max_std"]),
        "--adaptive-strategy", strategy, "--adaptive-protocol-config", str(adaptive_config_path(settings)),
        "--adaptive-validation-roles", str(roles), "--adaptive-augmentation-config", str(args.protocol_config),
        "--log-interval-seconds", str(args.log_interval_seconds)]
    command += ["--persistent-progress"] if args.verbose else ["--quiet"]
    if (run / "last.pt").exists():
        command += ["--resume", str(run / "last.pt")]
    return run, command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=[f"S{i}" for i in range(1, 7)])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--augmentations", nargs="+", choices=AUGMENTATIONS, default=list(AUGMENTATIONS))
    for name in ("data-root", "manifest-root", "roles-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-config", type=Path, default=CONFIG)
    for name, default in (("epochs", 15), ("batch-size", 16), ("workers", 2), ("patience", 3), ("seed", 42)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--log-interval-seconds", type=float, default=60)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if any(len(set(v)) != len(v) for v in (a.selections, a.strategies, a.augmentations)):
        p.error("Duplicate selections, strategies or augmentations")
    if a.seed != 42 or min(a.epochs, a.batch_size, a.patience) < 1 or a.workers < 0 or a.log_interval_seconds <= 0:
        p.error("Use seed42 with positive training settings")
    settings = load_experiment(a.protocol_config)
    protocol = json.loads(adaptive_config_path(settings).read_text())
    cfg = json.loads((ROOT / "configs/family_rotation_v1/selections.json").read_text())
    from train_xception_family_coverage import FIXED_BUDGETS, validate_manifest
    prepared = []
    for selection in a.selections:
        manifest = a.manifest_root / selection.lower() / "m7_seed42.csv"
        roles = a.roles_root / selection.lower() / "validation_roles.csv"
        load_roles(manifest, roles, protocol)
        methods = [cfg["families"][f][l] for f in ("FS", "FR", "EFS") for l in cfg["selections"][selection]]
        frame = validate_manifest(manifest, methods, FIXED_BUDGETS)
        absent = [path for path in frame.source_path if not (a.data_root / path).is_file()]
        if absent:
            raise FileNotFoundError(f"{selection}: {len(absent)} frozen images missing: {absent[:5]}")
        prepared.append((selection, manifest, roles))
        print(f"Ready: {selection} | train=39600 | original val=9360 | same validation roles", flush=True)
    if a.check_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime")
    total = len(prepared) * len(a.augmentations) * len(a.strategies)
    number = 0
    for selection, manifest, roles in prepared:
        for augmentation in a.augmentations:
            for strategy in a.strategies:
                number += 1
                run, command = command_for(a, settings, selection, strategy, augmentation, manifest, roles)
                print(f"\n===== [{number}/{total}] {selection} / {augmentation} / {strategy} / seed42 =====", flush=True)
                config = run / "config.json"
                if config.exists():
                    saved = json.loads(config.read_text())
                    validate_saved(saved, sha(manifest), sha(roles), protocol, strategy, augmentation, sha(a.protocol_config))
                    for key in ("epochs", "batch_size", "workers", "patience", "learning_rate", "weight_decay"):
                        if saved.get(key) != getattr(a, key):
                            raise RuntimeError(f"Existing run {key} differs: {run}")
                    adaptive = saved["adaptive_learning"]
                    if (adaptive["implementation_sha256"] != sha(ROOT / "scripts/family_rotation_adaptive_learning.py")
                            or adaptive["trainer_implementation_sha256"] != sha(ROOT / "scripts/train_xception_letterbox.py")):
                        raise RuntimeError(f"Trainer implementation changed since this run: {run}")
                elif (run / "last.pt").exists() or (run / "best.pt").exists():
                    raise RuntimeError(f"Run has weights but lacks config.json: {run}")
                last, best = run / "last.pt", run / "best.pt"
                if last.is_file():
                    terminal = torch.load(last, map_location="cpu", weights_only=False)
                    validate_saved(terminal["config"], sha(manifest), sha(roles), protocol, strategy, augmentation, sha(a.protocol_config))
                    complete = terminal["epoch"] >= a.epochs or terminal.get("epochs_without_improvement", 0) >= a.patience
                    del terminal
                    if complete:
                        if not best.is_file():
                            raise FileNotFoundError(f"Completed run lacks best.pt: {run}")
                        print("Already completed; skipping:", run.name, flush=True)
                        continue
                    print("Resuming last completed epoch:", run.name, flush=True)
                subprocess.run(command, check=True)
                if not last.is_file() or not best.is_file():
                    raise RuntimeError(f"Training returned without last.pt/best.pt: {run}")
    print("All requested augmented M7 runs complete.", flush=True)


if __name__ == "__main__":
    main()
