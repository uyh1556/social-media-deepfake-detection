#!/usr/bin/env python3
"""Train matched fixed-Q95 M7 uniform/difficulty/complementarity pilots."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prepare_family_rotation_adaptive_learning import ROOT, STRATEGIES, load_roles, run_directory, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=["S2", "S3", "S4"], default=["S2", "S3", "S4"])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    for name in ("data-root", "manifest-root", "roles-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-config", type=Path, default=ROOT / "configs/family_rotation_adaptive_learning_v1/protocol.json")
    for name, default in (("epochs", 15), ("batch-size", 16), ("workers", 2), ("patience", 3), ("seed", 42)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if len(set(a.selections)) != len(a.selections) or len(set(a.strategies)) != len(a.strategies):
        p.error("Selections and strategies must be unique")
    if a.seed != 42 or a.epochs < 1 or a.batch_size < 1 or a.workers < 0 or a.patience < 1:
        p.error("Pilot uses seed42 and positive training settings")
    protocol = json.loads(a.protocol_config.read_text())
    cfg = json.loads((ROOT / "configs/family_rotation_v1/selections.json").read_text())
    from train_xception_family_coverage import FIXED_BUDGETS, validate_manifest
    prepared = []
    for selection in a.selections:
        manifest = a.manifest_root / selection.lower() / "m7_seed42.csv"
        roles = a.roles_root / selection.lower() / "validation_roles.csv"
        load_roles(manifest, roles, protocol)
        methods = [cfg["families"][f][letter] for f in ("FS", "FR", "EFS") for letter in cfg["selections"][selection]]
        frame = validate_manifest(manifest, methods, FIXED_BUDGETS)
        absent = [path for path in frame.source_path if not (a.data_root / path).is_file()]
        if absent:
            raise FileNotFoundError(f"{selection}: {len(absent)} frozen images missing: {absent[:5]}")
        prepared.append((selection, manifest, roles))
        print(f"Ready: {selection}, train=39600, original val=9360, roles verified", flush=True)
    if a.check_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime")
    for selection, manifest, roles in prepared:
        for strategy in a.strategies:
            run = run_directory(a.output_root, selection, strategy, a.seed)
            last, best = run / "last.pt", run / "best.pt"
            print(f"\n===== {selection} / {strategy} / fixed Q95 / seed42 =====", flush=True)
            if (run / "config.json").exists():
                saved = json.loads((run / "config.json").read_text())
                adaptive = saved.get("adaptive_learning", {})
                expected = {"strategy": strategy, "protocol": protocol, "protocol_sha256": sha(a.protocol_config),
                    "validation_roles_sha256": sha(roles),
                    "implementation_sha256": sha(ROOT / "scripts/family_rotation_adaptive_learning.py"),
                    "trainer_implementation_sha256": sha(ROOT / "scripts/train_xception_letterbox.py")}
                if any(adaptive.get(k) != v for k, v in expected.items()) or saved.get("manifest_sha256") != sha(manifest):
                    raise RuntimeError(f"Existing run identity differs: {run}")
                for key, value in (("seed", a.seed), ("epochs", a.epochs), ("batch_size", a.batch_size),
                    ("workers", a.workers), ("patience", a.patience), ("learning_rate", a.learning_rate), ("weight_decay", a.weight_decay)):
                    if saved.get(key) != value:
                        raise RuntimeError(f"Existing run {key} differs: {run}")
            if last.is_file():
                checkpoint = torch.load(last, map_location="cpu", weights_only=False)
                terminal = checkpoint["epoch"] >= a.epochs or checkpoint.get("epochs_without_improvement", 0) >= a.patience
                del checkpoint
                if terminal:
                    if not best.is_file():
                        raise FileNotFoundError(f"Terminal run lacks best.pt: {run}")
                    print(f"Already completed: {best}", flush=True)
                    continue
            command = [sys.executable, "-u", str(ROOT / "scripts/train_xception_letterbox.py"),
                "--data-root", str(a.data_root), "--manifest", str(manifest), "--output-dir", str(run),
                "--split-protocol", "family_rotation_adaptive_learning_v1", "--model", "xception",
                "--epochs", str(a.epochs), "--batch-size", str(a.batch_size), "--workers", str(a.workers),
                "--learning-rate", str(a.learning_rate), "--weight-decay", str(a.weight_decay),
                "--patience", str(a.patience), "--seed", str(a.seed),
                "--experiment-family", "family_rotation_adaptive_learning_v1",
                "--condition-name", f"{selection.lower()}_m7_q95_{strategy}",
                "--canonical-size", "256", "--jpeg-quality", "95", "--jpeg-subsampling", "2",
                "--adaptive-strategy", strategy, "--adaptive-protocol-config", str(a.protocol_config),
                "--adaptive-validation-roles", str(roles), "--persistent-progress"]
            if last.exists():
                command += ["--resume", str(last)]
                print("Resume from last completed epoch:", last, flush=True)
            subprocess.run(command, check=True)
            if not last.is_file() or not best.is_file():
                raise RuntimeError(f"Training returned without required checkpoints: {run}")
    print("Requested matched pilot runs complete.", flush=True)


if __name__ == "__main__":
    main()
