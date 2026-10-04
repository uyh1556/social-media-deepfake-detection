#!/usr/bin/env python3
"""Compare matched augmented M7 models on unseen12, JPEG Q50 and noise std10."""

import argparse
import json
from pathlib import Path

from family_rotation_adaptive_augmentation import (
    AUGMENTATIONS, CONFIG, EXPERIMENT, ROOT, STRATEGIES, adaptive_config_path,
    load_experiment, run_directory, validate_saved, validate_checkpoint_config,
)
from prepare_family_rotation_adaptive_learning import atomic_csv, atomic_json, load_roles, sha
from summarize_family_rotation_results import selection_scope, summarize_method_rows


def write_summaries(output, rows, methods, compute, identity):
    import pandas as pd
    models = pd.DataFrame(rows)
    atomic_csv(output / "model_summary.csv", models)
    atomic_csv(output / "method_summary.csv", pd.DataFrame(methods))
    compute_path = output / "compute_summary.csv"
    if compute:
        current_compute = pd.DataFrame(compute)
        if compute_path.is_file():
            previous_compute = pd.read_csv(compute_path)
            current_compute = pd.concat([previous_compute, current_compute], ignore_index=True).drop_duplicates(
                ["selection", "augmentation", "strategy", "training_seed"], keep="last")
        atomic_csv(compute_path, current_compute)
    comparisons = []
    keys = ["selection", "augmentation", "evaluation_condition"]
    for key, group in models.groupby(keys, sort=False):
        base = group[group.strategy.eq("uniform")]
        adapted = group[group.strategy.eq("complementarity")]
        if len(base) == len(adapted) == 1:
            b, a = base.iloc[0], adapted.iloc[0]
            comparisons.append({**dict(zip(keys, key)), "training_seed": 42,
                "jpeg_quality": int(a.jpeg_quality), "noise_std": float(a.noise_std),
                **{f"uniform_{metric}": float(b[metric]) for metric in
                    ("df40_unseen_macro_auc", "real_fpr", "df40_seen_macro_auc")},
                **{f"complementarity_{metric}": float(a[metric]) for metric in
                    ("df40_unseen_macro_auc", "real_fpr", "df40_seen_macro_auc")},
                **{f"delta_{metric}": float(a[metric] - b[metric]) for metric in
                    ("df40_unseen_macro_auc", "real_fpr", "df40_seen_macro_auc", "ffpp_reference_macro_auc")}})
    columns = [*keys, "training_seed", "jpeg_quality", "noise_std",
        *[f"{strategy}_{metric}" for strategy in ("uniform", "complementarity") for metric in
          ("df40_unseen_macro_auc", "real_fpr", "df40_seen_macro_auc")],
        *[f"delta_{metric}" for metric in
          ("df40_unseen_macro_auc", "real_fpr", "df40_seen_macro_auc", "ffpp_reference_macro_auc")]]
    atomic_csv(output / "baseline_comparison.csv", pd.DataFrame(comparisons, columns=columns))
    expected = len(identity["selections"]) * len(identity["augmentations"]) * len(identity["strategies"]) * len(identity["conditions"])
    atomic_json(output / "evaluation_summary.json", {**identity, "completed_evaluations": len(rows),
        "expected_evaluations": expected, "complete": len(rows) == expected,
        "decision_threshold": 0.5, "test_threshold_optimized": False,
        "evaluation_role": "development; previously inspected DF40 methods"})
    lines = ["# Augmented M7: uniform versus original complementarity", "",
        "Seed42; 39600 images per run; matched selection-validation roles; no-noise Q95 model selection.",
        "Train JPEG: 75/80/85/90/95. Noise training: probability0.25, uniform std0-5, before JPEG.",
        "Test: paired JPEG95/90/75/60/50 plus paired noise2/5/10 before JPEG95. No threshold tuning.",
        "Primary: the selection-common 12 unseen DF40 method AUCs; Real is their negative class.",
        "Extra adaptive probe computation is recorded; selections overlap and are not independent replicates.", "",
        "| Selection | Training | Test | Uniform unseen12 | Complementarity unseen12 | Delta | Uniform FPR | Complementarity FPR |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['selection']} | {row['augmentation']} | {row['evaluation_condition']}"
            f" | {row['uniform_df40_unseen_macro_auc']:.5f} | {row['complementarity_df40_unseen_macro_auc']:.5f}"
            f" | {row['delta_df40_unseen_macro_auc']:+.5f} | {row['uniform_real_fpr']:.2%}"
            f" | {row['complementarity_real_fpr']:.2%} |")
    lines.append(f"\nCompleted {len(rows)}/{expected} requested evaluations.")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=[f"S{i}" for i in range(1, 7)])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("--augmentations", nargs="+", choices=AUGMENTATIONS, default=list(AUGMENTATIONS))
    for name in ("data-root", "df40-test-manifest", "ffpp-test-manifest", "manifest-root", "roles-root", "runs-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-config", type=Path, default=CONFIG)
    p.add_argument("--conditions", nargs="+", help="Subset of frozen condition names; default all eight")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-interval-seconds", type=float, default=60)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if any(len(v) != len(set(v)) for v in (a.selections, a.strategies, a.augmentations)) or a.batch_size < 1 or a.workers < 0 or a.log_interval_seconds <= 0:
        p.error("Invalid duplicate selections or loader/log settings")
    import pandas as pd
    settings = load_experiment(a.protocol_config)
    protocol = json.loads(adaptive_config_path(settings).read_text())
    config_path = ROOT / "configs/family_rotation_v1/selections.json"
    config = json.loads(config_path.read_text())
    requested = a.conditions or [c["name"] for c in settings["evaluation_conditions"]]
    allowed = {c["name"] for c in settings["evaluation_conditions"]}
    if len(set(requested)) != len(requested) or not set(requested) <= allowed:
        p.error(f"Use unique frozen condition names: {sorted(allowed)}")
    conditions = [next(c for c in settings["evaluation_conditions"] if c["name"] == name) for name in requested]
    test = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in (a.df40_test_manifest, a.ffpp_test_manifest)], ignore_index=True)
    expected = {"original", "Deepfakes", "Face2Face", *[m for f in config["families"].values() for m in f.values()]}
    counts = test.groupby("method").size()
    if len(test) != 42000 or set(counts.index) != expected or set(counts) != {2000} or set(test.split) != {"test"}:
        raise ValueError(f"Expected 21 groups x2000: {counts.to_dict()}")
    if not test.label.eq(test.method.map(lambda m: "real" if m == "original" else "fake")).all():
        raise ValueError("Test label/provenance mismatch")
    for column in ("sample_id", "source_path", "content_sha256"):
        if test[column].duplicated().any():
            raise ValueError(f"Duplicate test {column}")
    missing = [path for path in test.source_path if not (a.data_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images missing: {missing[:5]}")
    from evaluate_xception_family_coverage import cross_split_audit
    runs = []
    for selection in a.selections:
        manifest = a.manifest_root / selection.lower() / "m7_seed42.csv"
        roles = a.roles_root / selection.lower() / "validation_roles.csv"
        load_roles(manifest, roles, protocol)
        audit = cross_split_audit(pd.read_csv(manifest, dtype=str, keep_default_na=False), test)
        for augmentation in a.augmentations:
            for strategy in a.strategies:
                path = run_directory(a.runs_root, selection, strategy, augmentation) / "best.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                saved = json.loads(path.with_name("config.json").read_text())
                validate_saved(saved, sha(manifest), sha(roles), protocol, strategy, augmentation, sha(a.protocol_config))
                runs.append((selection, augmentation, strategy, manifest, roles, path, saved, audit))
    print(f"Ready: {len(runs)} checkpoints x {len(conditions)} conditions; same42000 test images", flush=True)
    if a.check_only:
        return
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from diagnose_xception_family_rotation_corruptions import corrupt
    from evaluate_checkpoint import evaluate_with_predictions
    from evaluate_xception_family_rotation import method_rows
    from xception_preprocessing import Letterbox, JpegRoundTrip, interpolation_mode
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime")
    device = torch.device("cuda")
    identity = {"experiment": EXPERIMENT, "training_seed": 42, "selections": a.selections,
        "strategies": a.strategies, "augmentations": a.augmentations, "conditions": conditions,
        "df40_test_sha256": sha(a.df40_test_manifest), "ffpp_test_sha256": sha(a.ffpp_test_manifest),
        "selection_config_sha256": sha(config_path), "experiment_config_sha256": sha(a.protocol_config),
        "evaluation_noise_seed": settings["evaluation_noise_seed"]}
    a.output_root.mkdir(parents=True, exist_ok=True)
    rows, all_methods, compute = [], [], []
    number = 0
    for selection, augmentation, strategy, manifest, roles, path, saved, audit in runs:
        checkpoint_hash = sha(path)
        base_identity = {key: identity[key] for key in
            ("df40_test_sha256", "ffpp_test_sha256", "selection_config_sha256", "experiment_config_sha256", "evaluation_noise_seed")}
        base_identity.update({"checkpoint_sha256": checkpoint_hash, "manifest_sha256": sha(manifest),
            "roles_sha256": sha(roles), "selection": selection, "augmentation": augmentation, "strategy": strategy,
            "threshold": 0.5, "evaluator_sha256": sha(Path(__file__)),
            "corruption_implementation_sha256": sha(ROOT / "scripts/diagnose_xception_family_rotation_corruptions.py"),
            "transform_implementation_sha256": sha(ROOT / "scripts/xception_preprocessing.py")})
        network = checkpoint = None
        trained, _ = selection_scope(config, selection)
        for condition in conditions:
            number += 1
            name = condition["name"]
            output = a.output_root / selection.lower() / augmentation / strategy / name
            output.mkdir(parents=True, exist_ok=True)
            result_identity = {**base_identity, "condition": condition}
            cache = output / "metrics.json"
            prediction_file = output / "predictions.csv"
            print(f"\n[{number}/{len(runs) * len(conditions)}] {selection} / {augmentation} / {strategy} / {name}", flush=True)
            if cache.is_file() and prediction_file.is_file():
                result = json.loads(cache.read_text())
                if result.get("identity") != result_identity or result.get("prediction_rows") != len(test):
                    raise RuntimeError(f"Existing evaluation identity differs: {output}")
                metrics = result["metrics"]
                print("  Reusing completed evaluation", flush=True)
            else:
                if checkpoint is None:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    validate_saved(checkpoint["config"], sha(manifest), sha(roles), protocol, strategy, augmentation, sha(a.protocol_config))
                    validate_checkpoint_config(checkpoint["config"], saved)
                    if checkpoint.get("label_map") != {"real": 0, "fake": 1} or checkpoint["model_name"] not in {"xception", "legacy_xception"}:
                        raise ValueError(f"Unexpected architecture/labels: {path}")
                    network = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
                    network.load_state_dict(checkpoint["model_state_dict"])
                    network.to(device).eval()
                    terminal_path = path.with_name("last.pt")
                    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
                    validate_saved(terminal["config"], sha(manifest), sha(roles), protocol, strategy, augmentation, sha(a.protocol_config))
                    state = terminal["adaptive_state"]
                    compute.append({"selection": selection, "augmentation": augmentation, "strategy": strategy,
                        "training_seed": 42, "best_epoch": checkpoint["epoch"], "completed_epochs": terminal["epoch"],
                        "ordinary_train_images_seen": terminal["ordinary_train_images_seen"],
                        "extra_forward_images": state["extra_forward_images"], "extra_backward_images": state["extra_backward_images"],
                        "refreshes": state["refreshes"], "adaptive_probe_seconds": state["seconds"],
                        "train_validation_seconds": terminal["adaptive_train_validation_seconds"]})
                    del terminal
                data = saved["data_config"]
                interpolation = interpolation_mode(data["interpolation"])
                canonical = Letterbox(256, interpolation=interpolation)
                model_letterbox = Letterbox(299, interpolation=interpolation)
                jpeg = JpegRoundTrip(condition["jpeg_quality"], subsampling=2)
                tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"])])

                class ConditionDataset(Dataset):
                    def __len__(self):
                        return len(test)

                    def __getitem__(self, index):
                        row = test.iloc[index]
                        with Image.open(a.data_root / row.source_path) as image:
                            image = canonical(image.convert("RGB"))
                        if condition["noise_std"]:
                            image = corrupt(image, {"kind": "gaussian_noise", "strength": condition["noise_std"]},
                                            row.sample_id, settings["evaluation_noise_seed"])
                        return {"image": tensor(model_letterbox(jpeg(image))),
                            "label": torch.tensor(0 if row.label == "real" else 1), "index": index}

                loader = DataLoader(ConditionDataset(), batch_size=a.batch_size, shuffle=False,
                                    num_workers=a.workers, pin_memory=True)
                metrics, labels, predictions, probabilities = evaluate_with_predictions(
                    network, loader, test, torch.nn.CrossEntropyLoss(), device,
                    progress_desc=f"{selection} {strategy} {name}", progress_enabled=False,
                    log_interval_seconds=a.log_interval_seconds)
                pred = test[["sample_id", "source_path", "method", "label", "group_id", "video_id"]].copy()
                pred["true_class"], pred["predicted_class"], pred["fake_probability"] = labels, predictions, probabilities
                atomic_csv(prediction_file, pred)
                atomic_json(cache, {"identity": result_identity, "metrics": metrics,
                    "prediction_rows": len(test), "cross_split_audit": audit, "best_epoch": checkpoint["epoch"]})
            mrows = method_rows(selection, augmentation, "M7", trained, test, metrics)
            summary = summarize_method_rows(mrows, config, selection)
            cm = metrics["confusion_matrix"]
            row = {"selection": selection, "augmentation": augmentation, "strategy": strategy,
                "training_seed": 42, "evaluation_condition": name, "jpeg_quality": condition["jpeg_quality"],
                "noise_std": condition["noise_std"], **summary,
                "real_fpr": float(cm[0][1] / sum(cm[0])), "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"], "f1": metrics["f1"], "checkpoint": str(path)}
            rows.append(row)
            for mrow in mrows:
                mrow.update({"augmentation": augmentation, "strategy": strategy, "training_seed": 42,
                    "evaluation_condition": name, "jpeg_quality": condition["jpeg_quality"], "noise_std": condition["noise_std"]})
            all_methods.extend(mrows)
            write_summaries(a.output_root, rows, all_methods, compute, identity)
            print(f"  Unseen12 AUC={row['df40_unseen_macro_auc']:.5f} | Real FPR={row['real_fpr']:.2%}", flush=True)
        if network is not None:
            del network, checkpoint
            torch.cuda.empty_cache()
    print("Requested evaluation complete:", a.output_root, flush=True)


if __name__ == "__main__":
    main()
