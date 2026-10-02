#!/usr/bin/env python3
"""Compare adaptive fixed-Q95 M7 pilots on 21 groups and optional native WildDeepfake."""

import argparse
import json
from pathlib import Path

from prepare_family_rotation_adaptive_learning import ROOT, STRATEGIES, atomic_json, atomic_csv, load_roles, run_directory, sha
from summarize_family_rotation_results import selection_scope, summarize_method_rows


def validate_run(saved, manifest, roles, protocol, strategy):
    adaptive = saved.get("adaptive_learning", {})
    if (saved.get("manifest_sha256") != sha(manifest) or saved.get("seed") != 42
            or adaptive.get("strategy") != strategy or adaptive.get("protocol") != protocol
            or adaptive.get("validation_roles_sha256") != sha(roles)):
        raise RuntimeError("Adaptive checkpoint/manifest/validation-role identity differs")
    expected = {"canonical_size": 256, "jpeg_quality": 95, "jpeg_subsampling": 2,
                "jpeg_optimize": False, "jpeg_progressive": False}
    preprocessing = saved.get("preprocessing", {})
    if any(preprocessing.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Expected fixed-Q95 input preprocessing")
    if saved.get("train_images") != 39600 or saved.get("preprocessing_name") != "canonical256_jpegq95_letterbox299":
        raise RuntimeError("Unexpected adaptive training budget or preprocessing name")


def save_summary(output, rows, methods, identity):
    import pandas as pd
    models = pd.DataFrame(rows)
    atomic_csv(output / "model_summary.csv", models)
    atomic_csv(output / "method_summary.csv", pd.DataFrame(methods))
    comparisons = []
    for selection, group in models.groupby("selection"):
        uniform = group[group.strategy == "uniform"]
        if len(uniform) != 1:
            continue
        base = uniform.iloc[0]
        for _, row in group[group.strategy != "uniform"].iterrows():
            comparisons.append({"selection": selection, "strategy": row.strategy,
                **{f"delta_{metric}": float(row[metric] - base[metric]) for metric in
                    ("df40_unseen_macro_auc", "df40_seen_macro_auc", "real_fpr", "ffpp_reference_macro_auc")}})
    atomic_csv(output / "baseline_comparison.csv", pd.DataFrame(comparisons, columns=["selection", "strategy",
        "delta_df40_unseen_macro_auc", "delta_df40_seen_macro_auc", "delta_real_fpr", "delta_ffpp_reference_macro_auc"]))
    atomic_json(output / "evaluation_summary.json", {**identity, "completed_evaluations": len(rows),
        "expected_evaluations": len(identity["selections"]) * len(identity["strategies"]),
        "evaluation_role": "development; repeatedly inspected DF40 methods", "test_used_for_tuning": False})
    lines = ["# Q95 M7 adaptive-learning pilot", "", "seed42. Same 39600 train images; same source-disjoint validation roles.",
        "Primary metric: selection-common unseen12 macro AUC. Real is the negative class, FF++ is separate.",
        "Extra temporary updates add computation; compare their logs before making a compute-budget claim.", "",
        "| Selection | Strategy | Unseen12 AUC | Real FPR | Seen6 AUC | FF++ AUC |",
        "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['selection']} | {row['strategy']} | {row['df40_unseen_macro_auc']:.6f} | {row['real_fpr']:.2%} | {row['df40_seen_macro_auc']:.6f} | {row['ffpp_reference_macro_auc']:.6f} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=["S2", "S3", "S4"], default=["S2", "S3", "S4"])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    for name in ("data-root", "df40-test-manifest", "ffpp-test-manifest", "manifest-root", "roles-root", "runs-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/family_rotation_v1/selections.json")
    p.add_argument("--protocol-config", type=Path, default=ROOT / "configs/family_rotation_adaptive_learning_v1/protocol.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--wild-data-root", type=Path)
    p.add_argument("--wild-manifest", type=Path)
    p.add_argument("--wild-only", action="store_true")
    a = p.parse_args()
    if bool(a.wild_data_root) != bool(a.wild_manifest) or a.wild_only and a.wild_manifest is None:
        p.error("Wild evaluation requires both --wild-data-root and --wild-manifest")
    if any(len(v) != len(set(v)) for v in (a.selections, a.strategies)) or a.batch_size < 1 or a.workers < 0:
        p.error("Invalid duplicate selections/strategies or loader settings")
    import pandas as pd
    cfg, protocol = [json.loads(path.read_text()) for path in (a.config, a.protocol_config)]
    test = None
    if not a.wild_only:
        test = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False)
            for path in (a.df40_test_manifest, a.ffpp_test_manifest)], ignore_index=True)
        expected = {"original", "Deepfakes", "Face2Face", *(m for family in cfg["families"].values() for m in family.values())}
        counts = test.groupby("method").size()
        if set(counts.index) != expected or set(counts) != {2000} or len(test) != 42000 or set(test.split) != {"test"}:
            raise ValueError(f"Expected frozen 21x2000 test groups: {counts.to_dict()}")
        if not test.label.eq(test.method.map(lambda m: "real" if m == "original" else "fake")).all():
            raise ValueError("Test method/label provenance differs")
        for column in ("sample_id", "source_path", "content_sha256"):
            if test[column].duplicated().any():
                raise ValueError(f"Duplicate test {column}")
        absent = [path for path in test.source_path if not (a.data_root / path).is_file()]
        if absent:
            raise FileNotFoundError(f"Test images missing: {absent[:5]}")
        test["resolved_path"] = test.source_path
    runs = []
    for selection in a.selections:
        manifest = a.manifest_root / selection.lower() / "m7_seed42.csv"
        roles = a.roles_root / selection.lower() / "validation_roles.csv"
        load_roles(manifest, roles, protocol)
        for strategy in a.strategies:
            path = run_directory(a.runs_root, selection, strategy, 42) / "best.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = json.loads((path.parent / "config.json").read_text())
            validate_run(saved, manifest, roles, protocol, strategy)
            runs.append((selection, strategy, manifest, roles, path))
            print(f"Ready: {selection} / {strategy}", flush=True)
    if a.check_only:
        return
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
    from evaluate_xception_family_coverage import cross_split_audit
    from evaluate_xception_family_rotation import method_rows
    from xception_preprocessing import LETTERBOX_NAME, evaluation_transform_from_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    a.output_root.mkdir(parents=True, exist_ok=True)
    if a.wild_manifest:
        from evaluate_xception_wilddeepfake import build_loaders, evaluate_one, summary_row, training_overlap_audit, validate_manifest
        wild = validate_manifest(a.wild_manifest, a.wild_data_root,
            json.loads((ROOT / "configs/wilddeepfake_evaluation_v1/protocol.json").read_text()))
    rows, methods, wild_rows, compute_rows = [], [], [], []
    identity = {"protocol": protocol["protocol"], "selections": a.selections, "strategies": a.strategies,
        "training_seed": 42, "jpeg_quality": 95, "test_groups": 21, "test_images": 42000,
        "decision_threshold": 0.5}
    for number, (selection, strategy, manifest, roles, path) in enumerate(runs, 1):
        print(f"\n===== [{number}/{len(runs)}] {selection} / {strategy} =====", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        validate_run(saved, manifest, roles, protocol, strategy)
        if checkpoint.get("label_map") != {"real": 0, "fake": 1} or checkpoint.get("model_name") not in {"xception", "legacy_xception"}:
            raise RuntimeError("Unexpected checkpoint architecture or labels")
        checkpoint_hash = sha(path)
        last_path = path.with_name("last.pt")
        if not last_path.is_file():
            raise FileNotFoundError(last_path)
        terminal = torch.load(last_path, map_location="cpu", weights_only=False)
        state = terminal["adaptive_state"]
        compute_rows.append({"selection": selection, "strategy": strategy,
            "completed_epochs": terminal["epoch"], "best_epoch": checkpoint["epoch"],
            "unique_train_images": saved["train_images"],
            "ordinary_train_images_seen": terminal["ordinary_train_images_seen"],
            "extra_forward_images": state["extra_forward_images"],
            "extra_backward_images": state["extra_backward_images"],
            "adaptive_refreshes": state["refreshes"], "adaptive_probe_seconds": state["seconds"],
            "train_validation_seconds": terminal["adaptive_train_validation_seconds"]})
        atomic_csv(a.output_root / "compute_summary.csv", pd.DataFrame(compute_rows))
        del terminal
        development = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if not a.wild_only:
            audit = cross_split_audit(development, test)
            output = a.output_root / selection.lower() / strategy / "q95"
            output.mkdir(parents=True, exist_ok=True)
            result_identity = {"checkpoint_sha256": checkpoint_hash, "manifest_sha256": sha(manifest),
                "roles_sha256": sha(roles), "df40_test_sha256": sha(a.df40_test_manifest),
                "ffpp_test_sha256": sha(a.ffpp_test_manifest), "selection_config_sha256": sha(a.config),
                "preprocessing": "fixed_q95", "threshold": 0.5}
            metrics_path = output / "metrics.json"
            if metrics_path.exists() and (output / "predictions.csv").exists():
                result = json.loads(metrics_path.read_text())
                if result["identity"] != result_identity:
                    raise RuntimeError(f"Completed evaluation identity differs: {output}")
                metrics = result["metrics"]
                print("Reusing completed evaluation", flush=True)
            else:
                network = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
                network.load_state_dict(checkpoint["model_state_dict"])
                network.to(device)
                transform = evaluation_transform_from_checkpoint(checkpoint)
                loader = DataLoader(EvaluationDataset(test, a.data_root, transform, LETTERBOX_NAME, 299),
                    batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
                metrics, labels, predictions, probabilities = evaluate_with_predictions(network, loader, test,
                    nn.CrossEntropyLoss(), device, progress_desc=f"{selection} {strategy} Q95", persistent_progress=True)
                prediction = test.drop(columns="resolved_path").copy()
                prediction["true_class"], prediction["predicted_class"], prediction["fake_probability"] = labels, predictions, probabilities
                atomic_csv(output / "predictions.csv", prediction)
                atomic_json(metrics_path, {"identity": result_identity, "metrics": metrics, "cross_split_audit": audit})
                del network, loader
            selected, _ = selection_scope(cfg, selection)
            method_result = method_rows(selection, strategy, "M7", selected, test, metrics)
            for row in method_result:
                row.update(strategy=strategy, training_seed=42, jpeg_quality=95)
            summary = summarize_method_rows(method_result, cfg, selection)
            cm = metrics["confusion_matrix"]
            rows.append({"selection": selection, "strategy": strategy, "training_seed": 42,
                "real_fpr": cm[0][1] / sum(cm[0]), **summary})
            methods.extend(method_result)
            save_summary(a.output_root, rows, methods, identity)
            print(f"Unseen12 AUC={summary['df40_unseen_macro_auc']:.6f}; Real FPR={rows[-1]['real_fpr']:.2%}", flush=True)
        if a.wild_manifest:
            audit = training_overlap_audit(wild, development)
            loaders = build_loaders(wild, a.wild_data_root, saved["data_config"], a.batch_size, a.workers)
            info = {"path": str(path), "sha256": checkpoint_hash, "epoch": checkpoint["epoch"],
                    "best_validation_auc": checkpoint["best_val_auc"]}
            frames, sequences = evaluate_one(strategy, "M7", 42, {"preprocessing": "native_letterbox299"},
                checkpoint, info, loaders["native_letterbox299"], wild, a.wild_manifest,
                a.output_root / "wilddeepfake" / selection.lower(), device)
            row = summary_row(strategy, "M7", {"name": f"{selection.lower()}_{strategy}"}, 42,
                "native_letterbox299", frames, sequences)
            row.update(selection=selection, strategy=strategy)
            wild_rows.append(row)
            atomic_csv(a.output_root / "wilddeepfake_summary.csv", pd.DataFrame(wild_rows))
            print(f"Wild sequence AUC={row['sequence_roc_auc']:.6f}; Real FPR={row['sequence_real_fpr']:.2%}", flush=True)
            del loaders
        del checkpoint
        torch.cuda.empty_cache()
    print("Completed requested evaluations:", a.output_root, flush=True)


if __name__ == "__main__":
    main()
