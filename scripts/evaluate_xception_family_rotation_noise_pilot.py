#!/usr/bin/env python3
"""Compare frozen M7 Mixed-JPEG/noise2/noise5 on 21 test groups at Q95/Q90/Q75."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from summarize_family_rotation_results import selection_scope, summarize_method_rows


VARIANTS = ("mixed_jpeg", "noise_std2", "noise_std5")
QUALITIES = (95, 90, 75)
POLICY = "m7_noise_pilot_21groups_jpeg_v1"


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", nargs="+", choices=["S2", "S3", "S4"], default=["S2", "S3", "S4"])
    for name in ("data-root", "df40-test-manifest", "ffpp-test-manifest", "manifest-root", "baseline-runs-root", "pilot-runs-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--jpeg-qualities", nargs="+", type=int, choices=QUALITIES, default=list(QUALITIES))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Check manifests/images/run config.json without loading model weights or running inference")
    args = parser.parse_args()
    for name in ("selections", "variants", "jpeg_qualities"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate {name}")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(args, config, selection, variant):
    if variant == "mixed_jpeg":
        name = config["models"]["M7"]["name"]
        run = f"xception_{selection.lower()}_m7_{name}_mixed_jpeg_family_rotation_v1_seed42"
        return args.baseline_runs_root / run / "best.pt"
    maximum = 2 if variant == "noise_std2" else 5
    run = f"xception_{selection.lower()}_m7_mixed_jpeg_noise_p25_std{maximum}_seed42"
    return args.pilot_runs_root / run / "best.pt"


def validate_saved_config(saved, training_hash, variant, path):
    if saved.get("manifest_sha256") != training_hash or int(saved.get("seed", -1)) != 42:
        raise RuntimeError(f"Checkpoint/manifest/seed mismatch: {path}")
    preprocessing = saved.get("preprocessing", {})
    expected = {
        "canonical_size": 256, "jpeg_subsampling": 2,
        "jpeg_optimize": False, "jpeg_progressive": False,
        "validation_jpeg_quality": 95,
        "train_jpeg_qualities": [75, 80, 85, 90, 95],
    }
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise RuntimeError(f"Unexpected {key} in {path}: {preprocessing.get(key)!r}")
    noise = preprocessing.get("train_noise", {})
    probability = float(noise.get("probability", 0))
    if variant == "mixed_jpeg":
        if probability != 0:
            raise RuntimeError(f"Baseline contains training noise: {path}")
    elif probability != 0.25 or noise.get("max_std_0_to_255") != (2 if variant == "noise_std2" else 5):
        raise RuntimeError(f"Wrong noise pilot configuration: {path}")
    data = saved.get("data_config", {})
    if list(data.get("input_size", [])) != [3, 299, 299] or data.get("interpolation") != "bicubic":
        raise RuntimeError(f"Unexpected Xception input configuration: {path}")
    if list(data.get("mean", [])) != [0.5] * 3 or list(data.get("std", [])) != [0.5] * 3:
        raise RuntimeError(f"Unexpected normalization: {path}")


def prepare(args, config):
    import pandas as pd

    print("Checking test manifests and nine run configurations...", flush=True)
    test = pd.concat([
        pd.read_csv(args.df40_test_manifest, dtype=str, keep_default_na=False),
        pd.read_csv(args.ffpp_test_manifest, dtype=str, keep_default_na=False),
    ], ignore_index=True)
    required = {"sample_id", "source_path", "content_sha256", "split", "label", "method", "family", "group_id", "video_id", "source_ids", "driver_id"}
    if missing := required - set(test.columns):
        raise ValueError(f"Missing test columns: {sorted(missing)}")
    methods = {"original", "Deepfakes", "Face2Face", *(method for family in config["families"].values() for method in family.values())}
    counts = test.groupby("method").size()
    if len(test) != 42000 or set(counts.index) != methods or set(counts.tolist()) != {2000}:
        raise ValueError(f"Expected 21 groups x 2000 images: {counts.to_dict()}")
    if set(test["split"]) != {"test"} or set(test["label"]) != {"real", "fake"}:
        raise ValueError("Test split/labels differ from frozen protocol")
    if not test.loc[test.method == "original", "label"].eq("real").all() or not test.loc[test.method != "original", "label"].eq("fake").all():
        raise ValueError("Unexpected Real/Fake method labels")
    for column in ("sample_id", "source_path", "content_sha256"):
        if test[column].duplicated().any():
            raise ValueError(f"Duplicate test {column}")
    absent = [path for path in test.source_path if not (args.data_root / path).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} test images missing; first: {absent[:5]}")
    runs = []
    reference_data = None
    for selection in args.selections:
        manifest = args.manifest_root / selection.lower() / "m7_seed42.csv"
        training = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        selected, _ = selection_scope(config, selection)
        if set(training.loc[training.label == "fake", "method"]) != selected:
            raise ValueError(f"Wrong M7 method membership: {manifest}")
        training_hash = sha256(manifest)
        for variant in args.variants:
            path = checkpoint_path(args, config, selection, variant)
            if not path.is_file():
                raise FileNotFoundError(path)
            saved = json.loads((path.parent / "config.json").read_text())
            validate_saved_config(saved, training_hash, variant, path)
            data = saved["data_config"]
            if reference_data is None:
                reference_data = data
            elif data != reference_data:
                raise RuntimeError(f"Model input configuration differs: {path}")
            runs.append({"selection": selection, "variant": variant, "path": path,
                         "manifest": manifest, "training_hash": training_hash, "saved": saved})
            print(f"Ready: {selection} / {variant} / seed42", flush=True)
    test["resolved_path"] = test["source_path"]
    test["source_reference_path"] = test["source_path"]
    print(f"Verified: {len(test)} test images, {len(runs)} checkpoints", flush=True)
    return test, runs


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(frame, path):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def comparison_table(models):
    import pandas as pd

    metrics = ["df40_unseen_macro_auc", "df40_seen_macro_auc", "ffpp_reference_macro_auc", "real_fpr", "df40_unseen_macro_fake_recall"]
    rows = []
    for (selection, quality), group in models.groupby(["selection", "jpeg_quality"], sort=False):
        baseline = group[group.variant == "mixed_jpeg"]
        if len(baseline) != 1:
            continue
        baseline = baseline.iloc[0]
        for _, pilot in group[group.variant != "mixed_jpeg"].iterrows():
            row = {"selection": selection, "jpeg_quality": quality, "variant": pilot.variant}
            for metric in metrics:
                row[f"baseline_{metric}"] = float(baseline[metric])
                row[f"pilot_{metric}"] = float(pilot[metric])
                row[f"delta_{metric}"] = float(pilot[metric] - baseline[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def save_summaries(args, config, model_rows, method_rows, test_hash):
    import pandas as pd

    models, methods = pd.DataFrame(model_rows), pd.DataFrame(method_rows)
    atomic_csv(models, args.output_root / "model_summary.csv")
    atomic_csv(methods, args.output_root / "method_summary.csv")
    comparison = comparison_table(models)
    atomic_csv(comparison, args.output_root / "baseline_comparison.csv")
    if not comparison.empty:
        metrics = [column for column in comparison if column.startswith("delta_")]
        aggregate = comparison.groupby(["variant", "jpeg_quality"], sort=False)[metrics].mean().reset_index()
        aggregate["selection_count"] = comparison.groupby(["variant", "jpeg_quality"], sort=False).size().to_numpy()
        atomic_csv(aggregate, args.output_root / "selection_mean_deltas.csv")
    scope = {selection: sorted(selection_scope(config, selection)[1]) for selection in args.selections}
    atomic_json(args.output_root / "evaluation_summary.json", {
        "protocol": POLICY, "training_seed": 42, "selections": args.selections,
        "variants": args.variants, "jpeg_qualities": args.jpeg_qualities,
        "test_manifest_sha256": test_hash, "test_images": 42000, "test_groups": 21,
        "expected_evaluations": len(args.selections) * len(args.variants) * len(args.jpeg_qualities),
        "completed_evaluations": len(models), "df40_common_unseen_methods": scope,
        "df40_unseen_definition": "18 DF40 methods minus the selection's six M7 training methods; Real is each AUC's negative class",
        "ffpp_in_df40_unseen": False, "decision_threshold": 0.5,
        "evaluation_preprocessing": "RGB -> bicubic Letterbox256 -> JPEG Q95/Q90/Q75 (4:2:0) -> Letterbox299 -> Normalize",
        "test_noise_applied": False, "training_performed": False,
        "evaluation_role": "development evaluation; repeatedly inspected DF40 methods",
        "selection_mean_is_not_seed_variance_or_confidence_interval": True,
    })
    columns = ["selection", "variant", "jpeg_quality", "df40_unseen_macro_auc", "real_fpr", "df40_unseen_macro_fake_recall", "ffpp_reference_macro_auc"]
    lines = ["# M7 JPEG·노이즈 학습 모델 비교", "", "S2·S3·S4, seed42. 동일한 21그룹 × 2,000장을 평가합니다.",
             "각 selection의 공통 unseen 12개는 다를 수 있으며, 동일 selection의 세 모델에서는 같습니다.",
             "평가 입력에는 노이즈를 추가하지 않습니다. FPR·recall 임계값은 0.5입니다.",
             "selection 평균은 반복 seed의 평균이나 신뢰구간이 아닙니다.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in models[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
    lines.extend(["", "baseline_comparison.csv의 delta는 pilot − baseline입니다. AUC/recall은 양수가 개선, FPR은 음수가 개선입니다.",
                  "노이즈 학습의 unseen/JPEG 보존 여부를 확인하는 개발 평가이며, 새 학습법의 최종 독립 검증으로 해석하지 않습니다."])
    temporary = args.output_root / "REPORT.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, args.output_root / "REPORT.md")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    test, runs = prepare(args, config)
    if args.check_only:
        return

    import pandas as pd
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
    from evaluate_xception_family_coverage import cross_split_audit
    from evaluate_xception_family_rotation import method_rows
    from xception_preprocessing import LETTERBOX_NAME, canonical_reencode_transform

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    test_hash = hashlib.sha256((sha256(args.df40_test_manifest) + sha256(args.ffpp_test_manifest)).encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_models, all_methods = [], []
    total = len(runs) * len(args.jpeg_qualities)
    current = 0
    for run in runs:
        selection, variant, checkpoint_file = run["selection"], run["variant"], run["path"]
        print(f"\nLoading {selection} / {variant}: {checkpoint_file}", flush=True)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        validate_saved_config(checkpoint["config"], run["training_hash"], variant, checkpoint_file)
        if checkpoint.get("label_map") != {"real": 0, "fake": 1} or checkpoint.get("num_classes") != 2:
            raise ValueError(f"Wrong checkpoint labels: {checkpoint_file}")
        if checkpoint.get("model_name") not in {"xception", "legacy_xception"}:
            raise ValueError(f"Wrong model: {checkpoint_file}")
        saved = checkpoint["config"]
        audit = cross_split_audit(pd.read_csv(run["manifest"], dtype=str, keep_default_na=False), test)
        checkpoint_hash = sha256(checkpoint_file)
        network = None
        for quality in args.jpeg_qualities:
            current += 1
            print(f"\n===== [{current}/{total}] {selection} / {variant} / JPEG Q{quality} =====", flush=True)
            output = args.output_root / selection.lower() / variant / f"q{quality}"
            identity = {"protocol": POLICY, "selection": selection, "variant": variant,
                        "training_seed": 42, "jpeg_quality": quality,
                        "checkpoint_sha256": checkpoint_hash,
                        "training_manifest_sha256": run["training_hash"],
                        "test_manifest_sha256": test_hash, "selection_config_sha256": sha256(args.config)}
            metrics_path = output / "metrics.json"
            if metrics_path.is_file() and (output / "predictions.csv").is_file():
                result = json.loads(metrics_path.read_text())
                if result.get("identity") != identity:
                    raise RuntimeError(f"Existing evaluation identity differs: {output}")
                print("Reusing completed evaluation", flush=True)
                metrics, real_score = result["metrics"], result["real_mean_fake_score"]
                scores_by_method = result["mean_fake_score_by_method"]
            else:
                if network is None:
                    network = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
                    network.load_state_dict(checkpoint["model_state_dict"])
                    network.to(device)
                preprocessing = saved["preprocessing"]
                transform = canonical_reencode_transform(saved["data_config"], canonical_size=256,
                    jpeg_quality=quality, jpeg_subsampling=preprocessing["jpeg_subsampling"],
                    jpeg_optimize=preprocessing["jpeg_optimize"], jpeg_progressive=preprocessing["jpeg_progressive"])
                dataset = EvaluationDataset(test, args.data_root, transform, LETTERBOX_NAME, 299)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
                metrics, labels, predictions, probabilities = evaluate_with_predictions(
                    network, loader, test, nn.CrossEntropyLoss(), device,
                    progress_desc=f"{selection} {variant} Q{quality}", persistent_progress=True)
                prediction = test.drop(columns=["resolved_path"]).copy()
                prediction["true_class"], prediction["predicted_class"] = labels, predictions
                prediction["fake_probability"] = probabilities
                real_score = float(prediction.loc[prediction.label == "real", "fake_probability"].mean())
                scores_by_method = prediction.groupby("method").fake_probability.mean().to_dict()
                output.mkdir(parents=True, exist_ok=True)
                atomic_csv(prediction, output / "predictions.csv")
                atomic_json(metrics_path, {"identity": identity, "checkpoint": str(checkpoint_file),
                    "cross_split_audit": audit, "metrics": metrics,
                    "real_mean_fake_score": real_score, "mean_fake_score_by_method": scores_by_method})
                del dataset, loader, prediction
            selected, unseen = selection_scope(config, selection)
            rows = method_rows(selection, variant, "M7", selected, test, metrics)
            for row in rows:
                row.update(variant=variant, training_seed=42, jpeg_quality=quality,
                           mean_fake_score=float(scores_by_method[row["method"]]))
            summary = summarize_method_rows(rows, config, selection)
            cm = metrics["confusion_matrix"]
            all_models.append({"selection": selection, "variant": variant, "model": "M7",
                "training_seed": 42, "jpeg_quality": quality, "real_fpr": cm[0][1] / sum(cm[0]),
                "real_mean_fake_score": real_score,
                "df40_unseen_macro_fake_recall": sum(row["fake_recall"] for row in rows if row["method"] in unseen) / 12,
                **summary})
            all_methods.extend(rows)
            save_summaries(args, config, all_models, all_methods, test_hash)
            last = all_models[-1]
            print(f"Unseen12 AUC={last['df40_unseen_macro_auc']:.4f} | Real FPR={last['real_fpr']:.2%}", flush=True)
        del network, checkpoint
        torch.cuda.empty_cache()
    print(f"\nCompleted {total}/{total} evaluations. Results: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
