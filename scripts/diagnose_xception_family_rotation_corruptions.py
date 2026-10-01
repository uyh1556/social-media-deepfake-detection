#!/usr/bin/env python3
"""Paired blur/resize/noise diagnosis of frozen seed42 Mixed-JPEG M7 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=[f"S{i}" for i in range(1, 7)])
    for name in ("data-root", "rotation-manifest-root", "s1-manifest-root", "rotation-runs-root", "s1-mixed-runs-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    p.add_argument("--diagnostic-config", type=Path, default=root / "configs/family_rotation_corruption_diagnostic_v1/protocol.json")
    p.add_argument("--full-validation", action="store_true", help="Use every validation row instead of the frozen 1200-image subset")
    p.add_argument("--available-paths", type=Path, help="Recorded TAR-available validation paths per selection; never changes the training manifest")
    p.add_argument("--pilot-runs-root", type=Path, help="Use noise-augmentation pilot checkpoints instead of original M7")
    p.add_argument("--pilot-noise-max-std", type=int, choices=[2, 5], default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def stable_seed(sample_id, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}|gaussian_noise|{sample_id}".encode()).digest()[:8], "big")


def corrupt(image, condition, sample_id, seed):
    if image.mode != "RGB" or image.size != (256, 256):
        raise ValueError("Corruption must receive a canonical 256x256 RGB image")
    kind, strength = condition["kind"], condition["strength"]
    if kind == "identity":
        return image.copy()
    if kind == "downsample_upsample":
        size = int(strength)
        return image.resize((size, size), Image.Resampling.BICUBIC).resize((256, 256), Image.Resampling.BICUBIC)
    values = np.asarray(image, dtype=np.float32)
    if kind == "gaussian_noise":
        rng = np.random.default_rng(stable_seed(sample_id, seed))
        values = values + rng.standard_normal(values.shape).astype(np.float32) * float(strength)
    elif kind == "gaussian_blur":
        radius = int(condition["kernel_size"]) // 2
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(offsets ** 2) / (2 * float(strength) ** 2))
        kernel /= kernel.sum()
        for axis in (0, 1):
            padding = [(0, 0)] * 3
            padding[axis] = (radius, radius)
            padded = np.pad(values, padding, mode="reflect")
            result = np.zeros_like(values)
            for index, weight in enumerate(kernel):
                slices = [slice(None)] * 3
                slices[axis] = slice(index, index + values.shape[axis])
                result += weight * padded[tuple(slices)]
            values = result
    else:
        raise ValueError(f"Unknown corruption: {kind}")
    return Image.fromarray(np.rint(values).clip(0, 255).astype(np.uint8))


def select_validation(training, methods, protocol, full=False):
    frame = training[training["split"] == "val"].copy()
    if frame.empty or set(frame["method"]) != {"original", *methods}:
        raise ValueError("Validation must contain Real and the selection's six seen methods")
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate validation sample IDs")
    expected = frame.method.map(lambda value: "real" if value == "original" else "fake")
    if not expected.equals(frame.label):
        raise ValueError("Validation labels disagree with method provenance")
    if full:
        return frame.sort_values("sample_id").reset_index(drop=True)
    parts = []
    for method in ["original", *sorted(methods)]:
        pool = frame[frame.method == method].copy()
        count = protocol["real_samples"] if method == "original" else protocol["fake_samples_per_method"]
        if len(pool) < count:
            raise ValueError(f"Insufficient validation images: {method}: need {count}, found {len(pool)}")
        pool["_rank"] = pool.sample_id.map(lambda value: hashlib.sha256(f"{protocol['sampling_seed']}|{method}|{value}".encode()).hexdigest())
        parts.append(pool.sort_values(["_rank", "sample_id"]).head(count).drop(columns="_rank"))
    return pd.concat(parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)


def score_summary(before, after, threshold):
    delta = after - before
    return {
        "images": len(delta), "mean_fake_score_before": float(before.mean()),
        "mean_fake_score_after": float(after.mean()), "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)), "delta_std": float(delta.std()),
        "delta_q05": float(np.quantile(delta, .05)), "delta_q95": float(np.quantile(delta, .95)),
        "score_increase_fraction": float((delta > 0).mean()),
        "score_decrease_fraction": float((delta < 0).mean()),
        "fake_prediction_rate_before": float((before >= threshold).mean()),
        "fake_prediction_rate_after": float((after >= threshold).mean()),
        "prediction_flip_fraction": float(((before >= threshold) != (after >= threshold)).mean()),
    }


def paired_summaries(frame, original, changed, condition, selection, threshold):
    pairs = frame[["sample_id", "source_path", "group_id", "video_id", "method", "label"]].copy()
    pairs.insert(0, "selection", selection)
    pairs["condition"] = condition
    pairs["fake_score_original"] = original
    pairs["fake_score_transformed"] = changed
    pairs["delta_fake_score"] = changed - original
    pairs["original_predicted_fake"] = original >= threshold
    pairs["transformed_predicted_fake"] = changed >= threshold
    rows = []
    groups = [("label", label, frame.label.eq(label).to_numpy()) for label in ("real", "fake")]
    groups += [("method", method, frame.method.eq(method).to_numpy()) for method in sorted(frame.method.unique())]
    for level, group, mask in groups:
        row = {"selection": selection, "condition": condition, "level": level, "group": group, **score_summary(original[mask], changed[mask], threshold)}
        if group in {"real", "original"}:
            row["real_fpr_before"] = row["fake_prediction_rate_before"]
            row["real_fpr_after"] = row["fake_prediction_rate_after"]
        else:
            row["fake_recall_before"] = row["fake_prediction_rate_before"]
            row["fake_recall_after"] = row["fake_prediction_rate_after"]
        rows.append(row)
    return pairs, rows


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def pixel_quality(before, after):
    a, b = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
    mse = float(np.mean((b - a) ** 2))
    gray = .299 * b[..., 0] + .587 * b[..., 1] + .114 * b[..., 2]
    gradient = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2
    return {"psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else float("inf"), "mean_absolute_pixel_difference": float(np.abs(b-a).mean()), "mean_gradient_magnitude": float(gradient)}


def main():
    args = parse_args()
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from evaluate_checkpoint import EvaluationDataset, evaluate_with_predictions
    from evaluate_xception_family_rotation import checkpoint_path, manifest_path, methods_for
    from train_baseline import sha256_file, write_json
    from xception_preprocessing import Letterbox, JpegRoundTrip, interpolation_mode, evaluation_transform_from_checkpoint, MIXED_JPEG_REENCODE_NAME

    config = json.loads(args.config.read_text())
    protocol = json.loads(args.diagnostic_config.read_text())
    if protocol["training_seed"] != 42 or protocol["model"] != "M7" or protocol["training_protocol"] != "mixed_jpeg":
        raise ValueError("This frozen diagnostic is restricted to seed42 Mixed-JPEG M7")
    if protocol["unseen_used"] or protocol["wilddeepfake_used"]:
        raise ValueError("Diagnostic cannot use unseen or WildDeepfake")
    if len(args.selections) != len(set(args.selections)):
        raise ValueError("Duplicate selections")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU runtime is required")
    args.seed = 42
    conditions = protocol["conditions"]
    all_conditions, all_groups = [], []
    args.output_root.mkdir(parents=True, exist_ok=True)

    for selection in args.selections:
        args.selection = selection
        methods = methods_for(config, selection, "M7")
        manifest = manifest_path(args, "M7")
        checkpoint_file = checkpoint_path(args, config, "mixed_jpeg", "M7")
        if args.pilot_runs_root is not None:
            checkpoint_file = args.pilot_runs_root / f"xception_{selection.lower()}_m7_mixed_jpeg_noise_p25_std{args.pilot_noise_max_std}_seed42" / "best.pt"
        print(f"Preparing {selection}: {checkpoint_file}", flush=True)
        training = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        saved = checkpoint["config"]
        if args.pilot_runs_root is not None:
            noise = saved.get("preprocessing", {}).get("train_noise", {})
            if noise.get("probability") != 0.25 or noise.get("max_std_0_to_255") != args.pilot_noise_max_std:
                raise RuntimeError(f"Pilot noise configuration mismatch: {checkpoint_file}")
        if saved["manifest_sha256"] != sha256_file(manifest) or int(saved["seed"]) != 42:
            raise RuntimeError(f"Checkpoint/manifest/seed mismatch: {checkpoint_file}")
        if saved["preprocessing_name"] != MIXED_JPEG_REENCODE_NAME or checkpoint["label_map"] != {"real": 0, "fake": 1}:
            raise RuntimeError(f"Unexpected checkpoint preprocessing or labels: {checkpoint_file}")
        eligible = training
        if args.available_paths:
            availability = json.loads(args.available_paths.read_text())
            allowed = set(availability[selection])
            eligible = training[training.source_path.isin(allowed)].copy()
        frame = select_validation(eligible, methods, protocol, args.full_validation)
        frame["resolved_path"] = frame["source_path"]
        frame["source_reference_path"] = frame["source_path"]
        missing = [value for value in frame.source_path if not (args.data_root/value).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing validation images; check TAR extraction: {missing[:5]}")
        prep, data = saved["preprocessing"], saved["data_config"]
        if prep["canonical_size"] != 256 or prep["jpeg_subsampling"] != 2 or prep["validation_jpeg_quality"] != 95 or tuple(data["input_size"]) != (3, 299, 299):
            raise RuntimeError("Checkpoint differs from the diagnostic's canonical256/Q95/299 protocol")
        interpolation = interpolation_mode(data["interpolation"])
        canonical = Letterbox(256, interpolation=interpolation)
        model_letterbox = Letterbox(299, interpolation=interpolation)
        jpeg = JpegRoundTrip(95, 2, prep.get("jpeg_optimize", False), prep.get("jpeg_progressive", False))
        tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(data["mean"], data["std"])])
        identity = {
            "protocol": protocol, "checkpoint_sha256": sha256_file(checkpoint_file),
            "training_manifest_sha256": sha256_file(manifest),
            "sample_ids": frame.sample_id.tolist(), "full_validation": args.full_validation,
            "selection": selection, "training_seed": 42, "data_config": data,
            "jpeg_optimize": jpeg.optimize, "jpeg_progressive": jpeg.progressive,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "Pillow": Image.__version__},
        }
        identity = json.loads(json.dumps(identity))
        folder = args.output_root/f"{selection.lower()}_seed42"
        identity_path = folder/"run_identity.json"
        if folder.exists() and any(folder.iterdir()):
            if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
                raise RuntimeError(f"Diagnostic output belongs to another setting: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        write_json(identity_path, identity)
        atomic_csv(frame, folder/"selected_validation_manifest.csv")

        # Verify the unperturbed branch exactly reproduces the existing evaluation transform.
        with Image.open(args.data_root/frame.iloc[0].source_path) as image:
            source = image.convert("RGB")
            expected = evaluation_transform_from_checkpoint(checkpoint)(source)
            actual = tensor_transform(model_letterbox(jpeg(canonical(source))))
        if not torch.equal(expected, actual):
            raise RuntimeError("Diagnostic original branch differs from checkpoint evaluation preprocessing")

        class VariantDataset(EvaluationDataset):
            def __init__(self, condition):
                self.condition = condition
                super().__init__(frame, args.data_root, tensor_transform, "diagnostic_variant", 299)

            def __getitem__(self, index):
                row = self.frame.iloc[index]
                with Image.open(self.data_root/row.source_path) as source:
                    base = canonical(source.convert("RGB"))
                changed = corrupt(base, self.condition, row.sample_id, protocol["noise_seed"])
                original_input = model_letterbox(jpeg(base))
                changed_input = model_letterbox(jpeg(changed))
                return {"image": tensor_transform(changed_input), "label": torch.tensor(0 if row.label == "real" else 1), "index": index, **pixel_quality(original_input, changed_input)}

        network = None
        scores, condition_rows, group_rows, quality_rows = {}, [], [], []
        reference_method_auc = {}
        for condition in conditions:
            name = condition["name"]
            out = folder/name
            cache = out/"predictions.csv"
            metadata_path = out/"metrics.json"
            if cache.is_file() and metadata_path.is_file() and (out/"input_quality.csv").is_file():
                predictions = pd.read_csv(cache, dtype={"sample_id": str})
                if predictions.sample_id.tolist() != frame.sample_id.tolist():
                    raise RuntimeError(f"Cached sample order differs: {cache}")
                metrics = json.loads(metadata_path.read_text())["metrics"]
                quality = pd.read_csv(out/"input_quality.csv")
                print(f"Reusing {selection}/{name}", flush=True)
            else:
                if network is None:
                    network = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=checkpoint["num_classes"])
                    network.load_state_dict(checkpoint["model_state_dict"])
                    network = network.to(device).eval()
                dataset = VariantDataset(condition)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                # Capture paired pixel diagnostics from the same batches as inference.
                batch_quality = []
                class QualityBatches:
                    def __len__(self):
                        return len(loader)

                    def __iter__(self):
                        for batch in loader:
                            batch_quality.extend({key: float(batch[key][i]) for key in ("psnr_db", "mean_absolute_pixel_difference", "mean_gradient_magnitude")} for i in range(len(batch["index"])))
                            yield batch
                metrics, labels, _, probabilities = evaluate_with_predictions(network, QualityBatches(), frame, nn.CrossEntropyLoss(), device, progress_desc=f"{selection} {name}", persistent_progress=True)
                predictions = frame[["sample_id", "method", "label"]].copy()
                predictions["fake_probability"] = probabilities
                quality = pd.DataFrame(batch_quality)
                quality.insert(0, "sample_id", frame.sample_id.to_numpy())
                out.mkdir(parents=True, exist_ok=True)
                atomic_csv(predictions, cache)
                atomic_csv(quality, out/"input_quality.csv")
                write_json(metadata_path, {"condition": condition, "images": len(frame), "metrics": metrics, "threshold": .5})
            scores[name] = predictions.fake_probability.to_numpy(float)
            pairs, groups = paired_summaries(frame, scores["original"], scores[name], name, selection, .5)
            atomic_csv(pairs, out/"paired_predictions.csv")
            group_rows.extend(groups)
            real = next(row for row in groups if row["level"] == "label" and row["group"] == "real")
            fake = next(row for row in groups if row["level"] == "label" and row["group"] == "fake")
            per_method = {method: metrics["by_manipulation"][f"original_vs_{method}"]["roc_auc"] for method in methods}
            if name == "original":
                reference_method_auc = per_method.copy()
            condition_rows.append({"selection": selection, "training_seed": 42, "condition": name, "kind": condition["kind"], "strength": condition["strength"], "images": len(frame), "seen_macro_auc": float(np.mean(list(per_method.values()))), "real_fpr": real["real_fpr_after"], "fake_recall": fake["fake_recall_after"], "real_mean_delta_fake_score": real["mean_delta"], "fake_mean_delta_fake_score": fake["mean_delta"], "real_mean_fake_score": real["mean_fake_score_after"], "fake_mean_fake_score": fake["mean_fake_score_after"]})
            for row in groups:
                if row["level"] == "method" and row["group"] in per_method:
                    row["roc_auc"] = per_method[row["group"]]
                    row["original_roc_auc"] = reference_method_auc[row["group"]]
                    row["roc_auc_delta"] = row["roc_auc"] - row["original_roc_auc"]
            qrow = {"selection": selection, "condition": name, "mean_psnr_db": float(quality.psnr_db.mean()), "mean_absolute_pixel_difference": float(quality.mean_absolute_pixel_difference.mean()), "mean_gradient_magnitude": float(quality.mean_gradient_magnitude.mean())}
            quality_rows.append(qrow)
            print(f"{selection} {name}: Real FPR={real['real_fpr_after']:.3%}, seen macro AUC={condition_rows[-1]['seen_macro_auc']:.4f}", flush=True)
        baseline = condition_rows[0]
        for row in condition_rows:
            row["seen_macro_auc_delta"] = row["seen_macro_auc"] - baseline["seen_macro_auc"]
            row["real_fpr_delta"] = row["real_fpr"] - baseline["real_fpr"]
        atomic_csv(pd.DataFrame(condition_rows), folder/"condition_summary.csv")
        atomic_csv(pd.DataFrame(group_rows), folder/"paired_group_summary.csv")
        atomic_csv(pd.DataFrame(quality_rows), folder/"input_quality_summary.csv")

        visuals = folder/"visuals"
        visuals.mkdir(exist_ok=True)
        from PIL import ImageDraw
        for method, row in frame.groupby("method", sort=True).first().iterrows():
            with Image.open(args.data_root/row.source_path) as source:
                base = canonical(source.convert("RGB"))
            sheet = Image.new("RGB", (299*len(conditions), 324), "white")
            draw = ImageDraw.Draw(sheet)
            for index, condition in enumerate(conditions):
                rendered = model_letterbox(jpeg(corrupt(base, condition, row.sample_id, protocol['noise_seed'])))
                sheet.paste(rendered, (299*index, 25))
                draw.text((299*index+3, 5), condition["name"], fill="black")
            sheet.save(visuals/(method.replace("/", "_")+".png"))
        all_conditions.extend(condition_rows)
        all_groups.extend(group_rows)
        del network, checkpoint
        torch.cuda.empty_cache()
    atomic_csv(pd.DataFrame(all_conditions), args.output_root/"condition_summary.csv")
    atomic_csv(pd.DataFrame(all_groups), args.output_root/"paired_group_summary.csv")
    summary = pd.DataFrame(all_conditions)
    lines = ["# Validation corruption diagnosis", "", "Frozen seed42 Mixed-JPEG M7; seen validation only. No training or threshold tuning.", "", "AUC loss alone is not evidence of a shortcut. Inspect Real score/FPR shifts and Fake recall together.", "", "| Selection | Condition | Seen AUC | AUC delta | Real FPR | FPR delta | Real score delta |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.selection} | {row.condition} | {row.seen_macro_auc:.4f} | {row.seen_macro_auc_delta:+.4f} | {row.real_fpr:.2%} | {row.real_fpr_delta:+.2%} | {row.real_mean_delta_fake_score:+.4f} |")
    (args.output_root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(args.output_root/"diagnostic_summary.json", {"protocol": protocol, "selections": args.selections, "full_validation": args.full_validation, "training_seed": 42, "test_used": False, "unseen_used": False, "wilddeepfake_used": False})
    print("Diagnostic results saved:", args.output_root, flush=True)


if __name__ == "__main__":
    main()
