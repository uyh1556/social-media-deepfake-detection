#!/usr/bin/env python3
"""Diagnose whether frozen Xception models treat lower image quality as fake.

The diagnostic is counterfactual and read-only with respect to the source data:
quality transformations are applied in memory to the same real test images.  The
fixed 0.5 decision threshold and each checkpoint's saved Letterbox transform are
reused without training or test-time tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import timm
import torch
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from evaluate_xception_family_coverage import (
    checkpoint_path,
    cross_split_audit,
    read_manifest,
    validate_checkpoint,
    validate_test_manifest,
)
from train_baseline import sha256_file, write_json
from xception_preprocessing import (
    LETTERBOX_NAME,
    Letterbox,
    evaluation_transform_from_checkpoint,
)


EXPECTED_IMAGE_SIZE = (256, 256)
MODEL_VISIBLE_SIZE = 299
MODEL_IDS = ("M3", "M5", "M7")
QUALITY_METRICS = (
    "laplacian_variance",
    "mean_gradient_magnitude",
    "high_frequency_energy_ratio",
    "brightness",
    "contrast",
    "edge_density",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--existing-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions-config",
        type=Path,
        default=project_root / "configs/family_coverage_v1/conditions.json",
    )
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=(
            project_root
            / "configs/quality_shortcut_diagnostic_v1/protocol.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{seed}|{namespace}|{value}".encode("utf-8")
    ).hexdigest()


def fixed_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"Requested {count} samples from only {len(frame)} rows")
    ranked = frame.assign(
        _sample_rank=frame["sample_id"].map(
            lambda value: stable_key(seed, "quality-calibration", str(value))
        )
    )
    return (
        ranked.sort_values("_sample_rank")
        .head(count)
        .drop(columns="_sample_rank")
        .reset_index(drop=True)
    )


def validate_protocol(config: dict) -> None:
    if tuple(config["models"]) != MODEL_IDS:
        raise ValueError(f"Diagnostic models must be {MODEL_IDS}")
    if float(config["classification_threshold"]) != 0.5:
        raise ValueError("This diagnostic is frozen to the existing 0.5 threshold")
    names = [item["name"] for item in config["transformations"]]
    if len(names) != len(set(names)) or names[0] != "original":
        raise ValueError("Transformation names must be unique and start with original")
    if tuple(config["quality_metrics"]) != QUALITY_METRICS:
        raise ValueError("Unexpected quality metric definition")
    if not set(config["quality_matching_metrics"]).issubset(QUALITY_METRICS):
        raise ValueError("Quality-matching metrics must be reported pixel metrics")


def transform_image(image: Image.Image, condition: dict) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Expected a 256x256 canonical image, found {image.size}"
        )
    kind = condition["kind"]
    if kind == "identity":
        return image.copy()
    if kind == "downsample_upsample":
        short_size = int(condition["short_size"])
        restore_size = int(condition["restore_size"])
        reduced = image.resize(
            (short_size, short_size),
            resample=Image.Resampling.BICUBIC,
        )
        return reduced.resize(
            (restore_size, restore_size),
            resample=Image.Resampling.BICUBIC,
        )
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(float(condition["sigma"])))
    if kind == "jpeg_round_trip":
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(condition["quality"]),
            subsampling=int(condition["subsampling"]),
            optimize=bool(condition["optimize"]),
            progressive=bool(condition["progressive"]),
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    raise ValueError(f"Unsupported transformation kind: {kind}")


def image_quality_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    gradient_y, gradient_x = np.gradient(gray)
    gradient = np.hypot(gradient_x, gradient_y)
    padded = np.pad(gray, 1, mode="reflect")
    laplacian = (
        padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        - 4.0 * gray
    )
    centered = gray - float(gray.mean())
    spectrum = np.fft.fft2(centered)
    power = np.abs(spectrum) ** 2
    frequency_y = np.fft.fftfreq(gray.shape[0])[:, None]
    frequency_x = np.fft.fftfreq(gray.shape[1])[None, :]
    high_frequency = np.hypot(frequency_x, frequency_y) > 0.25
    total_energy = float(power.sum())
    high_frequency_ratio = (
        float(power[high_frequency].sum() / total_energy)
        if total_energy > 0
        else 0.0
    )
    return {
        "laplacian_variance": float(laplacian.var()),
        "mean_gradient_magnitude": float(gradient.mean()),
        "high_frequency_energy_ratio": high_frequency_ratio,
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": float((gradient > 20.0).mean()),
    }


def model_visible_quality_image(image: Image.Image) -> Image.Image:
    """Apply the frozen geometric input policy without tensor normalization."""
    return Letterbox(size=MODEL_VISIBLE_SIZE)(image.convert("RGB"))


def threaded_map(function, values, workers: int, description: str) -> list:
    values = list(values)
    if workers <= 1:
        return [function(value) for value in tqdm(values, desc=description)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(function, values),
                total=len(values),
                desc=description,
            )
        )


def base_quality_row(row, condition: str, metrics: dict) -> dict:
    return {
        "sample_id": str(row.sample_id),
        "split": str(row.split),
        "label": str(row.label),
        "method": str(row.method),
        "family": str(row.family),
        "group_id": str(row.group_id),
        "video_id": str(row.video_id),
        "source_path": str(row.source_path),
        "condition": condition,
        **metrics,
    }


def compute_original_quality(
    frame: pd.DataFrame,
    data_root: Path,
    workers: int,
) -> pd.DataFrame:
    def process(row):
        with Image.open(data_root / row.source_path) as image:
            rgb = image.convert("RGB")
            metrics = image_quality_metrics(model_visible_quality_image(rgb))
        return base_quality_row(row, "original", metrics)

    rows = threaded_map(
        process,
        frame.itertuples(index=False),
        workers,
        "Canonical test quality",
    )
    return pd.DataFrame(rows)


def compute_condition_quality(
    frame: pd.DataFrame,
    data_root: Path,
    conditions: list[dict],
    workers: int,
    description: str,
) -> pd.DataFrame:
    def process(row):
        with Image.open(data_root / row.source_path) as image:
            rgb = image.convert("RGB")
        rows = []
        for condition in conditions:
            transformed = transform_image(rgb, condition)
            rows.append(
                base_quality_row(
                    row,
                    condition["name"],
                    image_quality_metrics(
                        model_visible_quality_image(transformed)
                    ),
                )
            )
        return rows

    nested = threaded_map(
        process,
        frame.itertuples(index=False),
        workers,
        description,
    )
    return pd.DataFrame([row for group in nested for row in group])


def prepare_calibration_samples(
    manifest: pd.DataFrame,
    config: dict,
    seed: int,
) -> pd.DataFrame:
    calibration = config["calibration"]
    split = calibration["split"]
    methods = [calibration["source_method"], *calibration["target_methods"]]
    count = int(calibration["samples_per_method"])
    parts = []
    for method in methods:
        candidates = manifest[
            (manifest["split"] == split) & (manifest["method"] == method)
        ]
        sampled = fixed_sample(candidates, count, seed).copy()
        sampled["calibration_role"] = (
            "source_real"
            if method == calibration["source_method"]
            else "target_efs"
        )
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def compute_calibration_quality(
    samples: pd.DataFrame,
    data_root: Path,
    conditions: list[dict],
    source_method: str,
    workers: int,
) -> pd.DataFrame:
    source = samples[samples["method"] == source_method]
    targets = samples[samples["method"] != source_method]
    source_quality = compute_condition_quality(
        source,
        data_root,
        conditions,
        workers,
        "Calibration real transformations",
    )
    target_quality = compute_original_quality(targets, data_root, workers)
    return pd.concat([source_quality, target_quality], ignore_index=True)


def quantile_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 201)
    return float(
        np.mean(
            np.abs(
                np.quantile(left, quantiles)
                - np.quantile(right, quantiles)
            )
        )
    )


def matching_scales(
    calibration_quality: pd.DataFrame,
    source_method: str,
    matching_metrics: list[str],
) -> dict[str, float]:
    reference = calibration_quality[
        (calibration_quality["condition"] == "original")
        | (calibration_quality["method"] != source_method)
    ]
    scales = {}
    for metric in matching_metrics:
        values = reference[metric].to_numpy(dtype=float)
        scale = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        if scale <= 1e-12:
            scale = float(np.std(values))
        scales[metric] = max(scale, 1e-12)
    return scales


def quality_matching_table(
    source_quality: pd.DataFrame,
    target_quality: pd.DataFrame,
    conditions: list[dict],
    target_methods: list[str],
    scales: dict[str, float],
    matching_metrics: list[str],
) -> pd.DataFrame:
    target_definitions = {
        method: target_quality[target_quality["method"] == method]
        for method in target_methods
    }
    target_definitions["EFS_pooled"] = target_quality[
        target_quality["method"].isin(target_methods)
    ]
    rows = []
    for target_name, target in target_definitions.items():
        if target.empty:
            raise ValueError(f"No quality rows for matching target {target_name}")
        for condition in conditions:
            candidate = source_quality[
                source_quality["condition"] == condition["name"]
            ]
            row = {
                "target": target_name,
                "condition": condition["name"],
                "kind": condition["kind"],
                "severity": int(condition["severity"]),
                "source_images": len(candidate),
                "target_images": len(target),
            }
            normalized = []
            for metric in matching_metrics:
                raw_distance = quantile_wasserstein(
                    candidate[metric].to_numpy(float),
                    target[metric].to_numpy(float),
                )
                row[f"{metric}_wasserstein"] = raw_distance
                row[f"{metric}_normalized_distance"] = (
                    raw_distance / scales[metric]
                )
                normalized.append(raw_distance / scales[metric])
            row["mean_normalized_distance"] = float(np.mean(normalized))
            rows.append(row)
    result = pd.DataFrame(rows)
    result["selected_on_calibration"] = False
    for target in result["target"].unique():
        candidates = result[result["target"] == target]
        best_index = candidates["mean_normalized_distance"].idxmin()
        result.loc[best_index, "selected_on_calibration"] = True
    return result


def validate_existing_predictions(
    path: Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"source_path", "label", "method", "fake_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns missing in {path}: {sorted(missing)}")
    if len(predictions) != len(frame):
        raise ValueError(f"Prediction/test row mismatch in {path}")
    for column in ("source_path", "label", "method"):
        if predictions[column].tolist() != frame[column].tolist():
            raise RuntimeError(f"Prediction order mismatch for {column} in {path}")
    predictions["fake_probability"] = pd.to_numeric(
        predictions["fake_probability"], errors="raise"
    )
    if not predictions["fake_probability"].between(0.0, 1.0).all():
        raise ValueError(f"Invalid probabilities in {path}")
    return predictions


class CounterfactualRealDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        condition: dict,
        model_transform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.condition = condition
        self.model_transform = model_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(self.data_root / row["source_path"]) as image:
            transformed = transform_image(image.convert("RGB"), self.condition)
            tensor = self.model_transform(transformed)
        return {"image": tensor, "index": index}


@torch.no_grad()
def predict_condition(
    model,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[list[int], list[float]]:
    model.eval()
    indices = []
    scores = []
    for batch in tqdm(loader, desc=description, leave=True):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        indices.extend(batch["index"].tolist())
        scores.extend(probabilities.float().cpu().tolist())
    return indices, scores


def infer_counterfactual_conditions(
    model_id: str,
    checkpoint_file: Path,
    real_test: pd.DataFrame,
    data_root: Path,
    conditions: list[dict],
    output_dir: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    reuse_cache: bool,
) -> pd.DataFrame:
    model_output = output_dir / "counterfactual_predictions" / model_id.lower()
    model_output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    if checkpoint["config"].get("preprocessing_name") != LETTERBOX_NAME:
        raise ValueError(f"{model_id} is not a Letterbox checkpoint")
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model_transform = evaluation_transform_from_checkpoint(checkpoint)

    for condition in conditions:
        output_path = model_output / f"{condition['name']}.csv"
        if output_path.is_file() and reuse_cache:
            completed = pd.read_csv(output_path)
            if (
                len(completed) == len(real_test)
                and completed["sample_id"].astype(str).tolist()
                == real_test["sample_id"].astype(str).tolist()
            ):
                print(f"Skipping completed {model_id} / {condition['name']}")
                continue
            raise RuntimeError(f"Incomplete prediction file exists: {output_path}")
        dataset = CounterfactualRealDataset(
            real_test,
            data_root,
            condition,
            model_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
        )
        indices, scores = predict_condition(
            model,
            loader,
            device,
            f"{model_id} {condition['name']}",
        )
        if indices != list(range(len(real_test))):
            raise RuntimeError("Counterfactual inference order changed")
        result = pd.DataFrame(
            {
                "sample_id": real_test["sample_id"],
                "group_id": real_test["group_id"],
                "video_id": real_test["video_id"],
                "source_path": real_test["source_path"],
                "condition": condition["name"],
                "fake_probability": scores,
            }
        )
        atomic_to_csv(result, output_path)

    del model, checkpoint
    torch.cuda.empty_cache()
    frames = []
    for condition in conditions:
        frame = pd.read_csv(
            model_output / f"{condition['name']}.csv",
            dtype={"sample_id": str, "group_id": str, "video_id": str},
            keep_default_na=False,
        )
        if frame["sample_id"].tolist() != real_test["sample_id"].tolist():
            raise RuntimeError(f"Stored prediction order mismatch for {model_id}")
        frame.insert(0, "model", model_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    delta_column: str,
    fp_delta_column: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    grouped = frame.groupby("group_id", sort=True)
    groups = []
    for _, group in grouped:
        groups.append(
            (
                float(group[delta_column].sum()),
                float(group[fp_delta_column].sum()),
                len(group),
            )
        )
    values = np.asarray(groups, dtype=float)
    rng = np.random.default_rng(seed)
    delta_samples = np.empty(replicates, dtype=float)
    fp_samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        sampled = values[selected]
        count = sampled[:, 2].sum()
        delta_samples[index] = sampled[:, 0].sum() / count
        fp_samples[index] = sampled[:, 1].sum() / count
    return {
        "mean_delta_ci_low": float(np.quantile(delta_samples, 0.025)),
        "mean_delta_ci_high": float(np.quantile(delta_samples, 0.975)),
        "fpr_delta_ci_low": float(np.quantile(fp_samples, 0.025)),
        "fpr_delta_ci_high": float(np.quantile(fp_samples, 0.975)),
    }


def summarize_counterfactuals(
    predictions: pd.DataFrame,
    conditions: list[dict],
    threshold: float,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_lookup = {item["name"]: item for item in conditions}
    paired_frames = []
    summary_rows = []
    for model_id, model_frame in predictions.groupby("model", sort=True):
        original = (
            model_frame[model_frame["condition"] == "original"]
            .set_index("sample_id")["fake_probability"]
            .astype(float)
        )
        for condition_name, condition_frame in model_frame.groupby(
            "condition", sort=False
        ):
            current = condition_frame.copy()
            current["fake_probability"] = current["fake_probability"].astype(float)
            current["fake_probability_original"] = current["sample_id"].map(original)
            current["delta_fake_probability"] = (
                current["fake_probability"]
                - current["fake_probability_original"]
            )
            current["original_false_positive"] = (
                current["fake_probability_original"] >= threshold
            ).astype(int)
            current["condition_false_positive"] = (
                current["fake_probability"] >= threshold
            ).astype(int)
            current["false_positive_delta"] = (
                current["condition_false_positive"]
                - current["original_false_positive"]
            )
            paired_frames.append(current)
            if condition_name == "original":
                continue
            deltas = current["delta_fake_probability"].to_numpy(float)
            bootstrap_seed = int(
                stable_key(seed, model_id, condition_name)[:8], 16
            )
            intervals = cluster_bootstrap_interval(
                current,
                "delta_fake_probability",
                "false_positive_delta",
                bootstrap_replicates,
                bootstrap_seed,
            )
            definition = condition_lookup[condition_name]
            summary_rows.append(
                {
                    "model": model_id,
                    "condition": condition_name,
                    "kind": definition["kind"],
                    "severity": int(definition["severity"]),
                    "images": len(current),
                    "groups": int(current["group_id"].nunique()),
                    "original_mean_fake_score": float(
                        current["fake_probability_original"].mean()
                    ),
                    "degraded_mean_fake_score": float(
                        current["fake_probability"].mean()
                    ),
                    "mean_delta_fake_score": float(deltas.mean()),
                    "median_delta_fake_score": float(np.median(deltas)),
                    "std_delta_fake_score": float(deltas.std()),
                    "delta_q025": float(np.quantile(deltas, 0.025)),
                    "delta_q25": float(np.quantile(deltas, 0.25)),
                    "delta_q75": float(np.quantile(deltas, 0.75)),
                    "delta_q975": float(np.quantile(deltas, 0.975)),
                    "fake_score_increase_rate": float((deltas > 0).mean()),
                    "original_fpr": float(
                        current["original_false_positive"].mean()
                    ),
                    "degraded_fpr": float(
                        current["condition_false_positive"].mean()
                    ),
                    "fpr_delta": float(current["false_positive_delta"].mean()),
                    **intervals,
                }
            )
    return pd.concat(paired_frames, ignore_index=True), pd.DataFrame(summary_rows)


def baseline_reproduction_table(
    predictions: pd.DataFrame,
    existing_scores: dict[str, pd.DataFrame],
    real_test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        current = predictions[
            (predictions["model"] == model_id)
            & (predictions["condition"] == "original")
        ].reset_index(drop=True)
        previous = existing_scores[model_id]
        previous = previous[previous["method"] == "original"].reset_index(drop=True)
        if current["source_path"].tolist() != real_test["source_path"].tolist():
            raise RuntimeError(f"Current original order mismatch for {model_id}")
        if previous["source_path"].tolist() != real_test["source_path"].tolist():
            raise RuntimeError(f"Previous original order mismatch for {model_id}")
        current_score = current["fake_probability"].to_numpy(float)
        previous_score = previous["fake_probability"].to_numpy(float)
        absolute = np.abs(current_score - previous_score)
        rows.append(
            {
                "model": model_id,
                "images": len(current),
                "previous_mean_fake_score": float(previous_score.mean()),
                "current_mean_fake_score": float(current_score.mean()),
                "mean_absolute_probability_difference": float(absolute.mean()),
                "max_absolute_probability_difference": float(absolute.max()),
                "classification_agreement": float(
                    (
                        (current_score >= threshold)
                        == (previous_score >= threshold)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = pd.Series(left).rank(method="average").to_numpy(float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(float)
    return pearson(left_rank, right_rank)


def quality_score_correlations(
    test_quality: pd.DataFrame,
    scores_by_model: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    base_columns = [
        "sample_id",
        "method",
        "family",
        "group_id",
        "video_id",
        *QUALITY_METRICS,
    ]
    for model_id, predictions in scores_by_model.items():
        scores = predictions[["source_path", "fake_probability"]].copy()
        merged = test_quality[base_columns + ["source_path"]].merge(
            scores,
            on="source_path",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(test_quality):
            raise RuntimeError(f"Quality/prediction merge mismatch for {model_id}")
        for method, method_frame in merged.groupby("method", sort=True):
            analysis_levels = {"image": method_frame}
            grouped = (
                method_frame.groupby("video_id", as_index=False)[
                    [*QUALITY_METRICS, "fake_probability"]
                ]
                .mean()
            )
            analysis_levels["logical_video_mean"] = grouped
            for level, analysis_frame in analysis_levels.items():
                for metric in QUALITY_METRICS:
                    rows.append(
                        {
                            "model": model_id,
                            "method": method,
                            "family": method_frame["family"].iloc[0],
                            "level": level,
                            "metric": metric,
                            "observations": len(analysis_frame),
                            "spearman": spearman(
                                analysis_frame[metric].to_numpy(float),
                                analysis_frame["fake_probability"].to_numpy(float),
                            ),
                            "pearson": pearson(
                                analysis_frame[metric].to_numpy(float),
                                analysis_frame["fake_probability"].to_numpy(float),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def dose_response_summary(
    paired: pd.DataFrame,
    families: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for model_id, model_frame in paired.groupby("model", sort=True):
        wide = model_frame.pivot(
            index="sample_id",
            columns="condition",
            values="fake_probability",
        )
        for family, ordered_conditions in families.items():
            values = wide[ordered_conditions].to_numpy(float)
            mean_scores = values.mean(axis=0)
            differences = np.diff(values, axis=1)
            nondecreasing = np.all(differences >= -1e-8, axis=1)
            rows.append(
                {
                    "model": model_id,
                    "degradation_family": family,
                    "conditions": "|".join(ordered_conditions),
                    "mean_fake_scores": "|".join(
                        f"{value:.10f}" for value in mean_scores
                    ),
                    "mean_score_spearman_vs_severity": spearman(
                        np.arange(len(mean_scores)), mean_scores
                    ),
                    "imagewise_nondecreasing_rate": float(nondecreasing.mean()),
                    "final_minus_original_mean_score": float(
                        mean_scores[-1] - mean_scores[0]
                    ),
                }
            )
    return pd.DataFrame(rows)


def quality_summary(test_quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in test_quality.groupby("method", sort=True):
        for metric in QUALITY_METRICS:
            values = frame[metric].to_numpy(float)
            rows.append(
                {
                    "method": method,
                    "family": frame["family"].iloc[0],
                    "metric": metric,
                    "images": len(frame),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std()),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    counterfactual_summary: pd.DataFrame,
    real_correlations: pd.DataFrame,
    calibration_matching: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    conditions = counterfactual_summary["condition"].drop_duplicates().tolist()
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for axis, model_id in zip(axes, MODEL_IDS):
        frame = counterfactual_summary[
            counterfactual_summary["model"] == model_id
        ].set_index("condition").loc[conditions]
        positions = np.arange(len(frame))
        lower = frame["mean_delta_fake_score"] - frame["mean_delta_ci_low"]
        upper = frame["mean_delta_ci_high"] - frame["mean_delta_fake_score"]
        axis.bar(positions, frame["mean_delta_fake_score"], color="#4c78a8")
        axis.errorbar(
            positions,
            frame["mean_delta_fake_score"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(f"{model_id} mean delta")
    axes[-1].set_xticks(np.arange(len(conditions)))
    axes[-1].set_xticklabels(conditions, rotation=35, ha="right")
    figure.suptitle("Real counterfactual fake-score shift (cluster-bootstrap 95% CI)")
    figure.tight_layout()
    figure.savefig(plot_dir / "counterfactual_score_shift.png", dpi=180)
    plt.close(figure)

    image_real = real_correlations[real_correlations["level"] == "image"]
    matrix = image_real.pivot(index="model", columns="metric", values="spearman")
    matrix = matrix.reindex(index=MODEL_IDS, columns=QUALITY_METRICS)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Spearman correlation")
    axis.set_title("Real-only quality metric vs fake score")
    figure.tight_layout()
    figure.savefig(plot_dir / "real_quality_score_correlations.png", dpi=180)
    plt.close(figure)

    pooled = calibration_matching[calibration_matching["target"] == "EFS_pooled"]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    colors = [
        "#e45756" if selected else "#72b7b2"
        for selected in pooled["selected_on_calibration"]
    ]
    axis.bar(
        np.arange(len(pooled)),
        pooled["mean_normalized_distance"],
        color=colors,
    )
    axis.set_xticks(np.arange(len(pooled)))
    axis.set_xticklabels(pooled["condition"], rotation=35, ha="right")
    axis.set_ylabel("Mean normalized Wasserstein distance")
    axis.set_title("Real degradation quality match to EFS training data")
    figure.tight_layout()
    figure.savefig(plot_dir / "efs_quality_matching.png", dpi=180)
    plt.close(figure)


def build_report(
    summary: pd.DataFrame,
    dose: pd.DataFrame,
    real_correlations: pd.DataFrame,
    calibration_matching: pd.DataFrame,
) -> str:
    lines = [
        "# Quality Shortcut Diagnostic",
        "",
        "> Diagnostic-only evaluation of frozen M3, M5, and M7 checkpoints. No model was retrained and the decision threshold remained 0.5.",
        "",
        "## Calibration-selected quality matches",
        "",
        "Selection used only M3 training images and pixel-quality metrics; model scores and test images were not used.",
        "",
        "| Target | Selected Real transformation | Normalized distance |",
        "|---|---|---:|",
    ]
    selected = calibration_matching[
        calibration_matching["selected_on_calibration"]
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.condition} | {row.mean_normalized_distance:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Real counterfactual shifts",
            "",
            "| Model | Strongest positive condition | Mean score delta | 95% CI | FPR delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_id in MODEL_IDS:
        frame = summary[summary["model"] == model_id]
        strongest = frame.loc[frame["mean_delta_fake_score"].idxmax()]
        lines.append(
            f"| {model_id} | {strongest['condition']} | "
            f"{strongest['mean_delta_fake_score']:+.4f} | "
            f"[{strongest['mean_delta_ci_low']:+.4f}, {strongest['mean_delta_ci_high']:+.4f}] | "
            f"{strongest['fpr_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dose response",
            "",
            "| Model | Family | Mean-score Spearman | Imagewise nondecreasing rate | Final delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in dose.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.degradation_family} | "
            f"{row.mean_score_spearman_vs_severity:+.3f} | "
            f"{row.imagewise_nondecreasing_rate:.3f} | "
            f"{row.final_minus_original_mean_score:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Real-only correlations",
            "",
            "Negative correlations for sharpness, gradient, high-frequency energy, contrast, or edge density are consistent with `lower quality -> higher fake score`.",
            "",
            "| Model | Metric | Image Spearman | Video-mean Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for model_id in MODEL_IDS:
        for metric in QUALITY_METRICS:
            subset = real_correlations[
                (real_correlations["model"] == model_id)
                & (real_correlations["metric"] == metric)
            ].set_index("level")
            image_value = subset.loc["image", "spearman"]
            video_value = subset.loc["logical_video_mean", "spearman"]
            lines.append(
                f"| {model_id} | {metric} | {image_value:+.3f} | {video_value:+.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A positive paired shift with increasing degradation strength supports quality sensitivity because image content is held fixed. It does not by itself prove that every EFS prediction is caused by quality: generator-specific artifacts may coexist. Quality matching is based on a fixed transformation grid; if no candidate closely matches EFS, the result must be reported as unmatched rather than treated as an EFS simulation.",
            "",
        ]
    )
    return "\n".join(lines)


def versions() -> dict[str, str]:
    import matplotlib

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "pillow": PIL.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }


def main() -> None:
    args = parse_args()
    paths = {
        "data_root": args.data_root.resolve(),
        "test_manifest": args.test_manifest.resolve(),
        "training_manifest_dir": args.training_manifest_dir.resolve(),
        "runs_root": args.runs_root.resolve(),
        "existing_evaluation_root": args.existing_evaluation_root.resolve(),
        "output_dir": args.output_dir.resolve(),
        "conditions_config": args.conditions_config.resolve(),
        "diagnostic_config": args.diagnostic_config.resolve(),
    }
    for name, path in paths.items():
        if name != "output_dir" and not path.exists():
            raise FileNotFoundError(path)

    family_config = json.loads(
        paths["conditions_config"].read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        paths["diagnostic_config"].read_text(encoding="utf-8")
    )
    validate_protocol(diagnostic)
    conditions = diagnostic["transformations"]
    threshold = float(diagnostic["classification_threshold"])

    test_frame = read_manifest(paths["test_manifest"])
    validate_test_manifest(test_frame)
    real_test = test_frame[test_frame["method"] == "original"].reset_index(drop=True)
    if len(real_test) != 2000:
        raise ValueError(f"Expected 2,000 canonical real test images, found {len(real_test)}")

    training_manifests = {}
    checkpoint_files = {}
    checkpoint_hashes = {}
    existing_scores = {}
    for model_id in MODEL_IDS:
        condition = family_config["conditions"][model_id]
        training_manifest = (
            paths["training_manifest_dir"] / f"{model_id.lower()}_seed42.csv"
        )
        checkpoint_file = checkpoint_path(
            paths["runs_root"], model_id, condition, args.seed
        )
        for path in (training_manifest, checkpoint_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        validate_checkpoint(
            checkpoint,
            checkpoint_file,
            training_manifest,
            model_id,
            condition,
        )
        if checkpoint["config"].get("preprocessing_name") != LETTERBOX_NAME:
            raise ValueError(f"Unexpected preprocessing for {model_id}")
        development = read_manifest(training_manifest)
        cross_split_audit(development, test_frame)
        training_manifests[model_id] = training_manifest
        checkpoint_files[model_id] = checkpoint_file
        checkpoint_hashes[model_id] = sha256_file(checkpoint_file)
        del checkpoint

        evaluation_dir = paths["existing_evaluation_root"] / model_id.lower()
        metrics_path = evaluation_dir / "metrics.json"
        predictions_path = evaluation_dir / "predictions.csv"
        for path in (metrics_path, predictions_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics["test_manifest"]["sha256"] != sha256_file(paths["test_manifest"]):
            raise RuntimeError(f"Existing {model_id} evaluation used another test manifest")
        if metrics["checkpoint"]["sha256"] != checkpoint_hashes[model_id]:
            raise RuntimeError(f"Existing {model_id} evaluation used another checkpoint")
        existing_scores[model_id] = validate_existing_predictions(
            predictions_path, test_frame
        )

    output_dir = paths["output_dir"]
    metadata_path = output_dir / "run_metadata.json"
    invalidate_cached_artifacts = False
    expected_metadata = {
        "protocol": diagnostic["protocol"],
        "seed": args.seed,
        "test_manifest_sha256": sha256_file(paths["test_manifest"]),
        "diagnostic_config_sha256": sha256_file(paths["diagnostic_config"]),
        "conditions_config_sha256": sha256_file(paths["conditions_config"]),
        "checkpoint_sha256": checkpoint_hashes,
        "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        if not metadata_path.is_file():
            raise FileExistsError(
                f"Non-empty directory is not a resumable diagnostic: {output_dir}"
            )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "protocol",
            "seed",
            "test_manifest_sha256",
            "conditions_config_sha256",
            "checkpoint_sha256",
        )
        for key in invariant_keys:
            value = expected_metadata[key]
            if existing_metadata.get(key) != value:
                raise RuntimeError(f"Diagnostic resume mismatch for {key}")
        changed_implementation = any(
            existing_metadata.get(key) != expected_metadata[key]
            for key in (
                "diagnostic_config_sha256",
                "diagnostic_script_sha256",
            )
        )
        if changed_implementation:
            if (output_dir / "diagnostic_summary.json").is_file():
                raise RuntimeError(
                    "A completed diagnostic used an older implementation. "
                    "Preserve it and choose a different output directory."
                )
            invalidate_cached_artifacts = True
            print(
                "Refreshing incomplete diagnostic artifacts after a code/config fix",
                flush=True,
            )
            write_json(
                metadata_path,
                {
                    **existing_metadata,
                    **expected_metadata,
                    "implementation_refreshed_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            metadata_path,
            {
                **expected_metadata,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "decision_rule": "fixed fake probability >= 0.5",
                "test_threshold_tuning": False,
                "training_or_finetuning": False,
                "source_images_modified": False,
                "paths": {key: str(value) for key, value in paths.items()},
                "versions": versions(),
            },
        )

    m3_manifest = read_manifest(training_manifests["M3"])
    calibration_samples_path = output_dir / "calibration_samples.csv"
    if calibration_samples_path.is_file() and not invalidate_cached_artifacts:
        calibration_samples = read_manifest(calibration_samples_path)
    else:
        calibration_samples = prepare_calibration_samples(
            m3_manifest, diagnostic, args.seed
        )
        atomic_to_csv(calibration_samples, calibration_samples_path)
    required_paths = pd.concat(
        [test_frame[["source_path"]], calibration_samples[["source_path"]]],
        ignore_index=True,
    )["source_path"].drop_duplicates()
    missing = [
        value for value in required_paths if not (paths["data_root"] / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required archives are not fully extracted. First missing paths: "
            + ", ".join(missing[:5])
        )

    calibration_quality_path = output_dir / "calibration_quality_metrics.csv"
    if calibration_quality_path.is_file() and not invalidate_cached_artifacts:
        calibration_quality = pd.read_csv(calibration_quality_path)
    else:
        calibration_quality = compute_calibration_quality(
            calibration_samples,
            paths["data_root"],
            conditions,
            diagnostic["calibration"]["source_method"],
            args.metric_workers,
        )
        atomic_to_csv(calibration_quality, calibration_quality_path)

    source_method = diagnostic["calibration"]["source_method"]
    target_methods = diagnostic["calibration"]["target_methods"]
    matching_metrics = diagnostic["quality_matching_metrics"]
    scales = matching_scales(
        calibration_quality, source_method, matching_metrics
    )
    source_calibration_quality = calibration_quality[
        calibration_quality["method"] == source_method
    ]
    target_calibration_quality = calibration_quality[
        calibration_quality["method"].isin(target_methods)
    ]
    calibration_matching = quality_matching_table(
        source_calibration_quality,
        target_calibration_quality,
        conditions,
        target_methods,
        scales,
        matching_metrics,
    )
    atomic_to_csv(calibration_matching, output_dir / "quality_matching_calibration.csv")

    test_quality_path = output_dir / "canonical_test_quality_metrics.csv"
    if test_quality_path.is_file() and not invalidate_cached_artifacts:
        test_quality = pd.read_csv(test_quality_path)
    else:
        test_quality = compute_original_quality(
            test_frame,
            paths["data_root"],
            args.metric_workers,
        )
        atomic_to_csv(test_quality, test_quality_path)
    atomic_to_csv(quality_summary(test_quality), output_dir / "quality_summary_by_method.csv")

    real_condition_quality_path = output_dir / "real_condition_quality_metrics.csv"
    if real_condition_quality_path.is_file() and not invalidate_cached_artifacts:
        real_condition_quality = pd.read_csv(real_condition_quality_path)
    else:
        real_condition_quality = compute_condition_quality(
            real_test,
            paths["data_root"],
            conditions,
            args.metric_workers,
            "Real test condition quality",
        )
        atomic_to_csv(real_condition_quality, real_condition_quality_path)

    target_test_quality = test_quality[test_quality["method"].isin(target_methods)]
    test_matching = quality_matching_table(
        real_condition_quality,
        target_test_quality,
        conditions,
        target_methods,
        scales,
        matching_metrics,
    )
    test_matching = test_matching.rename(
        columns={"selected_on_calibration": "best_on_test_quality_only"}
    )
    selected_lookup = {
        row.target: row.condition
        for row in calibration_matching[
            calibration_matching["selected_on_calibration"]
        ].itertuples(index=False)
    }
    test_matching["calibration_selected_condition"] = test_matching["target"].map(
        selected_lookup
    )
    test_matching["is_calibration_selected_condition"] = (
        test_matching["condition"]
        == test_matching["calibration_selected_condition"]
    )
    atomic_to_csv(test_matching, output_dir / "quality_matching_test_validation.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU runtime is required for the diagnostic")
    all_counterfactual_predictions = []
    for model_id in MODEL_IDS:
        all_counterfactual_predictions.append(
            infer_counterfactual_conditions(
                model_id,
                checkpoint_files[model_id],
                real_test,
                paths["data_root"],
                conditions,
                output_dir,
                args.batch_size,
                args.workers,
                device,
                not invalidate_cached_artifacts,
            )
        )
    counterfactual_predictions = pd.concat(
        all_counterfactual_predictions, ignore_index=True
    )
    baseline_reproduction = baseline_reproduction_table(
        counterfactual_predictions,
        existing_scores,
        real_test,
        threshold,
    )
    atomic_to_csv(
        baseline_reproduction,
        output_dir / "baseline_reproduction_check.csv",
    )
    paired, counterfactual_summary = summarize_counterfactuals(
        counterfactual_predictions,
        conditions,
        threshold,
        args.bootstrap_replicates,
        args.seed,
    )
    atomic_to_csv(paired, output_dir / "counterfactual_paired_predictions.csv")
    atomic_to_csv(counterfactual_summary, output_dir / "counterfactual_summary.csv")

    dose = dose_response_summary(
        paired, diagnostic["dose_response_families"]
    )
    atomic_to_csv(dose, output_dir / "dose_response_summary.csv")

    correlations = quality_score_correlations(test_quality, existing_scores)
    real_correlations = correlations[correlations["method"] == "original"].copy()
    atomic_to_csv(correlations, output_dir / "method_quality_score_correlations.csv")
    atomic_to_csv(real_correlations, output_dir / "real_quality_score_correlations.csv")

    save_plots(
        counterfactual_summary,
        real_correlations,
        calibration_matching,
        output_dir,
    )
    report = build_report(
        counterfactual_summary,
        dose,
        real_correlations,
        calibration_matching,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        output_dir / "diagnostic_summary.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": diagnostic["protocol"],
            "models": list(MODEL_IDS),
            "real_test_images": len(real_test),
            "threshold": threshold,
            "threshold_source": "existing fixed argmax/0.5 evaluation rule",
            "calibration": {
                "source": "M3 train split only",
                "samples_per_method": diagnostic["calibration"][
                    "samples_per_method"
                ],
                "model_scores_used": False,
                "test_images_used": False,
                "selected_conditions": selected_lookup,
                "robust_scales": scales,
            },
            "bootstrap": {
                "unit": "group_id cluster",
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
            },
            "outputs": {
                "report": "REPORT.md",
                "counterfactual_summary": "counterfactual_summary.csv",
                "baseline_reproduction": "baseline_reproduction_check.csv",
                "dose_response": "dose_response_summary.csv",
                "method_correlations": "method_quality_score_correlations.csv",
                "real_correlations": "real_quality_score_correlations.csv",
                "quality_matching_calibration": "quality_matching_calibration.csv",
                "quality_matching_test_validation": "quality_matching_test_validation.csv",
                "plots": "plots/",
            },
        },
    )
    print(f"Diagnostic complete: {output_dir}")
    print(f"Read first: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
