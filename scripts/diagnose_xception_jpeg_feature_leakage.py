#!/usr/bin/env python3
"""Measure how much JPEG-quality information remains in frozen Xception features.

This diagnostic uses development validation images only.  Each source image is
rendered at every configured JPEG quality, and all variants from the same
source group stay together when fitting/evaluating the linear quality probe.
No detector checkpoint is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import sklearn
import timm
import torch
import torchvision
from PIL import Image, features
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from train_baseline import (
    LABEL_MAP,
    json_ready,
    seed_worker,
    set_seed,
    write_json,
)
from xception_preprocessing import (
    JpegRoundTrip,
    Letterbox,
    interpolation_mode,
)


PROTOCOL_NAME = "jpeg_feature_leakage_probe_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Checkpoint must use NAME=/absolute/path/to/best.pt"
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not name.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint name: {name!r}")
    return name, Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Repeat NAME=/path/best.pt for every frozen model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=(
            project_root
            / "configs/jpeg_feature_leakage_probe_v1/protocol.json"
        ),
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        type=int,
        default=[75, 80, 85, 90, 95],
    )
    parser.add_argument("--canonical-size", type=int, default=256)
    parser.add_argument("--jpeg-subsampling", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--probe-test-size", type=float, default=0.3)
    parser.add_argument(
        "--probe-seeds", nargs="+", type=int, default=[42, 43, 44]
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_probe_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL_NAME:
        raise ValueError("Unexpected JPEG feature-leakage protocol.")
    if config.get("data_scope") != "M7 validation split only":
        raise ValueError("Probe must remain restricted to M7 validation.")
    if config.get("protected_unseen_used") is not False:
        raise ValueError("Protected unseen data must not enter this probe.")
    if config.get("wilddeepfake_used") is not False:
        raise ValueError("WildDeepfake must not enter this probe.")
    return config


class QualityVariantDataset(Dataset):
    """Return every fixed JPEG quality for each validation source image."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        data_config: dict,
        *,
        qualities: list[int],
        canonical_size: int,
        jpeg_subsampling: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.qualities = tuple(int(value) for value in qualities)
        interpolation = interpolation_mode(data_config["interpolation"])
        self.canonical = Letterbox(
            size=canonical_size,
            interpolation=interpolation,
        )
        self.model_letterbox = Letterbox(
            size=int(data_config["input_size"][1]),
            interpolation=interpolation,
        )
        self.jpeg_subsampling = int(jpeg_subsampling)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=data_config["mean"],
            std=data_config["std"],
        )

    def __len__(self) -> int:
        return len(self.frame) * len(self.qualities)

    def __getitem__(self, index: int) -> dict:
        source_index, quality_index = divmod(index, len(self.qualities))
        row = self.frame.iloc[source_index]
        quality = self.qualities[quality_index]
        image_path = self.data_root / row["source_path"]
        with Image.open(image_path) as image:
            image = self.canonical(image.convert("RGB"))
            image = JpegRoundTrip(
                quality=quality,
                subsampling=self.jpeg_subsampling,
                optimize=False,
                progressive=False,
            )(image)
            image = self.model_letterbox(image)
            tensor = self.normalize(self.to_tensor(image))
        return {
            "image": tensor,
            "source_index": source_index,
            "quality": quality,
            "quality_index": quality_index,
            "label_index": LABEL_MAP[str(row["label"])],
        }


class FeatureCapture:
    """Capture Xception's global-pooled feature without changing the model."""

    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "global_pool"):
            raise AttributeError("Xception model has no global_pool module.")
        self.values: list[torch.Tensor] = []
        self.handle = model.global_pool.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        self.values.append(output.flatten(1))

    def pop(self) -> torch.Tensor:
        if len(self.values) != 1:
            raise RuntimeError(
                f"Expected one captured feature tensor, got {len(self.values)}"
            )
        return self.values.pop()

    def close(self) -> None:
        self.handle.remove()


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model, checkpoint


def validate_checkpoint(
    name: str,
    checkpoint: dict,
    manifest_sha256: str,
) -> None:
    expected_families = {
        "fixed_q95": "family_coverage_reencoding_control_v1",
        "mixed_jpeg": "family_coverage_jpeg_mixed_v1",
    }
    if name not in expected_families:
        raise ValueError(
            "Frozen probe checkpoint names must be fixed_q95 and mixed_jpeg."
        )
    config = checkpoint.get("config", {})
    if config.get("experiment_family") != expected_families[name]:
        raise ValueError(
            f"Unexpected experiment family for {name}: "
            f"{config.get('experiment_family')}"
        )
    if not str(config.get("condition_name", "")).startswith("m7_"):
        raise ValueError(f"Checkpoint is not an M7 run: {name}")
    if int(config.get("seed", -1)) != 42:
        raise ValueError(f"Checkpoint is not seed 42: {name}")
    if config.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Checkpoint manifest mismatch: {name}")


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    capture = FeatureCapture(model)
    all_features: list[np.ndarray] = []
    all_fake_scores: list[np.ndarray] = []
    all_source_indices: list[np.ndarray] = []
    all_quality_indices: list[np.ndarray] = []
    all_label_indices: list[np.ndarray] = []
    try:
        for batch in tqdm(loader, desc=description):
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
            pooled = capture.pop()
            fake_scores = torch.softmax(logits.float(), dim=1)[:, 1]
            all_features.append(pooled.float().cpu().numpy())
            all_fake_scores.append(fake_scores.cpu().numpy())
            all_source_indices.append(batch["source_index"].numpy())
            all_quality_indices.append(batch["quality_index"].numpy())
            all_label_indices.append(batch["label_index"].numpy())
    finally:
        capture.close()
    return (
        np.concatenate(all_features).astype(np.float32, copy=False),
        np.concatenate(all_fake_scores).astype(np.float32, copy=False),
        np.concatenate(all_source_indices),
        np.concatenate(all_quality_indices),
        np.concatenate(all_label_indices),
    )


def probe_rows(
    *,
    model_name: str,
    features: np.ndarray,
    quality_indices: np.ndarray,
    label_indices: np.ndarray,
    group_ids: np.ndarray,
    probe_seeds: list[int],
    test_size: float,
) -> list[dict]:
    rows: list[dict] = []
    subsets = {
        "all": np.ones(len(label_indices), dtype=bool),
        "real": label_indices == LABEL_MAP["real"],
        "fake": label_indices == LABEL_MAP["fake"],
    }
    for subset_name, subset_mask in subsets.items():
        subset_features = normalize(features[subset_mask], norm="l2")
        subset_targets = quality_indices[subset_mask]
        subset_groups = group_ids[subset_mask]
        for probe_seed in probe_seeds:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=probe_seed,
            )
            train_index, test_index = next(
                splitter.split(
                    subset_features,
                    subset_targets,
                    groups=subset_groups,
                )
            )
            if set(subset_targets[train_index]) != set(
                range(len(np.unique(quality_indices)))
            ):
                raise RuntimeError("Probe train split lost a JPEG-quality class.")
            classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=200,
                tol=1e-4,
                average=True,
                random_state=probe_seed,
            )
            classifier.fit(
                subset_features[train_index],
                subset_targets[train_index],
            )
            predictions = classifier.predict(subset_features[test_index])
            rows.append(
                {
                    "model": model_name,
                    "subset": subset_name,
                    "probe_seed": probe_seed,
                    "train_variants": len(train_index),
                    "test_variants": len(test_index),
                    "train_groups": len(np.unique(subset_groups[train_index])),
                    "test_groups": len(np.unique(subset_groups[test_index])),
                    "accuracy": accuracy_score(
                        subset_targets[test_index], predictions
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        subset_targets[test_index], predictions
                    ),
                    "macro_f1": f1_score(
                        subset_targets[test_index],
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "confusion_matrix": confusion_matrix(
                        subset_targets[test_index],
                        predictions,
                    ).tolist(),
                }
            )
    return rows


def write_report(
    path: Path,
    *,
    qualities: list[int],
    source_images: int,
    probe_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display_frame = frame.copy()
        for column in display_frame.select_dtypes(include=["float"]).columns:
            display_frame[column] = display_frame[column].map(
                lambda value: f"{value:.6f}"
            )
        headers = [str(value) for value in display_frame.columns]
        rows = [
            [str(value) for value in row]
            for row in display_frame.itertuples(index=False, name=None)
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    lines = [
        "# JPEG feature-leakage probe",
        "",
        "This diagnostic freezes every detector and uses only the M7 validation "
        "images. It does not use protected unseen methods or WildDeepfake.",
        "",
        f"- Source images: {source_images:,}",
        f"- JPEG qualities: {qualities}",
        f"- Chance quality-classification accuracy: {1 / len(qualities):.3f}",
        "- Probe split: source `group_id` disjoint",
        "",
        "## Linear quality probe",
        "",
        "Higher accuracy means that JPEG quality remains more linearly readable "
        "from the frozen detector representation. It does not by itself prove "
        "that the Real/Fake decision uses that information.",
        "",
        markdown_table(probe_summary),
        "",
        "## Detector score by quality",
        "",
        markdown_table(score_summary),
        "",
        "## Decision rule",
        "",
        "- If Mixed-JPEG substantially lowers probe accuracy and stabilizes "
        "Real/Fake scores, an additional quality-adversarial head has weak "
        "empirical justification.",
        "- If Mixed-JPEG retains high quality-probe accuracy and the quality "
        "signal tracks score/FPR shifts, a frozen development-only adversarial "
        "pilot may be justified after the literature-overlap review.",
        "- This report must not be used to select on protected unseen or Wild "
        "performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.probe_config = args.probe_config.resolve()
    args.checkpoint = [
        (name, path.resolve()) for name, path in args.checkpoint
    ]
    if len({name for name, _ in args.checkpoint}) != len(args.checkpoint):
        raise ValueError("Checkpoint names must be unique.")
    if len(set(args.qualities)) != len(args.qualities):
        raise ValueError("JPEG qualities must be unique.")
    if any(not 1 <= quality <= 100 for quality in args.qualities):
        raise ValueError("JPEG qualities must be between 1 and 100.")
    if not 0 < args.probe_test_size < 1:
        raise ValueError("--probe-test-size must be between 0 and 1.")
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    frozen_probe = load_probe_config(args.probe_config)
    expected_preprocessing = frozen_probe["preprocessing"]
    expected_probe = frozen_probe["linear_probe"]
    expected_manifest_hash = frozen_probe["manifest_sha256"]
    actual_manifest_hash = sha256_file(args.manifest)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "M7 manifest hash mismatch: "
            f"expected={expected_manifest_hash}, actual={actual_manifest_hash}"
        )
    expected_arguments = {
        "qualities": expected_preprocessing["jpeg_qualities"],
        "canonical_size": expected_preprocessing["canonical_size"],
        "jpeg_subsampling": expected_preprocessing["jpeg_subsampling"],
        "probe_test_size": expected_probe["test_size"],
        "probe_seeds": expected_probe["seeds"],
    }
    for key, expected_value in expected_arguments.items():
        actual_value = getattr(args, key)
        if actual_value != expected_value:
            raise ValueError(
                f"Frozen probe mismatch for {key}: "
                f"expected={expected_value}, actual={actual_value}"
            )
    for _, checkpoint_path in args.checkpoint:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for feature extraction.")

    manifest = pd.read_csv(args.manifest)
    required = {
        "sample_id",
        "split",
        "label",
        "method",
        "group_id",
        "source_path",
    }
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {sorted(missing_columns)}")
    frame = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Manifest has no validation rows.")
    missing_files = [
        value
        for value in frame["source_path"]
        if not (args.data_root / value).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} validation images; first: "
            f"{missing_files[0]}"
        )

    protocol = {
        "protocol": PROTOCOL_NAME,
        "purpose": (
            "Frozen-feature diagnostic of model-visible JPEG-quality leakage"
        ),
        "data_scope": "M7 development validation split only",
        "protected_unseen_used": False,
        "wilddeepfake_used": False,
        "source_images": len(frame),
        "qualities": args.qualities,
        "canonical_size": args.canonical_size,
        "jpeg_subsampling": args.jpeg_subsampling,
        "model_input_size": 299,
        "probe": {
            "type": "L2-normalized multinomial linear logistic regression",
            "group_column": "group_id",
            "test_size": args.probe_test_size,
            "seeds": args.probe_seeds,
            "subsets": ["all", "real", "fake"],
            "chance_accuracy": 1.0 / len(args.qualities),
        },
        "manifest": str(args.manifest),
        "manifest_sha256": actual_manifest_hash,
        "frozen_config": str(args.probe_config),
        "frozen_config_sha256": sha256_file(args.probe_config),
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in args.checkpoint
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "pillow": PIL.__version__,
            "libjpeg": features.version_codec("jpg"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)

    probe_results: list[dict] = []
    score_results: list[pd.DataFrame] = []
    base_group_ids = frame["group_id"].fillna(frame["sample_id"]).to_numpy()
    for model_name, checkpoint_path in args.checkpoint:
        model, checkpoint = load_model(checkpoint_path, device)
        validate_checkpoint(model_name, checkpoint, actual_manifest_hash)
        data_config = checkpoint["config"]["data_config"]
        dataset = QualityVariantDataset(
            frame,
            args.data_root,
            data_config,
            qualities=args.qualities,
            canonical_size=args.canonical_size,
            jpeg_subsampling=args.jpeg_subsampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker,
        )
        (
            model_features,
            fake_scores,
            source_indices,
            quality_indices,
            label_indices,
        ) = extract_features(
            model,
            loader,
            device,
            description=f"Features: {model_name}",
        )
        variant_group_ids = base_group_ids[source_indices]
        probe_results.extend(
            probe_rows(
                model_name=model_name,
                features=model_features,
                quality_indices=quality_indices,
                label_indices=label_indices,
                group_ids=variant_group_ids,
                probe_seeds=args.probe_seeds,
                test_size=args.probe_test_size,
            )
        )

        prediction_frame = pd.DataFrame(
            {
                "model": model_name,
                "sample_id": frame.iloc[source_indices]["sample_id"].to_numpy(),
                "group_id": variant_group_ids,
                "label": frame.iloc[source_indices]["label"].to_numpy(),
                "method": frame.iloc[source_indices]["method"].to_numpy(),
                "jpeg_quality": np.asarray(args.qualities)[quality_indices],
                "fake_probability": fake_scores,
            }
        )
        prediction_frame.to_csv(
            args.output_dir / f"{model_name}_quality_predictions.csv",
            index=False,
        )
        score_summary = (
            prediction_frame.groupby(["model", "label", "jpeg_quality"])
            .agg(
                count=("fake_probability", "size"),
                fake_probability_mean=("fake_probability", "mean"),
                fake_probability_median=("fake_probability", "median"),
                fake_probability_std=("fake_probability", "std"),
                predicted_fake_rate=(
                    "fake_probability",
                    lambda values: float((values >= 0.5).mean()),
                ),
            )
            .reset_index()
        )
        score_results.append(score_summary)
        del model, model_features
        torch.cuda.empty_cache()

    probe_frame = pd.DataFrame(probe_results)
    probe_frame.to_csv(args.output_dir / "probe_folds.csv", index=False)
    probe_summary = (
        probe_frame.groupby(["model", "subset"])[
            ["accuracy", "balanced_accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    probe_summary.columns = [
        "_".join(value).rstrip("_")
        if isinstance(value, tuple)
        else value
        for value in probe_summary.columns
    ]
    probe_summary.to_csv(
        args.output_dir / "probe_summary.csv", index=False
    )
    score_summary = pd.concat(score_results, ignore_index=True)
    score_summary.to_csv(
        args.output_dir / "quality_score_summary.csv", index=False
    )
    write_report(
        args.output_dir / "REPORT.md",
        qualities=args.qualities,
        source_images=len(frame),
        probe_summary=probe_summary,
        score_summary=score_summary,
    )
    summary = {
        "protocol": PROTOCOL_NAME,
        "probe_summary": probe_summary.to_dict(orient="records"),
        "quality_score_summary": score_summary.to_dict(orient="records"),
        "report": str(args.output_dir / "REPORT.md"),
    }
    write_json(args.output_dir / "diagnostic_summary.json", json_ready(summary))
    print("Saved:", args.output_dir)
    print(probe_summary.to_string(index=False))


if __name__ == "__main__":
    main()
