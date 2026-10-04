"""Frozen settings shared by augmented adaptive training and evaluation."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/family_rotation_adaptive_augmentation_v1/protocol.json"
EXPERIMENT = "family_rotation_adaptive_augmentation_v1"
AUGMENTATIONS = ("mixed_jpeg", "mixed_jpeg_noise")
STRATEGIES = ("uniform", "complementarity")


def load_experiment(path=CONFIG):
    settings = json.loads(Path(path).read_text())
    if (settings.get("experiment") != EXPERIMENT or settings.get("training_seed") != 42
            or settings.get("train_jpeg_qualities") != [75, 80, 85, 90, 95]
            or settings.get("noise_probability") != 0.25 or settings.get("noise_max_std") != 5.0
            or settings.get("validation_jpeg_quality") != 95 or settings.get("validation_noise_std") != 0
            or settings.get("unique_train_images") != 39600):
        raise ValueError("Expected frozen seed42 Mixed-JPEG / noise25%-std5 experiment")
    names = set()
    for condition in settings["evaluation_conditions"]:
        if (condition["name"] in names or not 1 <= condition["jpeg_quality"] <= 100
                or condition["noise_std"] not in (0, 2, 5, 10)):
            raise ValueError("Invalid or duplicate evaluation condition")
        names.add(condition["name"])
    return settings


def adaptive_config_path(settings):
    return ROOT / settings["adaptive_protocol_config"]


def run_directory(root, selection, strategy, augmentation, seed=42):
    if selection not in {f"S{i}" for i in range(1, 7)} or strategy not in STRATEGIES or augmentation not in AUGMENTATIONS:
        raise ValueError("Unknown selection/strategy/augmentation")
    return Path(root) / f"xception_{selection.lower()}_m7_{augmentation}_{strategy}_adaptive_aug_v1_seed{seed}"


def check_training_arguments(args, settings):
    """Explicit opt-in; the original fixed-Q95 adaptive entry remains restricted."""
    if (args.seed != settings["training_seed"] or args.train_jpeg_qualities != settings["train_jpeg_qualities"]
            or args.canonical_size != 256 or args.jpeg_quality != 95 or args.jpeg_subsampling != 2
            or args.train_noise_probability not in (0.0, settings["noise_probability"])
            or args.train_noise_max_std != settings["noise_max_std"]):
        raise ValueError("Augmented adaptive CLI settings differ from the frozen experiment")
    return "mixed_jpeg_noise" if args.train_noise_probability else "mixed_jpeg"


def validate_saved(saved, manifest_hash, roles_hash, protocol, strategy, augmentation, experiment_hash):
    adaptive = saved.get("adaptive_learning", {})
    prep = saved.get("preprocessing", {})
    noise = prep.get("train_noise", {})
    expected_probability = 0.25 if augmentation == "mixed_jpeg_noise" else 0.0
    if (saved.get("seed") != 42 or saved.get("train_images") != 39600
            or saved.get("manifest_sha256") != manifest_hash
            or saved.get("experiment_family") != EXPERIMENT
            or adaptive.get("strategy") != strategy or adaptive.get("protocol") != protocol
            or adaptive.get("validation_roles_sha256") != roles_hash
            or adaptive.get("augmentation_config_sha256") != experiment_hash
            or adaptive.get("augmentation") != augmentation):
        raise RuntimeError("Augmented checkpoint/manifest/roles/protocol identity differs")
    if (saved.get("preprocessing_name") != "canonical256_jpegmixed_letterbox299"
            or prep.get("canonical_size") != 256 or prep.get("train_jpeg_qualities") != [75, 80, 85, 90, 95]
            or prep.get("train_jpeg_sampling") != "uniform" or prep.get("validation_jpeg_quality") != 95
            or prep.get("jpeg_subsampling") != 2 or prep.get("jpeg_optimize") is not False
            or prep.get("jpeg_progressive") is not False or noise.get("probability") != expected_probability
            or noise.get("max_std_0_to_255") != 5.0 or noise.get("validation_noise") is not False):
        raise RuntimeError("Augmented checkpoint preprocessing differs")


def validate_checkpoint_config(checkpoint_config, saved):
    """best.pt may predate a resume; compare scientific settings, not log/path flags."""
    from train_baseline import json_ready
    fields = ("manifest_sha256", "seed", "train_images", "val_images", "preprocessing_name",
        "preprocessing", "data_config", "adaptive_learning", "epochs", "batch_size", "workers",
        "learning_rate", "weight_decay", "patience", "min_delta", "split_protocol",
        "experiment_family", "condition_name")
    for field in fields:
        if json_ready(checkpoint_config.get(field)) != json_ready(saved.get(field)):
            raise RuntimeError(f"config.json and best.pt scientific setting differs: {field}")
