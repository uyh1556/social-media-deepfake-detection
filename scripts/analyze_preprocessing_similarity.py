import argparse
import hashlib
import json
import math
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL
import timm
import torchvision
from PIL import Image, ImageDraw, ImageOps

from analyze_instagram_pairs import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure paired raw/Instagram differences before and after the "
            "unchanged Standard Xception geometry preprocessing."
        )
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("instagram_pipeline/pilot_v1"),
    )
    parser.add_argument(
        "--proxy-dir",
        type=Path,
        default=Path("instagram_pipeline/proxy_web_q94_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/preprocessing_diagnosis_v1"),
    )
    parser.add_argument("--visual-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_standard_xception_geometry():
    model = timm.create_model("xception", pretrained=False, num_classes=2)
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    evaluation_transform = timm.data.create_transform(
        **data_config, is_training=False
    )
    transforms = evaluation_transform.transforms
    names = [type(transform).__name__ for transform in transforms]
    if names[:2] != ["Resize", "CenterCrop"]:
        raise RuntimeError(f"Unexpected Xception geometry transforms: {names}")
    if tuple(data_config["input_size"]) != (3, 299, 299):
        raise RuntimeError(f"Unexpected input size: {data_config['input_size']}")

    def geometry(image):
        result = image
        for transform in transforms[:2]:
            result = transform(result)
        return result

    return geometry, data_config, repr(evaluation_transform)


def load_actual_pilot_pairs(repo_root, pilot_dir):
    source_manifest_path = pilot_dir / "pilot_manifest.csv"
    target_manifest_path = pilot_dir / "download_manifest.csv"
    source = pd.read_csv(source_manifest_path)
    target = pd.read_csv(target_manifest_path)
    frame = source.merge(target, on="sample_id", validate="one_to_one")
    pairs = []
    for row in frame.sort_values("upload_order").itertuples():
        pairs.append(
            {
                "dataset": "real_instagram_pilot",
                "sample_id": row.sample_id,
                "split": row.split,
                "label": row.label,
                "method": row.method,
                "group_id": row.group_id,
                "video_id": row.video_id,
                "source_resolution": (
                    f"{row.source_width}x{row.source_height}"
                ),
                "source_path": repo_root / row.source_path,
                "target_path": pilot_dir / row.download_path,
                "source_reference_path": row.source_path,
            }
        )
    return pairs, [source_manifest_path, target_manifest_path]


def load_proxy_pairs(repo_root, proxy_dir):
    manifest_path = proxy_dir / "manifest.csv"
    frame = pd.read_csv(
        manifest_path,
        dtype={"group_id": str, "video_id": str, "source_ids": str},
    )
    pairs = []
    for row in frame.itertuples():
        pairs.append(
            {
                "dataset": "instagram_proxy_test",
                "sample_id": f"{row.method}:{row.source_reference_path}",
                "split": row.split,
                "label": row.label,
                "method": row.method,
                "group_id": row.group_id,
                "video_id": row.video_id,
                "source_resolution": f"{row.width}x{row.height}",
                "source_path": repo_root / row.source_reference_path,
                "target_path": proxy_dir / row.source_path,
                "source_reference_path": row.source_reference_path,
            }
        )
    return pairs, [manifest_path, proxy_dir / "config.json"]


def rgb_float(image):
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def metrics(source, target):
    if source.shape != target.shape:
        return {
            "status": "dimension_mismatch",
            "psnr_db": None,
            "ssim_rgb": None,
            "rmse_rgb": None,
            "mae_rgb": None,
        }
    psnr, rmse, mae = calculate_psnr(source, target)
    return {
        "status": "direct_same_dimensions",
        "psnr_db": psnr if math.isfinite(psnr) else None,
        "ssim_rgb": calculate_ssim(source, target),
        "rmse_rgb": rmse,
        "mae_rgb": mae,
    }


def analyze_pairs(pairs, geometry):
    rows = []
    prepared_images = {}
    for number, pair in enumerate(pairs, start=1):
        for path in [pair["source_path"], pair["target_path"]]:
            if not path.is_file():
                raise FileNotFoundError(path)
        with Image.open(pair["source_path"]) as image:
            source = image.convert("RGB")
        with Image.open(pair["target_path"]) as image:
            target = image.convert("RGB")
        source_input = geometry(source)
        target_input = geometry(target)
        before = metrics(rgb_float(source), rgb_float(target))
        after = metrics(rgb_float(source_input), rgb_float(target_input))
        row = {
            key: value
            for key, value in pair.items()
            if key not in {"source_path", "target_path"}
        }
        row.update(
            {
                "source_path": str(pair["source_path"]),
                "target_path": str(pair["target_path"]),
                "source_width": source.width,
                "source_height": source.height,
                "target_width": target.width,
                "target_height": target.height,
                "source_resolution": f"{source.width}x{source.height}",
                "source_bytes": pair["source_path"].stat().st_size,
                "target_bytes": pair["target_path"].stat().st_size,
                "before_status": before["status"],
                "before_psnr_db": before["psnr_db"],
                "before_ssim_rgb": before["ssim_rgb"],
                "before_rmse_rgb": before["rmse_rgb"],
                "before_mae_rgb": before["mae_rgb"],
                "after_width": source_input.width,
                "after_height": source_input.height,
                "after_status": after["status"],
                "after_psnr_db": after["psnr_db"],
                "after_ssim_rgb": after["ssim_rgb"],
                "after_rmse_rgb": after["rmse_rgb"],
                "after_mae_rgb": after["mae_rgb"],
                "psnr_change_db": (
                    after["psnr_db"] - before["psnr_db"]
                    if after["psnr_db"] is not None
                    and before["psnr_db"] is not None
                    else None
                ),
                "ssim_change": (
                    after["ssim_rgb"] - before["ssim_rgb"]
                    if after["ssim_rgb"] is not None
                    and before["ssim_rgb"] is not None
                    else None
                ),
                "mae_change": (
                    after["mae_rgb"] - before["mae_rgb"]
                    if after["mae_rgb"] is not None
                    and before["mae_rgb"] is not None
                    else None
                ),
            }
        )
        rows.append(row)
        if pair["dataset"].startswith("real_instagram"):
            prepared_images[pair["sample_id"]] = (
                source,
                target,
                source_input,
                target_input,
            )
        if number % 250 == 0 or number == len(pairs):
            print(f"{pair['dataset']}: {number}/{len(pairs)}")
    return pd.DataFrame(rows), prepared_images


def distribution(values):
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return {
        "n": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def summarize(frame):
    result = {"pairs": int(len(frame))}
    for column in [
        "before_psnr_db",
        "after_psnr_db",
        "psnr_change_db",
        "before_ssim_rgb",
        "after_ssim_rgb",
        "ssim_change",
        "before_mae_rgb",
        "after_mae_rgb",
        "mae_change",
    ]:
        result[column] = distribution(frame[column])
    result["psnr_increased_fraction"] = float(
        (frame["psnr_change_db"] > 0).mean()
    )
    result["ssim_increased_fraction"] = float(
        (frame["ssim_change"] > 0).mean()
    )
    result["mae_decreased_fraction"] = float(
        (frame["mae_change"] < 0).mean()
    )
    return result


def display_panel(image, size=299):
    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
    )


def save_visual(path, images):
    source, target, source_input, target_input = images
    source_array = np.asarray(source_input, dtype=np.int16)
    target_array = np.asarray(target_input, dtype=np.int16)
    difference = np.clip(
        np.abs(source_array - target_array) * 8, 0, 255
    ).astype(np.uint8)
    panels = [
        ("Raw", display_panel(source)),
        ("Instagram", display_panel(target)),
        ("Raw Xception input", source_input),
        ("Instagram Xception input", target_input),
        ("Absolute difference x8", Image.fromarray(difference)),
    ]
    label_height = 28
    canvas = Image.new(
        "RGB", (299 * len(panels), 299 + label_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * 299
        canvas.paste(panel, (x, label_height))
        draw.text((x + 6, 7), label, fill="black")
    canvas.save(path)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    pilot_dir = args.pilot_dir.resolve()
    proxy_dir = args.proxy_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_files = [
        output_dir / "real_instagram_pilot_similarity.csv",
        output_dir / "instagram_proxy_test_similarity.csv",
        output_dir / "similarity_summary.json",
    ]
    if not args.overwrite and any(path.exists() for path in output_files):
        raise FileExistsError("Similarity outputs exist; use --overwrite.")

    geometry, data_config, transform_repr = (
        resolve_standard_xception_geometry()
    )
    actual_pairs, actual_inputs = load_actual_pilot_pairs(
        repo_root, pilot_dir
    )
    proxy_pairs, proxy_inputs = load_proxy_pairs(repo_root, proxy_dir)
    actual, prepared = analyze_pairs(actual_pairs, geometry)
    proxy, _ = analyze_pairs(proxy_pairs, geometry)

    output_dir.mkdir(parents=True, exist_ok=True)
    actual.to_csv(output_files[0], index=False)
    proxy.to_csv(output_files[1], index=False)
    all_inputs = actual_inputs + proxy_inputs
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "Standard Xception preprocessing masking diagnosis",
        "standard_xception_unchanged": True,
        "data_config": data_config,
        "transform": transform_repr,
        "comparison_representation": (
            "RGB after exact Resize+CenterCrop geometry and before tensor "
            "normalization; normalization is an identical linear mapping"
        ),
        "input_files": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in all_inputs
        },
        "datasets": {
            "real_instagram_pilot": summarize(actual),
            "instagram_proxy_test": summarize(proxy),
        },
        "proxy_by_method": {
            method: summarize(group)
            for method, group in proxy.groupby("method", sort=True)
        },
        "proxy_by_source_resolution": {
            resolution: summarize(group)
            for resolution, group in proxy.groupby(
                "source_resolution", sort=True
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
            "timm": timm.__version__,
            "torchvision": torchvision.__version__,
        },
    }
    output_files[2].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(exist_ok=True)
    sample_ids = sorted(prepared)
    selected = random.Random(args.seed).sample(
        sample_ids, min(args.visual_count, len(sample_ids))
    )
    for sample_id in selected:
        save_visual(visual_dir / f"{sample_id}.png", prepared[sample_id])

    print(f"Real Instagram pairs: {len(actual)}")
    print(f"Proxy test pairs: {len(proxy)}")
    print(f"Summary: {output_files[2]}")
    print(f"Visuals: {visual_dir}")


if __name__ == "__main__":
    main()
