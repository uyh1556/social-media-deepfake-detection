import argparse
import hashlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import PIL
from PIL import Image, JpegImagePlugin


SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
JPEG_SAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "unknown"}
PILLOW_SAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze paired FF++ source and Instagram images."
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("instagram_pipeline/pilot_v1"),
    )
    parser.add_argument(
        "--source-manifest-name",
        default="pilot_manifest.csv",
        help="Source manifest filename inside --pilot-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing analysis CSV and summary JSON.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def calculate_psnr(source, target):
    difference = source.astype(np.float32) - target.astype(np.float32)
    mse = float(np.mean(difference * difference))
    if mse == 0:
        return float("inf"), 0.0, 0.0
    rmse = math.sqrt(mse)
    psnr = 20.0 * math.log10(255.0 / rmse)
    mae = float(np.mean(np.abs(difference)))
    return psnr, rmse, mae


def calculate_ssim(source, target):
    if source.shape != target.shape:
        raise ValueError("SSIM requires arrays with identical shapes.")
    if min(source.shape[:2]) < SSIM_WINDOW:
        raise ValueError("Image is smaller than the SSIM window.")

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    channel_scores = []
    border = SSIM_WINDOW // 2

    for channel in range(3):
        x = source[:, :, channel].astype(np.float32)
        y = target[:, :, channel].astype(np.float32)
        mu_x = cv2.GaussianBlur(x, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA)
        mu_y = cv2.GaussianBlur(y, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(
            x * x, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA
        ) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(
            y * y, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA
        ) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(
            x * y, (SSIM_WINDOW, SSIM_WINDOW), SSIM_SIGMA
        ) - mu_xy
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (
            sigma_x_sq + sigma_y_sq + c2
        )
        score_map = numerator / denominator
        valid = score_map[border:-border, border:-border]
        channel_scores.append(float(np.mean(valid)))

    return float(np.mean(channel_scores))


def pillow_quantization_candidates():
    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    candidates = {}
    for quality in range(1, 101):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            candidates[quality] = {
                key: np.asarray(value, dtype=np.float32)
                for key, value in encoded.quantization.items()
            }
    return candidates


def estimate_jpeg_equivalent_quality(path, candidates):
    with Image.open(path) as image:
        if image.format != "JPEG" or not image.quantization:
            return None, None, "not_jpeg", None, None
        target_tables = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in image.quantization.items()
        }
        sampling_code = JpegImagePlugin.get_sampling(image)
        progressive = bool(
            image.info.get("progressive") or image.info.get("progression")
        )

    scores = {}
    for quality, reference_tables in candidates.items():
        if set(reference_tables) != set(target_tables):
            continue
        differences = [
            np.abs(target_tables[key] - reference_tables[key])
            for key in sorted(target_tables)
        ]
        scores[quality] = float(np.mean(np.concatenate(differences)))
    if not scores:
            return (
                None,
                None,
                JPEG_SAMPLING.get(sampling_code, "unknown"),
                progressive,
                len(target_tables),
            )
    quality = min(scores, key=scores.get)
    return (
        quality,
        scores[quality],
        JPEG_SAMPLING.get(sampling_code, "unknown"),
        progressive,
        len(target_tables),
    )


def create_inferred_proxy_rgb(
    source_path, quality, sampling, progressive
):
    buffer = io.BytesIO()
    with Image.open(source_path) as image:
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=PILLOW_SAMPLING[sampling],
            progressive=progressive,
        )
    content = buffer.getvalue()
    with Image.open(io.BytesIO(content)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return rgb, len(content)


def finite_or_none(value):
    if value is None or not math.isfinite(value):
        return None
    return value


def summarize_group(frame):
    result = {"images": int(len(frame))}
    for column in [
        "byte_reduction_pct",
        "psnr_db",
        "ssim_rgb",
        "mae_rgb",
        "estimated_jpeg_equivalent_quality",
    ]:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(values):
            result[column] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return result


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    pilot_dir = args.pilot_dir.resolve()
    source_manifest_path = pilot_dir / args.source_manifest_name
    download_manifest_path = pilot_dir / "download_manifest.csv"
    analysis_dir = pilot_dir / "analysis"
    metrics_path = analysis_dir / "pair_metrics.csv"
    summary_path = analysis_dir / "summary.json"

    for path in [source_manifest_path, download_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.overwrite and (metrics_path.exists() or summary_path.exists()):
        raise FileExistsError(
            "Analysis output already exists. Use --overwrite to replace it."
        )

    source_manifest = pd.read_csv(source_manifest_path)
    download_manifest = pd.read_csv(download_manifest_path)
    required_source = {
        "sample_id", "upload_order", "split", "label", "method",
        "source_path", "source_bytes", "source_sha256",
    }
    required_target = {
        "sample_id", "download_path", "target_format", "target_width",
        "target_height", "target_bytes", "target_sha256",
    }
    if missing := required_source - set(source_manifest.columns):
        raise ValueError(f"Missing source-manifest columns: {sorted(missing)}")
    if missing := required_target - set(download_manifest.columns):
        raise ValueError(f"Missing download-manifest columns: {sorted(missing)}")
    if source_manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate source sample IDs.")
    if download_manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate target sample IDs.")
    if set(source_manifest.sample_id) != set(download_manifest.sample_id):
        raise ValueError("Source and target sample-ID sets differ.")

    paired = source_manifest.merge(
        download_manifest, on="sample_id", how="inner", validate="one_to_one"
    ).sort_values("upload_order")
    quantization_candidates = pillow_quantization_candidates()
    rows = []

    for _, item in paired.iterrows():
        source_path = repo_root / item.source_path
        target_path = pilot_dir / item.download_path
        for path in [source_path, target_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        if source_path.stat().st_size != int(item.source_bytes):
            raise ValueError(f"Source byte count changed: {item.sample_id}")
        if target_path.stat().st_size != int(item.target_bytes):
            raise ValueError(f"Target byte count changed: {item.sample_id}")
        if sha256_file(source_path) != item.source_sha256:
            raise ValueError(f"Source hash changed: {item.sample_id}")
        if sha256_file(target_path) != item.target_sha256:
            raise ValueError(f"Target hash changed: {item.sample_id}")

        with Image.open(source_path) as image:
            source_format = image.format
            source_width, source_height = image.size
        with Image.open(target_path) as image:
            target_format = image.format
            target_width, target_height = image.size
        if target_format != item.target_format:
            raise ValueError(f"Target format changed: {item.sample_id}")
        if (target_width, target_height) != (
            int(item.target_width), int(item.target_height)
        ):
            raise ValueError(f"Target dimensions changed: {item.sample_id}")

        dimensions_match = (source_width, source_height) == (
            target_width, target_height
        )
        psnr = rmse = mae = ssim = None
        if dimensions_match:
            source_rgb = load_rgb(source_path)
            target_rgb = load_rgb(target_path)
            psnr, rmse, mae = calculate_psnr(source_rgb, target_rgb)
            ssim = calculate_ssim(source_rgb, target_rgb)

        quality, quality_distance, sampling, progressive, table_count = (
            estimate_jpeg_equivalent_quality(
                target_path, quantization_candidates
            )
        )
        proxy_target_exact = None
        proxy_target_psnr = None
        proxy_target_ssim = None
        proxy_target_mae = None
        inferred_proxy_bytes = None
        if (
            dimensions_match
            and quality is not None
            and sampling in PILLOW_SAMPLING
            and progressive is not None
        ):
            proxy_rgb, inferred_proxy_bytes = create_inferred_proxy_rgb(
                source_path, quality, sampling, progressive
            )
            proxy_target_exact = bool(np.array_equal(proxy_rgb, target_rgb))
            (
                proxy_target_psnr,
                _,
                proxy_target_mae,
            ) = calculate_psnr(proxy_rgb, target_rgb)
            proxy_target_ssim = calculate_ssim(proxy_rgb, target_rgb)
        rows.append(
            {
                "sample_id": item.sample_id,
                "upload_order": int(item.upload_order),
                "split": item.split,
                "label": item.label,
                "method": item.method,
                "source_path": item.source_path,
                "target_path": item.download_path,
                "source_format": source_format,
                "target_format": target_format,
                "source_width": source_width,
                "source_height": source_height,
                "source_resolution": f"{source_width}x{source_height}",
                "target_width": target_width,
                "target_height": target_height,
                "width_ratio": target_width / source_width,
                "height_ratio": target_height / source_height,
                "aspect_ratio_change": (
                    target_width / target_height
                    - source_width / source_height
                ),
                "dimensions_match": dimensions_match,
                "source_bytes": int(item.source_bytes),
                "target_bytes": int(item.target_bytes),
                "byte_reduction_pct": (
                    100.0 * (1.0 - int(item.target_bytes) / int(item.source_bytes))
                ),
                "source_sha256": item.source_sha256,
                "target_sha256": item.target_sha256,
                "pixel_metrics_status": (
                    "direct_same_dimensions"
                    if dimensions_match
                    else "not_computed_dimension_mismatch"
                ),
                "psnr_db": finite_or_none(psnr),
                "rmse_rgb": rmse,
                "mae_rgb": mae,
                "ssim_rgb": ssim,
                "estimated_jpeg_equivalent_quality": quality,
                "jpeg_quantization_mae": quality_distance,
                "jpeg_sampling": sampling,
                "jpeg_progressive": progressive,
                "jpeg_quantization_table_count": table_count,
                "inferred_proxy_bytes": inferred_proxy_bytes,
                "proxy_target_decoded_pixels_exact": proxy_target_exact,
                "proxy_target_psnr_db": finite_or_none(proxy_target_psnr),
                "proxy_target_mae_rgb": proxy_target_mae,
                "proxy_target_ssim_rgb": proxy_target_ssim,
            }
        )
        print(
            f"Analyzed {item.sample_id}: PSNR={psnr:.3f} dB, "
            f"SSIM={ssim:.5f}, estimated JPEG-equivalent quality={quality}"
        )

    metrics = pd.DataFrame(rows)
    proxy_valid = metrics[
        metrics["proxy_target_decoded_pixels_exact"].notna()
    ]
    summary = {
        "pilot": pilot_dir.name,
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": int(len(metrics)),
        "paired_unique_ids": int(metrics.sample_id.nunique()),
        "input_manifests": {
            "source_manifest": {
                "path": str(source_manifest_path.relative_to(repo_root)),
                "sha256": sha256_file(source_manifest_path),
            },
            "download_manifest": {
                "path": str(download_manifest_path.relative_to(repo_root)),
                "sha256": sha256_file(download_manifest_path),
            },
        },
        "all_dimensions_match": bool(metrics.dimensions_match.all()),
        "all_source_hashes_differ_from_target": bool(
            (metrics.source_sha256 != metrics.target_sha256).all()
        ),
        "overall": summarize_group(metrics),
        "by_method": {
            method: summarize_group(group)
            for method, group in metrics.groupby("method", sort=True)
        },
        "by_source_resolution": {
            resolution: summarize_group(group)
            for resolution, group in metrics.groupby(
                "source_resolution", sort=True
            )
        },
        "proxy_fidelity": {
            "images": int(len(proxy_valid)),
            "all_decoded_pixels_exact": bool(
                len(proxy_valid)
                and proxy_valid[
                    "proxy_target_decoded_pixels_exact"
                ].astype(bool).all()
            ),
            "mean_target_ssim": (
                float(proxy_valid["proxy_target_ssim_rgb"].mean())
                if len(proxy_valid)
                else None
            ),
            "mean_target_mae": (
                float(proxy_valid["proxy_target_mae_rgb"].mean())
                if len(proxy_valid)
                else None
            ),
        },
        "protocol": {
            "pixel_alignment": "none; direct comparison only for identical dimensions",
            "pixel_dtype": "float32 RGB decoded values",
            "psnr_data_range": 255.0,
            "ssim": (
                "mean of RGB-channel SSIM; 11x11 Gaussian window, sigma=1.5, "
                "K1=0.01, K2=0.03, valid interior"
            ),
            "jpeg_quality": (
                "estimated JPEG-equivalent quality from nearest Pillow/libjpeg "
                "standard quantization tables; not Instagram's encoder setting"
            ),
            "proxy_validation": (
                "source encoded with each target's estimated quality, sampling, "
                "and progressive mode, then decoded RGB compared with target"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "opencv": cv2.__version__,
        },
    }

    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Pair metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
