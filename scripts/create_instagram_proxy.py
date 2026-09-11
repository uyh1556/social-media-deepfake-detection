import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import PIL
from PIL import Image


SAMPLING_TO_PILLOW = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a held-out JPEG proxy using a unanimous Instagram pilot "
            "transformation. Source images are never modified."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--strip-source-prefix",
        default=None,
        help=(
            "Explicitly remove one leading path component when resolving "
            "source files, e.g. source_images_raw for a flattened Colab tar."
        ),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_path(data_root, relative_path, strip_prefix):
    relative = Path(relative_path)
    if strip_prefix is not None:
        if not relative.parts or relative.parts[0] != strip_prefix:
            raise ValueError(
                f"Path does not start with {strip_prefix!r}: {relative}"
            )
        relative = Path(*relative.parts[1:])
    return data_root / relative


def infer_proxy_settings(pilot_metrics_path):
    metrics = pd.read_csv(pilot_metrics_path)
    required = {
        "dimensions_match",
        "target_format",
        "estimated_jpeg_equivalent_quality",
        "jpeg_sampling",
        "jpeg_progressive",
        "proxy_target_decoded_pixels_exact",
    }
    if missing := required - set(metrics.columns):
        raise ValueError(f"Pilot metrics missing columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("Pilot metrics are empty.")
    if not metrics["dimensions_match"].astype(bool).all():
        raise ValueError("Pilot includes a resize or crop; this proxy forbids it.")
    if not metrics["proxy_target_decoded_pixels_exact"].astype(bool).all():
        raise ValueError(
            "Inferred proxy does not exactly reproduce every decoded pilot target."
        )

    formats = set(metrics["target_format"].str.upper())
    qualities = set(metrics["estimated_jpeg_equivalent_quality"].dropna())
    samplings = set(metrics["jpeg_sampling"].dropna())
    progressive_values = set(metrics["jpeg_progressive"].astype(bool))
    if formats != {"JPEG"}:
        raise ValueError(f"Pilot target formats are not unanimous: {formats}")
    if len(qualities) != 1:
        raise ValueError(f"Pilot JPEG qualities are not unanimous: {qualities}")
    if len(samplings) != 1:
        raise ValueError(f"Pilot JPEG samplings are not unanimous: {samplings}")
    if len(progressive_values) != 1:
        raise ValueError(
            f"Pilot progressive modes are not unanimous: {progressive_values}"
        )

    quality = int(next(iter(qualities)))
    sampling = next(iter(samplings))
    if sampling not in SAMPLING_TO_PILLOW:
        raise ValueError(f"Unsupported JPEG sampling: {sampling}")
    return {
        "format": "JPEG",
        "quality": quality,
        "sampling": sampling,
        "pillow_subsampling": SAMPLING_TO_PILLOW[sampling],
        "progressive": bool(next(iter(progressive_values))),
        "resize": None,
        "pilot_validation_images": int(len(metrics)),
        "pilot_decoded_rgb_exact": True,
    }


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = args.manifest.resolve()
    pilot_metrics_path = args.pilot_metrics.resolve()
    output_dir = args.output_dir.resolve()
    building_dir = output_dir.with_name(output_dir.name + ".building")

    for path in [data_root, manifest_path, pilot_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if building_dir.exists():
        raise FileExistsError(
            f"Incomplete build directory already exists: {building_dir}"
        )

    settings = infer_proxy_settings(pilot_metrics_path)
    manifest = pd.read_csv(
        manifest_path,
        dtype={
            "group_id": str,
            "video_id": str,
            "source_ids": str,
            "source_path": str,
        },
    )
    required = {
        "split", "label", "method", "group_id", "video_id",
        "source_ids", "source_path",
    }
    if missing := required - set(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    selected = manifest[manifest["split"] == args.split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split={args.split!r}")
    if selected["source_path"].duplicated().any():
        raise ValueError("Selected source paths are not unique.")

    rows = []
    building_dir.mkdir(parents=True)
    try:
        for number, (_, item) in enumerate(selected.iterrows(), start=1):
            source_path = resolve_source_path(
                data_root,
                item.source_path,
                args.strip_source_prefix,
            )
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            relative_target = (
                Path("images") / item.method / (source_path.stem + ".jpg")
            )
            target_path = building_dir / relative_target
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source_path) as source_image:
                source_format = source_image.format
                source_width, source_height = source_image.size
                source_image.convert("RGB").save(
                    target_path,
                    format="JPEG",
                    quality=settings["quality"],
                    subsampling=settings["pillow_subsampling"],
                    progressive=settings["progressive"],
                )
            with Image.open(target_path) as target_image:
                target_width, target_height = target_image.size
                target_format = target_image.format
            if (source_width, source_height) != (
                target_width, target_height
            ):
                raise RuntimeError(f"Proxy dimensions changed: {source_path}")

            row = item.to_dict()
            row["source_reference_path"] = row["source_path"]
            row["source_path"] = relative_target.as_posix()
            row.update(
                {
                    "source_format": source_format,
                    "proxy_format": target_format,
                    "width": source_width,
                    "height": source_height,
                    "source_bytes": source_path.stat().st_size,
                    "proxy_bytes": target_path.stat().st_size,
                    "source_sha256": sha256_file(source_path),
                    "proxy_sha256": sha256_file(target_path),
                }
            )
            rows.append(row)
            if number % 250 == 0 or number == len(selected):
                print(f"Converted {number}/{len(selected)}")

        proxy_manifest = pd.DataFrame(rows)
        proxy_manifest.to_csv(building_dir / "manifest.csv", index=False)
        config = {
            "name": output_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "offline proxy for the observed Instagram web transformation",
            "is_real_instagram_data": False,
            "split": args.split,
            "images": int(len(proxy_manifest)),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "pilot_metrics": str(pilot_metrics_path),
            "pilot_metrics_sha256": sha256_file(pilot_metrics_path),
            "strip_source_prefix": args.strip_source_prefix,
            "transformation": settings,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pillow": PIL.__version__,
            },
        }
        (building_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        building_dir.replace(output_dir)
    except Exception:
        print(f"Incomplete output retained for inspection: {building_dir}")
        raise

    print(f"Created proxy dataset: {output_dir}")
    print(f"Images: {len(rows)}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
