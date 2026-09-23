#!/usr/bin/env python3
"""Create one fixed-pool train/validation TAR per DF40 method."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd


METHOD_SLUGS = {
    "SimSwap": "simswap",
    "BlendFace": "blendface",
    "InSwapper": "inswapper",
    "FaceDancer": "facedancer",
    "FSGAN": "fsgan",
    "e4s": "e4s",
    "Wav2Lip": "wav2lip",
    "FOMM": "fomm",
    "SadTalker": "sadtalker",
    "HyperReenact": "hyperreenact",
    "TPSM": "tpsm",
    "PIRender": "pirender",
    "StyleGAN3": "stylegan3",
    "DiT": "dit",
    "StyleGAN-XL": "styleganxl",
    "PixArt-alpha": "pixartalpha",
    "VQGAN": "vqgan",
    "SD2.1": "sd21",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-method", type=int, default=9900)
    parser.add_argument("--val-per-method", type=int, default=2340)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method names or slugs to package.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def stable_rank(sample_id: str, seed: int, split: str, method: str) -> str:
    value = f"{seed}|{split}|{method}|{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    split: str,
    method: str,
) -> pd.DataFrame:
    candidates = frame[(frame["split"] == split) & (frame["method"] == method)]
    if len(candidates) < count:
        raise RuntimeError(
            f"Not enough {split}/{method}: need={count}, available={len(candidates)}"
        )
    return (
        candidates.assign(
            _rank=candidates["sample_id"].map(
                lambda value: stable_rank(value, seed, split, method)
            )
        )
        .sort_values(["_rank", "sample_id"])
        .head(count)
        .drop(columns="_rank")
    )


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_file(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source), arcname=archive_name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    candidates = frame[
        (frame["label"] == "fake")
        & (frame["method"].isin(METHOD_SLUGS))
        & (frame["split"].isin(["train", "val"]))
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(path.name for path in args.output_dir.iterdir())
    if unexpected and not args.skip_existing:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}: {unexpected[:3]}"
        )

    inventory_path = args.output_dir / "inventory.json"
    inventory: dict[str, object] = {
        "protocol": "family_rotation_method_archives_v1",
        "extract_root": "deepfake_family_rotation_v1",
        "source_manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "sampling": "deterministic SHA-256 rank without replacement",
        "train_per_method": args.train_per_method,
        "val_per_method": args.val_per_method,
        "planned_missing_methods": [],
        "archives": {},
    }
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    requested = set(args.methods or ())
    selected_methods = {
        method: slug
        for method, slug in METHOD_SLUGS.items()
        if not requested or method in requested or slug in requested
    }
    if requested and not selected_methods:
        raise ValueError(f"No matching methods: {sorted(requested)}")

    for method, slug in selected_methods.items():
        selected = pd.concat(
            [
                select(
                    candidates,
                    args.train_per_method,
                    args.seed,
                    "train",
                    method,
                ),
                select(
                    candidates,
                    args.val_per_method,
                    args.seed,
                    "val",
                    method,
                ),
            ],
            ignore_index=True,
        ).sort_values(["split", "source_path"])
        if selected["sample_id"].duplicated().any():
            raise RuntimeError(f"Duplicate sample IDs in {method}")

        missing = [
            path
            for value in selected["local_source_path"]
            if not (path := project_root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{method}: first missing file {missing[0]}")

        output = args.output_dir / f"df40_{slug}_trainval_v1.tar"
        if output.exists() and args.skip_existing:
            print(f"Skipping existing {output.name}", flush=True)
            continue
        metadata = {
            "method": method,
            "family": selected["family"].iloc[0],
            "images": len(selected),
            "splits": selected.groupby("split").size().to_dict(),
        }
        manifest_name = (
            f"deepfake_family_rotation_v1/manifests/methods/{slug}_trainval.csv"
        )
        metadata_name = (
            f"deepfake_family_rotation_v1/manifests/methods/{slug}_trainval.json"
        )
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            add_bytes(archive, manifest_name, selected.to_csv(index=False).encode())
            add_bytes(
                archive,
                metadata_name,
                (json.dumps(metadata, indent=2) + "\n").encode(),
            )
            for index, row in enumerate(selected.itertuples(index=False), start=1):
                add_file(
                    archive,
                    project_root / row.local_source_path,
                    f"deepfake_family_rotation_v1/{row.source_path}",
                )
                if index % 2000 == 0:
                    print(f"{method}: {index}/{len(selected)}", flush=True)

        inventory["archives"][slug] = {
            **metadata,
            "filename": output.name,
            "bytes": output.stat().st_size,
        }
        print(f"Completed {output.name}", flush=True)

    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
