#!/usr/bin/env python3
"""Package only the images required by the fixed M0-M7 study."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import pandas as pd


ARCHIVES = {
    "real_trainval": "real_trainval_v1.tar",
    "ffpp_fake_trainval": "ffpp_fake_trainval_v1.tar",
    "df40_fs_trainval": "df40_fs_trainval_v1.tar",
    "df40_fr_trainval": "df40_fr_trainval_v1.tar",
    "df40_efs_trainval": "df40_efs_trainval_v1.tar",
    "all_methods_test": "all_methods_test_v1.tar",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--balanced-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_condition(root: Path, model_id: str) -> pd.DataFrame:
    return pd.read_csv(
        root / "conditions" / f"{model_id.lower()}_seed42.csv",
        dtype=str,
        keep_default_na=False,
    )


def unique_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop_duplicates("sample_id").copy()
    if result["source_path"].duplicated().any():
        raise RuntimeError("Duplicate portable source_path")
    return result.sort_values("source_path").reset_index(drop=True)


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_file(
    archive: tarfile.TarFile,
    source: Path,
    archive_name: str,
) -> None:
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
    m0 = load_condition(args.split_root, "M0")
    m1 = load_condition(args.split_root, "M1")
    m2 = load_condition(args.split_root, "M2")
    m3 = load_condition(args.split_root, "M3")
    balanced_test = pd.read_csv(
        args.balanced_test,
        dtype=str,
        keep_default_na=False,
    )

    modules = {
        "real_trainval": unique_rows(m0[m0["label"] == "real"]),
        "ffpp_fake_trainval": unique_rows(m0[m0["label"] == "fake"]),
        "df40_fs_trainval": unique_rows(m1[m1["label"] == "fake"]),
        "df40_fr_trainval": unique_rows(m2[m2["label"] == "fake"]),
        "df40_efs_trainval": unique_rows(m3[m3["label"] == "fake"]),
        "all_methods_test": unique_rows(balanced_test),
    }
    expected_counts = {
        "real_trainval": 24480,
        "ffpp_fake_trainval": 24480,
        "df40_fs_trainval": 24480,
        "df40_fr_trainval": 24480,
        "df40_efs_trainval": 24480,
        "all_methods_test": 30000,
    }
    actual_counts = {key: len(value) for key, value in modules.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected module counts: {actual_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, object] = {
        "protocol": "family_coverage_colab_archives_v1",
        "extract_root": "deepfake_family_v1",
        "archives": {},
    }
    for module, frame in modules.items():
        output = args.output_dir / ARCHIVES[module]
        if output.exists():
            raise FileExistsError(output)
        missing = [
            path
            for value in frame["local_source_path"]
            if not (path := project_root / value).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{module}: first missing file {missing[0]}")

        csv_payload = frame.to_csv(index=False).encode("utf-8")
        metadata = {
            "module": module,
            "images": len(frame),
            "methods": frame.groupby("method").size().to_dict(),
            "splits": frame.groupby("split").size().to_dict(),
        }
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            add_bytes(
                archive,
                f"deepfake_family_v1/manifests/modules/{module}.csv",
                csv_payload,
            )
            add_bytes(
                archive,
                f"deepfake_family_v1/manifests/modules/{module}.json",
                (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
            )
            for index, row in enumerate(frame.itertuples(index=False), start=1):
                add_file(
                    archive,
                    project_root / row.local_source_path,
                    f"deepfake_family_v1/{row.source_path}",
                )
                if index % 2000 == 0:
                    print(f"{module}: {index}/{len(frame)}", flush=True)

        archive_record = {
            **metadata,
            "filename": output.name,
            "bytes": output.stat().st_size,
        }
        inventory["archives"][module] = archive_record
        print(json.dumps(archive_record, ensure_ascii=False), flush=True)

    (args.output_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
