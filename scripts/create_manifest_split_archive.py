import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Archive only the image files referenced by one manifest split."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        default=[],
        help="Additional file under data-root to include; may be repeated.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_tar_info(info):
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    return info


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output.exists():
        raise FileExistsError(output)

    paths = []
    distribution = Counter()
    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"split", "label", "method", "source_path"}
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != args.split:
                continue
            relative = Path(row["source_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe source path: {relative}")
            path = data_root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(relative)
            distribution[(row["label"], row["method"])] += 1
    if not paths:
        raise ValueError(f"No manifest rows found for split={args.split!r}")
    if len(paths) != len(set(paths)):
        raise RuntimeError("The selected split repeats physical source paths.")

    includes = []
    for value in args.include:
        path = value if value.is_absolute() else data_root / value
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(data_root)
        except ValueError as error:
            raise ValueError(f"Included file is outside data root: {path}") from error
        includes.append(relative)

    members = sorted(set(paths + includes), key=lambda path: path.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.GNU_FORMAT) as archive:
        for relative in members:
            archive.add(
                data_root / relative,
                arcname=relative.as_posix(),
                recursive=False,
                filter=normalized_tar_info,
            )

    result = {
        "output": str(output),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "split": args.split,
        "images": len(paths),
        "additional_files": len(includes),
        "distribution": {
            f"{label}:{method}": count
            for (label, method), count in sorted(distribution.items())
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
