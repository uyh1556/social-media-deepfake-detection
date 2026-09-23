#!/usr/bin/env python3
"""Rewrite legacy support TAR roots for the family-rotation package."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


OLD = "deepfake_family_v1/"
NEW = "deepfake_family_rotation_v1/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for source in args.archives:
        temporary = source.with_suffix(".repacked.tar")
        if temporary.exists():
            raise FileExistsError(temporary)
        with tarfile.open(source, "r") as old, tarfile.open(
            temporary, "w", format=tarfile.PAX_FORMAT
        ) as new:
            for member in old:
                if not member.name.startswith(OLD):
                    raise ValueError(f"Unexpected member: {member.name}")
                member.name = NEW + member.name[len(OLD):]
                payload = old.extractfile(member) if member.isfile() else None
                new.addfile(member, payload)
        temporary.replace(source)
        print(f"Repacked {source}", flush=True)


if __name__ == "__main__":
    main()
