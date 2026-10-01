#!/usr/bin/env python3
"""Verify extracted S1-S6 manifests against the frozen three-seed bundle."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument(
        "--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)],
        default=[f"S{i}" for i in range(1, 7)],
    )
    args = parser.parse_args()

    required = {
        f"family_rotation_manifests/{selection.lower()}/m{model}_seed42.csv":
        args.manifest_root / selection.lower() / f"m{model}_seed42.csv"
        for selection in args.selections for model in range(1, 8)
    }
    found = set()
    with tarfile.open(args.archive, "r|*") as archive:
        for member in archive:
            extracted = required.get(member.name)
            if extracted is None:
                continue
            if not member.isfile():
                raise FileNotFoundError(f"Not a file in frozen archive: {member.name}")
            if not extracted.is_file():
                raise FileNotFoundError(f"Missing extracted manifest: {extracted}")
            stream = archive.extractfile(member)
            if stream is None:
                raise FileNotFoundError(f"Cannot read frozen archive member: {member.name}")
            expected = hashlib.sha256(stream.read()).hexdigest()
            actual = hashlib.sha256(extracted.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Manifest differs from completed-run authority: {extracted}; "
                    f"expected SHA-256 {expected}, got {actual}"
                )
            found.add(member.name)
    if missing := set(required) - found:
        raise FileNotFoundError(f"Missing frozen archive members: {sorted(missing)[:5]}")
    checked = len(found)
    print(f"Verified {checked} frozen manifests for {', '.join(args.selections)}")


if __name__ == "__main__":
    main()
