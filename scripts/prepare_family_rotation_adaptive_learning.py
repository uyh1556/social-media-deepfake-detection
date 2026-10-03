#!/usr/bin/env python3
"""Freeze validation roles and prepare exact M7 images for an adaptive pilot."""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

from create_family_rotation_method_archives import METHOD_SLUGS
from diagnose_xception_family_rotation_joint_updates import assert_disjoint

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SHA = "1b15447177f8b30f9982a6f41d45cd049443967ef92958b3166326d770af563e"
STRATEGIES = ("uniform", "difficulty", "complementarity")


def validate_protocol(protocol):
    if protocol.get("protocol") != "family_rotation_adaptive_learning_v1" or protocol.get("training_seed") not in (42, 43, 44):
        raise ValueError("Expected adaptive pilot protocol with training seed 42, 43 or 44")
    if not 0 < protocol["meta_validation_fraction"] < 1 or not 0 < protocol["uniform_floor"] <= 1:
        raise ValueError("Invalid validation fraction or uniform floor")
    if not 0 <= protocol["ema_decay"] < 1 or protocol["maximum_method_weight"] < 1:
        raise ValueError("Invalid EMA or maximum method weight")
    for key in ("probe_real", "probe_fake_per_method", "update_real", "update_fake_per_method",
                "refresh_epoch_fraction", "temporary_sgd_learning_rate", "softmax_temperature",
                "loss_denominator_floor", "maximum_relative_temporary_update"):
        if not isinstance(protocol[key], (int, float)) or not 0 < protocol[key] < float("inf"):
            raise ValueError(f"Invalid positive protocol field: {key}")
    for key in ("warmup_epoch_fraction", "minimum_complementarity_gain", "real_penalty"):
        if not 0 <= protocol[key] < float("inf"):
            raise ValueError(f"Invalid nonnegative protocol field: {key}")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_directory(root, selection, strategy, seed):
    return Path(root) / f"xception_{selection.lower()}_m7_q95_{strategy}_adaptive_v1_seed{seed}"


def components(frame):
    """Connect rows sharing any recorded source, driver, video or content identity."""
    parent = list(range(len(frame)))
    owners = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(frame.itertuples(index=False)):
        keys = [("group", row.group_id), ("content", row.content_sha256),
                ("video", f"{row.method}/{row.video_id}")]
        for column in ("source_ids", "driver_id"):
            for token in str(getattr(row, column)).replace(",", "|").split("|"):
                if token.strip():
                    # Source and driving-video IDs belong to the same FF++ namespace.
                    keys.append(("source", token.strip()))
        for key in keys:
            if not key[1]:
                continue
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i in range(len(frame)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def make_roles(frame, protocol):
    validate_protocol(protocol)
    val = frame[frame.split == "val"].reset_index(drop=True).copy()
    val["validation_role"] = "selection"
    for indices in components(val):
        key = min(val.iloc[indices].sample_id)
        value = int(hashlib.sha256(f"{protocol['role_seed']}|{key}".encode()).hexdigest()[:16], 16) / 2**64
        if value < protocol["meta_validation_fraction"]:
            val.loc[indices, "validation_role"] = "meta"
    meta, selection = [val[val.validation_role == role] for role in ("meta", "selection")]
    assert_disjoint(meta, selection)
    assert_disjoint(frame[frame.split == "train"], val)
    for role, subset in (("meta", meta), ("selection", selection)):
        counts = subset.groupby("method").size()
        for method in frame.method.unique():
            minimum = (protocol["probe_real"] if method == "original" else protocol["probe_fake_per_method"]) if role == "meta" else 1
            if counts.get(method, 0) < minimum:
                raise ValueError(f"{role}/{method}: insufficient source-disjoint validation rows ({counts.get(method, 0)} < {minimum})")
    return val


def load_roles(manifest, roles_path, protocol):
    validate_protocol(protocol)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    roles_path = Path(roles_path)
    roles = pd.read_csv(roles_path, dtype=str, keep_default_na=False)
    metadata = json.loads(roles_path.with_name("roles.json").read_text())
    if metadata["source_manifest_sha256"] != sha(manifest) or metadata["protocol"] != protocol or metadata["roles_sha256"] != sha(roles_path):
        raise RuntimeError(f"Frozen validation-role identity differs: {roles_path}")
    val = frame[frame.split == "val"].reset_index(drop=True)
    if set(roles.validation_role) != {"meta", "selection"} or not roles.drop(columns="validation_role").equals(val):
        raise RuntimeError("Validation roles must preserve every original validation row exactly")
    assert_disjoint(roles[roles.validation_role == "meta"], roles[roles.validation_role == "selection"])
    assert_disjoint(frame[frame.split == "train"], roles)
    return roles


def extract_needed(archive_path, data_root, requested, supplements=()):
    needed = {p for p in requested if not (data_root / p).is_file()}
    if not needed:
        print(f"Already available: {archive_path.name}", flush=True)
        return
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    prefix = "deepfake_family_rotation_v1/"
    print(f"Extracting selected images: {archive_path.name} ({len(needed)})", flush=True)
    for source_archive in (archive_path, *supplements):
        if not needed:
            break
        if not source_archive.is_file():
            if source_archive in supplements:
                continue
            raise FileNotFoundError(source_archive)
        if source_archive != archive_path:
            print(f"Extracting S1 supplement: {source_archive.name} ({len(needed)})", flush=True)
        with tarfile.open(source_archive, "r:") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                relative = name.removeprefix(prefix)
                if member.isfile() and name.startswith(prefix) and relative in needed:
                    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                        raise ValueError(f"Unsafe image path: {relative}")
                    destination = data_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".tmp")
                    with archive.extractfile(member) as source, temporary.open("wb") as target:
                        import shutil
                        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                    os.replace(temporary, destination)
                    needed.remove(relative)
                    if not needed:
                        break
    if needed:
        hint = f" Upload {supplements[0].name} beside the existing trainval TARs." if supplements else ""
        raise FileNotFoundError(f"Archive lacks {len(needed)} frozen images: {sorted(needed)[:5]}.{hint}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=["S2", "S3", "S4"])
    for name in ("archive-root", "data-root", "manifest-root", "output-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--protocol-config", type=Path, default=ROOT / "configs/family_rotation_adaptive_learning_v1/protocol.json")
    p.add_argument("--test-only", action="store_true", help="Extract only the canonical 21-group test images")
    p.add_argument("--manifests-only", action="store_true", help="Freeze manifests and roles without training image extraction")
    a = p.parse_args()
    if a.test_only:
        for csv_name, archive_name in (
            ("balanced_df40_test_2000_v1.csv", "all_df40_methods_test_v1.tar"),
            ("balanced_ffpp_test_2000_v1.csv", "ffpp_deepfakes_face2face_test_v1.tar")):
            frame = pd.read_csv(a.archive_root / csv_name, dtype=str, keep_default_na=False)
            if set(frame.split) != {"test"}:
                raise ValueError(f"Expected only test rows: {csv_name}")
            extract_needed(a.archive_root / archive_name, a.data_root, set(frame.source_path))
        print("Canonical test images prepared.", flush=True)
        return
    protocol = json.loads(a.protocol_config.read_text())
    bundle = a.archive_root / "family_rotation_evaluation_manifests_v1.tar.xz"
    if sha(bundle) != BUNDLE_SHA:
        raise RuntimeError("Not the frozen checkpoint-compatible manifest bundle")
    frames = []
    with tarfile.open(bundle, "r:xz") as archive:
        for selection in a.selections:
            manifest = a.manifest_root / selection.lower() / "m7_seed42.csv"
            content = archive.extractfile(f"family_rotation_manifests/{selection.lower()}/m7_seed42.csv").read()
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.is_file() and manifest.read_bytes() != content:
                raise RuntimeError(f"Existing manifest differs: {manifest}")
            if not manifest.exists():
                temporary = manifest.with_name(manifest.name + ".tmp")
                temporary.write_bytes(content)
                os.replace(temporary, manifest)
            frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            from train_xception_family_coverage import FIXED_BUDGETS, validate_manifest
            cfg = json.loads((ROOT / "configs/family_rotation_v1/selections.json").read_text())
            methods = [cfg["families"][f][letter] for f in ("FS", "FR", "EFS") for letter in cfg["selections"][selection]]
            validate_manifest(manifest, methods, FIXED_BUDGETS)
            directory = a.output_root / selection.lower()
            directory.mkdir(parents=True, exist_ok=True)
            roles_path = directory / "validation_roles.csv"
            if roles_path.exists() and (directory / "roles.json").exists():
                roles = load_roles(manifest, roles_path, protocol)
            else:
                roles = make_roles(frame, protocol)
                if roles_path.exists():
                    if not pd.read_csv(roles_path, dtype=str, keep_default_na=False).equals(roles):
                        raise RuntimeError(f"Incomplete role CSV differs from deterministic roles: {roles_path}")
                else:
                    atomic_csv(roles_path, roles)
                atomic_json(directory / "roles.json", {"source_manifest_sha256": sha(manifest),
                    "roles_sha256": sha(roles_path), "protocol": protocol,
                    "counts": [
                        {"role": role, "method": method, "images": int(count)}
                        for (role, method), count in roles.groupby(["validation_role", "method"]).size().items()]})
            print(selection, roles.groupby("validation_role").size().to_dict(), flush=True)
            frames.append(frame)
    if a.manifests_only:
        print("Frozen manifests and validation roles prepared.", flush=True)
        return
    required = pd.concat(frames).drop_duplicates("source_path")
    for method, part in required.groupby("method"):
        name = "real_trainval_v1.tar" if method == "original" else f"df40_{METHOD_SLUGS[method]}_trainval_v1.tar"
        supplements = (a.archive_root / "df40_simswap_s1_supplement_v1.tar",) if method == "SimSwap" and "S1" in a.selections else ()
        extract_needed(a.archive_root / name, a.data_root, set(part.source_path), supplements)
    print("All frozen M7 training/validation images prepared; no test archive required.", flush=True)


if __name__ == "__main__":
    main()
