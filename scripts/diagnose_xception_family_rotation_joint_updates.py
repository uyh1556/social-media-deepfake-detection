#!/usr/bin/env python3
"""Probe cross-method influence of temporary SGD updates to fixed-Q95 M7."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def methods_for(config, selection):
    return [config["families"][family][letter]
            for family in ("FS", "FR", "EFS")
            for letter in config["selections"][selection]]


def checkpoint_path(args, config, selection):
    name = config["models"]["M7"]["name"]
    if selection == "S1":
        run = f"xception_m7_{name}_canonical256_jpegq95_letterbox299_control_v1_seed42"
        return args.s1_fixed_runs_root / run / "best.pt"
    run = f"xception_{selection.lower()}_m7_{name}_fixed_q95_family_rotation_v1_seed42"
    return args.rotation_runs_root / run / "best.pt"


def validate_config(saved, manifest, checkpoint):
    if saved.get("manifest_sha256") != digest(manifest) or saved.get("seed") != 42:
        raise RuntimeError(f"Frozen manifest/seed mismatch: {checkpoint}")
    if saved.get("model") not in {"xception", "legacy_xception"}:
        raise RuntimeError(f"Expected Xception model: {checkpoint}")
    if saved.get("preprocessing_name") != "canonical256_jpegq95_letterbox299":
        raise RuntimeError(f"Expected controlled Q95 preprocessing: {checkpoint}")
    prep, data = saved.get("preprocessing", {}), saved.get("data_config", {})
    expected = {"canonical_size": 256, "jpeg_quality": 95, "jpeg_subsampling": 2,
                "jpeg_optimize": False, "jpeg_progressive": False}
    if any(prep.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Expected fixed-Q95 canonical256 checkpoint: {checkpoint}")
    if prep.get("train_jpeg_qualities") or prep.get("train_noise", {}).get("probability", 0):
        raise RuntimeError(f"Checkpoint has Mixed-JPEG or noise augmentation: {checkpoint}")
    if (list(data.get("input_size", [])) != [3, 299, 299]
            or data.get("interpolation") != "bicubic"
            or list(data.get("mean", [])) != [0.5] * 3
            or list(data.get("std", [])) != [0.5] * 3):
        raise RuntimeError(f"Unexpected input preprocessing: {checkpoint}")


def read_sources(args, config):
    import pandas as pd

    sources = {}
    for selection in args.selections:
        manifest = args.manifest_root / selection.lower() / "m7_seed42.csv"
        checkpoint = checkpoint_path(args, config, selection)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        saved = json.loads((checkpoint.parent / "config.json").read_text())
        validate_config(saved, manifest, checkpoint)
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        required = {"sample_id", "source_path", "split", "label", "method", "group_id",
                    "video_id", "source_ids", "driver_id", "content_sha256"}
        if required - set(frame.columns):
            raise ValueError(f"Missing manifest columns: {required - set(frame.columns)}")
        methods = methods_for(config, selection)
        if (set(frame["split"]) != {"train", "val"}
                or set(frame["method"]) != {"original", *methods}
                or frame.sample_id.duplicated().any()
                or frame.source_path.duplicated().any()):
            raise ValueError(f"Unexpected frozen M7 manifest: {manifest}")
        expected_labels = frame.method.map(lambda m: "real" if m == "original" else "fake")
        if not expected_labels.equals(frame.label):
            raise ValueError(f"Wrong method/label provenance: {manifest}")
        for value in frame.source_path:
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe source path: {value}")
        sources[selection] = {"manifest": manifest, "checkpoint": checkpoint,
                              "saved": saved, "frame": frame, "methods": methods}
        print(f"Verified fixed-Q95 M7: {selection}", flush=True)
    return sources


def rank_pool(pool, seed, split, method):
    ranked = pool.copy()
    ranked["_rank"] = ranked.sample_id.map(
        lambda value: hashlib.sha256(f"{seed}|{split}|{method}|{value}".encode()).hexdigest())
    return ranked.sort_values(["_rank", "sample_id"]).drop(columns="_rank")


def choose_samples(frame, methods, protocol, available):
    import pandas as pd

    parts, excluded = [], {}
    for split in ("train", "val"):
        for method in ["original", *methods]:
            original = frame[(frame.split == split) & (frame.method == method)]
            pool = original[original.source_path.isin(available)]
            excluded[f"{split}/{method}"] = len(original) - len(pool)
            is_real = method == "original"
            count = (protocol["update_real_per_repeat"] if is_real
                     else protocol["update_fake_per_method_per_repeat"]) if split == "train" else (
                         protocol["probe_real"] if is_real else protocol["probe_fake_per_method"])
            repeats = protocol["repeats"] if split == "train" else 1
            if len(pool) < count * repeats:
                raise ValueError(f"{split}/{method}: need {count * repeats} available frozen images, found {len(pool)}")
            ranked = rank_pool(pool, protocol["sampling_seed"], split, method)
            for repeat in range(repeats):
                selected = ranked.iloc[repeat * count:(repeat + 1) * count].copy()
                selected["diagnostic_role"] = "update" if split == "train" else "probe"
                selected["repeat"] = repeat + 1 if split == "train" else 0
                parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    if result.sample_id.duplicated().any():
        raise ValueError("Diagnostic sampling must not duplicate samples across repeats")
    assert_disjoint(result[result.diagnostic_role == "update"], result[result.diagnostic_role == "probe"])
    return result, excluded


def assert_disjoint(update, probe):
    def tokens(series):
        return {token.strip() for value in series for token in str(value).replace(",", "|").split("|") if token.strip()}

    for column in ("sample_id", "source_path", "content_sha256", "group_id", "source_ids", "driver_id"):
        if overlap := tokens(update[column]) & tokens(probe[column]):
            raise RuntimeError(f"Update/probe leakage in {column}: {sorted(overlap)[:5]}")
    a = set(zip(update.method, update.video_id))
    b = set(zip(probe.method, probe.video_id))
    if a & b:
        raise RuntimeError("Update/probe method-video overlap")


def preparation_identity(source, protocol):
    return {"protocol": protocol, "manifest_sha256": digest(source["manifest"]),
            "checkpoint_path": str(source["checkpoint"]), "saved_config": source["saved"]}


def prepare_data(args, config, protocol, sources):
    """Index uncompressed TAR headers, then copy only selected regular-file members."""
    import pandas as pd
    from create_family_rotation_method_archives import METHOD_SLUGS

    ready = True
    for selection, source in sources.items():
        previous = args.output_root / selection.lower() / "preparation.json"
        if not previous.is_file():
            ready = False
            continue
        if json.loads(previous.read_text())["identity"] != preparation_identity(source, protocol):
            raise RuntimeError(f"Preparation identity differs: {previous}")
        samples_file = previous.parent / "samples.csv"
        if not samples_file.is_file():
            raise FileNotFoundError(samples_file)
        paths = pd.read_csv(samples_file, dtype=str, keep_default_na=False).source_path
        ready = ready and all((args.data_root / path).is_file() for path in paths)
    if ready:
        for selection, source in sources.items():
            load_samples(args, selection, source, protocol)
        print("All frozen diagnostic subsets already available.", flush=True)
        return

    requested = set().union(*(set(value["frame"].source_path) for value in sources.values()))
    available = {value for value in requested if (args.data_root / value).is_file()}
    inventory, archive_paths = {}, {}
    all_methods = {"original", *(method for value in sources.values() for method in value["methods"])}
    prefix = "deepfake_family_rotation_v1/"
    for method in sorted(all_methods):
        name = "real_trainval_v1.tar" if method == "original" else f"df40_{METHOD_SLUGS[method]}_trainval_v1.tar"
        archive = args.archive_root / name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        print(f"Indexing archive: {name}", flush=True)
        entries = {}
        with tarfile.open(archive, "r:") as stream:
            for member in stream:
                name_in_archive = member.name.removeprefix("./")
                if member.isfile() and name_in_archive.startswith(prefix):
                    relative = name_in_archive[len(prefix):]
                    if relative in requested:
                        if relative in entries:
                            raise ValueError(f"Duplicate TAR image member: {relative}")
                        entries[relative] = member
        inventory[method], archive_paths[method] = entries, archive
        available.update(entries)

    selected_paths = set()
    for selection, source in sources.items():
        directory = args.output_root / selection.lower()
        identity = preparation_identity(source, protocol)
        previous = directory / "preparation.json"
        if previous.is_file():
            record = json.loads(previous.read_text())
            samples = pd.read_csv(directory / "samples.csv", dtype=str, keep_default_na=False)
            excluded = record["excluded_unavailable_frozen_rows"]
        else:
            samples, excluded = choose_samples(source["frame"], source["methods"], protocol, available)
            write_csv(directory / "samples.csv", samples)
            write_json(previous, {"identity": identity, "excluded_unavailable_frozen_rows": excluded,
                                  "sample_count": len(samples), "sampling_uses_only_original_manifest_rows": True})
        selected_paths.update(samples.source_path)
        print(f"{selection}: {len(samples)} diagnostic images; unavailable manifest rows: {sum(excluded.values())}", flush=True)

    for method, entries in inventory.items():
        needed = sorted(value for value in selected_paths & entries.keys()
                        if not (args.data_root / value).is_file())
        if not needed:
            print(f"Already available: {method}", flush=True)
            continue
        print(f"Extracting selected images: {method}, {len(needed)}", flush=True)
        with tarfile.open(archive_paths[method], "r:") as stream:
            for relative in needed:
                destination = args.data_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".joint_update.tmp")
                try:
                    with stream.extractfile(entries[relative]) as input_file, temporary.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    missing = [value for value in selected_paths if not (args.data_root / value).is_file()]
    if missing:
        raise FileNotFoundError(f"Diagnostic images unavailable after extraction: {missing[:5]}")
    print("Diagnostic subsets ready. No full training/test archive was extracted.", flush=True)


def load_samples(args, selection, source, protocol):
    import pandas as pd

    directory = args.output_root / selection.lower()
    preparation = json.loads((directory / "preparation.json").read_text())
    if preparation["identity"] != preparation_identity(source, protocol):
        raise RuntimeError(f"Preparation changed; use a new output root: {directory}")
    samples = pd.read_csv(directory / "samples.csv", dtype=str, keep_default_na=False)
    frozen = source["frame"].set_index("sample_id", drop=False)
    if samples.sample_id.duplicated().any() or not set(samples.sample_id).issubset(frozen.index):
        raise ValueError("Prepared sample IDs differ from frozen manifest")
    for column in source["frame"].columns:
        if not samples[column].tolist() == frozen.loc[samples.sample_id, column].tolist():
            raise ValueError(f"Prepared samples changed frozen {column}")
    missing = [value for value in samples.source_path if not (args.data_root / value).is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared images missing: {missing[:5]}")
    update = samples[samples.diagnostic_role == "update"]
    probe = samples[samples.diagnostic_role == "probe"]
    assert_disjoint(update, probe)
    expected_counts = {"original": protocol["probe_real"],
                       **{m: protocol["probe_fake_per_method"] for m in source["methods"]}}
    if probe.groupby("method").size().to_dict() != expected_counts:
        raise ValueError("Wrong prepared validation counts")
    for repeat in range(1, protocol["repeats"] + 1):
        counts = update[update["repeat"] == str(repeat)].groupby("method").size().to_dict()
        expected_counts = {"original": protocol["update_real_per_repeat"],
                           **{m: protocol["update_fake_per_method_per_repeat"] for m in source["methods"]}}
        if counts != expected_counts:
            raise ValueError(f"Wrong prepared update counts for repeat {repeat}")
    return samples


def auc_score(real, fake):
    """Mann-Whitney form of ROC AUC, including average ranks for tied scores."""
    import numpy as np
    import pandas as pd

    ranks = pd.Series(np.r_[real, fake]).rank(method="average").to_numpy()
    n_real, n_fake = len(real), len(fake)
    return float((ranks[n_real:].sum() - n_fake * (n_fake + 1) / 2) / (n_real * n_fake))


def score_metrics(frame, scores, losses, methods, threshold):
    real_mask = frame.label.eq("real").to_numpy()
    real_scores, real_losses = scores[real_mask], losses[real_mask]
    result = {}
    for method in methods:
        mask = frame.method.eq(method).to_numpy()
        fake_scores, fake_losses = scores[mask], losses[mask]
        result[method] = {
            "auc": auc_score(real_scores, fake_scores),
            "balanced_ce": float((real_losses.mean() + fake_losses.mean()) / 2),
            "real_ce": float(real_losses.mean()), "fake_ce": float(fake_losses.mean()),
            "real_fpr": float((real_scores >= threshold).mean()),
            "fake_recall": float((fake_scores >= threshold).mean()),
            "real_mean_fake_score": float(real_scores.mean()),
            "fake_mean_fake_score": float(fake_scores.mean()),
        }
    return result


def mean_gradient(model, loader, device, description, *, progress_enabled=True):
    import torch
    from tqdm.auto import tqdm

    model.eval()
    model.zero_grad(set_to_none=True)
    total = len(loader.dataset)
    for images, labels in tqdm(loader, desc=description, leave=False, disable=not progress_enabled):
        logits = model(images.to(device, non_blocking=True))
        loss = torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum") / total
        loss.backward()
    vector = torch.cat([p.grad.detach().float().cpu().reshape(-1) if p.grad is not None
                        else torch.zeros(p.numel()) for p in model.parameters() if p.requires_grad])
    model.zero_grad(set_to_none=True)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Nonfinite diagnostic gradient")
    return vector


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b) / denominator) if denominator > 0 else None


def run_selection(args, config, protocol, selection, source, samples):
    import numpy as np
    import pandas as pd
    import PIL
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from xception_preprocessing import evaluation_transform_from_checkpoint

    directory = args.output_root / selection.lower()
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    normalized_config = json.loads(json.dumps(checkpoint["config"], default=str))
    if normalized_config != source["saved"]:
        raise RuntimeError("Saved config.json differs from checkpoint config")
    if checkpoint.get("label_map") != {"real": 0, "fake": 1}:
        raise ValueError("Unexpected checkpoint labels")
    identity = {"preparation": preparation_identity(source, protocol),
                "checkpoint_sha256": digest(source["checkpoint"]),
                "samples_sha256": digest(directory / "samples.csv"),
                "implementation_sha256": digest(Path(__file__)),
                "selection_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "versions": {"torch": torch.__version__, "timm": timm.__version__,
                             "Pillow": PIL.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    completion = directory / "diagnostic_summary.json"
    if completion.is_file():
        previous = json.loads(completion.read_text())
        if previous["identity"] != identity:
            raise RuntimeError(f"Completed selection identity differs: {directory}")
        paths = [directory / "update_target_summary.csv", directory / "gradient_alignment.csv"]
        if all(path.is_file() for path in paths):
            print(f"Reusing completed selection: {selection}", flush=True)
            return tuple(pd.read_csv(path) for path in paths)
    state = checkpoint["model_state_dict"]
    del checkpoint
    model = timm.create_model(source["saved"]["model"], pretrained=False, num_classes=2)
    model.load_state_dict(state)
    del state
    device = torch.device(args.device)
    model.to(device).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    original = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    frozen_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    transform = evaluation_transform_from_checkpoint({"config": source["saved"]})

    class Images(Dataset):
        def __init__(self, frame):
            self.rows = list(frame[["source_path", "label"]].itertuples(index=False, name=None))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            path, label = self.rows[index]
            with Image.open(args.data_root / path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, 0 if label == "real" else 1

    def loader(frame, size):
        return DataLoader(Images(frame), batch_size=size, shuffle=False, num_workers=args.workers,
                          pin_memory=device.type == "cuda")

    probe = samples[samples.diagnostic_role == "probe"].reset_index(drop=True)
    probe_loader = loader(probe, args.batch_size)

    def evaluate(description):
        scores, losses = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(probe_loader, desc=description, leave=True):
                logits = model(images.to(device, non_blocking=True))
                scores.append(logits.softmax(1)[:, 1].cpu().numpy())
                losses.append(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="none").cpu().numpy())
        scores, losses = np.concatenate(scores), np.concatenate(losses)
        if not np.isfinite(scores).all() or not np.isfinite(losses).all():
            raise FloatingPointError("Nonfinite validation predictions/losses")
        return scores, losses

    print(f"\n===== {selection}: Q95 M7 epoch {identity['checkpoint_epoch']} =====", flush=True)
    baseline_scores, baseline_losses = evaluate(f"{selection} baseline")
    baseline = score_metrics(probe, baseline_scores, baseline_losses, source["methods"], protocol["decision_threshold"])
    baseline_frame = probe.copy()
    baseline_frame["fake_score"] = baseline_scores
    baseline_frame["ce"] = baseline_losses
    write_csv(directory / "baseline_predictions.csv", baseline_frame)
    write_json(directory / "baseline_metrics.json", {"identity": identity, "metrics": baseline})
    rows, alignments = [], []
    family = {method: f for f, members in config["families"].items() for method in members.values()}
    expected_updates = protocol["repeats"] * (len(source["methods"]) + 2) * len(protocol["learning_rates"])
    progress = 0
    try:
        for repeat in range(1, protocol["repeats"] + 1):
            update = samples[(samples.diagnostic_role == "update") & (samples["repeat"] == str(repeat))]
            real = mean_gradient(model, loader(update[update.label == "real"], args.update_batch_size),
                                 device, f"{selection} repeat{repeat} Real gradient")
            fake = {}
            for method in source["methods"]:
                fake[method] = mean_gradient(model, loader(update[update.method == method], args.update_batch_size),
                                            device, f"{selection} repeat{repeat} {method} gradient")
            mean_fake = sum(fake.values()) / len(fake)
            uniform = 0.5 * real + 0.5 * mean_fake
            for source_method in source["methods"]:
                for target_method in source["methods"]:
                    a, b = fake[source_method], fake[target_method]
                    alignments.append({"selection": selection, "repeat": repeat,
                        "source_method": source_method, "target_method": target_method,
                        "source_family": family[source_method], "target_family": family[target_method],
                        "fake_only_cosine": cosine(a, b),
                        "balanced_task_cosine": cosine(0.5 * (real + a), 0.5 * (real + b)),
                        "source_fake_real_cosine": cosine(a, real),
                        "source_fake_gradient_norm": float(a.norm()),
                        "target_fake_gradient_norm": float(b.norm()), "real_gradient_norm": float(real.norm()),
                        "uniform_update_target_dot": float(uniform.dot(0.5 * (real + b)))})
            directions = {method: 0.5 * (real + fake[method]) for method in source["methods"]}
            directions.update(real_only=0.5 * real, uniform_all6=uniform)
            for source_method, gradient in directions.items():
                for learning_rate in protocol["learning_rates"]:
                    progress += 1
                    slug = source_method.lower().replace("/", "_")
                    branch = directory / f"repeat{repeat}" / slug / f"lr{learning_rate:g}"
                    completed = branch / "metrics.json"
                    branch_identity = {"run": identity, "repeat": repeat, "source_method": source_method,
                                       "learning_rate": learning_rate}
                    if completed.is_file():
                        previous = json.loads(completed.read_text())
                        if previous["identity"] != branch_identity:
                            raise RuntimeError(f"Completed branch identity differs: {branch}")
                        rows.extend(previous["rows"])
                        print(f"[{progress}/{expected_updates}] Reusing {selection} {source_method} repeat{repeat}", flush=True)
                        continue
                    print(f"\n[{progress}/{expected_updates}] {selection} / repeat{repeat} / {source_method} / SGD lr={learning_rate:g}", flush=True)
                    step = learning_rate * gradient.to(device)
                    relative_norm = float(step.norm() / original.norm().clamp_min(1e-12))
                    if not math.isfinite(relative_norm) or relative_norm > 0.01:
                        raise RuntimeError(f"Update exceeds local diagnostic range: relative norm={relative_norm}")
                    torch.nn.utils.vector_to_parameters(original - step, parameters)
                    try:
                        scores, losses = evaluate(f"{selection} updated by {source_method}")
                    finally:
                        torch.nn.utils.vector_to_parameters(original, parameters)
                    after = score_metrics(probe, scores, losses, source["methods"], protocol["decision_threshold"])
                    branch_rows = []
                    for target in source["methods"]:
                        row = {"selection": selection, "repeat": repeat, "source_method": source_method,
                               "source_family": family.get(source_method, "control"), "target_method": target,
                               "target_family": family[target], "self_update": source_method == target,
                               "learning_rate": learning_rate, "gradient_norm": float(gradient.norm()),
                               "relative_parameter_update_norm": relative_norm}
                        for metric, value in after[target].items():
                            row[f"baseline_{metric}"] = baseline[target][metric]
                            row[f"after_{metric}"] = value
                            row[f"delta_{metric}"] = value - baseline[target][metric]
                        branch_rows.append(row)
                    paired = probe[["sample_id", "source_path", "label", "method", "group_id"]].copy()
                    paired["baseline_fake_score"], paired["after_fake_score"] = baseline_scores, scores
                    paired["delta_fake_score"] = scores - baseline_scores
                    paired["baseline_ce"], paired["after_ce"] = baseline_losses, losses
                    write_csv(branch / "paired_predictions.csv", paired)
                    write_json(completed, {"identity": branch_identity, "rows": branch_rows})
                    rows.extend(branch_rows)
            write_csv(directory / "update_target_summary.csv", pd.DataFrame(rows))
            write_csv(directory / "gradient_alignment.csv", pd.DataFrame(alignments))
    finally:
        torch.nn.utils.vector_to_parameters(original, parameters)
    if any(not torch.equal(value, frozen_buffers[name]) for name, value in model.named_buffers()):
        raise RuntimeError("Model buffers changed during the eval-mode diagnostic")
    write_json(directory / "diagnostic_summary.json", {"identity": identity,
        "completed_updates": expected_updates, "original_checkpoint_modified": False,
        "batchnorm_buffers_unchanged": True, "probe_images": len(probe), "unseen_used": False,
        "interpretation": protocol["interpretation"]})
    return pd.DataFrame(rows), pd.DataFrame(alignments)


def save_aggregate(args, protocol, results, alignments):
    import pandas as pd

    rows = pd.concat(results, ignore_index=True)
    write_csv(args.output_root / "update_target_summary.csv", rows)
    write_csv(args.output_root / "gradient_alignment.csv", pd.concat(alignments, ignore_index=True))
    metrics = ["delta_auc", "delta_balanced_ce", "delta_real_ce", "delta_fake_ce",
               "delta_real_fpr", "delta_fake_recall", "delta_real_mean_fake_score", "delta_fake_mean_fake_score"]
    means = rows.groupby(["selection", "learning_rate", "source_method", "source_family", "target_method", "target_family"], as_index=False)[metrics].mean()
    consistency = rows.assign(auc_improved=rows.delta_auc > 0, balanced_ce_improved=rows.delta_balanced_ce < 0).groupby(
        ["selection", "learning_rate", "source_method", "target_method"], as_index=False).agg(
            auc_improved_repeats=("auc_improved", "sum"), balanced_ce_improved_repeats=("balanced_ce_improved", "sum"),
            repeats=("repeat", "nunique"))
    means = means.merge(consistency, on=["selection", "learning_rate", "source_method", "target_method"])
    controls = means[means.source_method.isin(["real_only", "uniform_all6"])]
    methods = means[~means.source_method.isin(["real_only", "uniform_all6"])].copy()
    for control in ("real_only", "uniform_all6"):
        reference = controls[controls.source_method == control][["selection", "learning_rate", "target_method", *metrics]]
        reference = reference.rename(columns={m: f"{control}_{m}" for m in metrics})
        methods = methods.merge(reference, on=["selection", "learning_rate", "target_method"], validate="many_to_one")
        methods[f"auc_gain_vs_{control}"] = methods.delta_auc - methods[f"{control}_delta_auc"]
        methods[f"balanced_ce_change_vs_{control}"] = methods.delta_balanced_ce - methods[f"{control}_delta_balanced_ce"]
    write_csv(args.output_root / "method_interaction_summary.csv", methods)
    cross = methods[methods.source_method != methods.target_method]
    family = cross.groupby(["selection", "learning_rate", "source_family", "target_family"], as_index=False)[
        ["delta_auc", "delta_balanced_ce", "delta_real_fpr", "auc_gain_vs_real_only", "auc_gain_vs_uniform_all6"]].mean()
    write_csv(args.output_root / "family_interaction_summary.csv", family)
    write_csv(args.output_root / "control_summary.csv", controls)
    for selection, group in methods.groupby("selection"):
        rate = protocol["learning_rates"][0]
        matrix = group[group.learning_rate == rate].pivot(index="source_method", columns="target_method", values="delta_balanced_ce")
        write_csv(args.output_root / f"{selection.lower()}_balanced_ce_delta_matrix.csv", matrix.reset_index())
    report = ["# Q95 M7 기법 간 임시 업데이트 진단", "",
        "기존 best.pt 주변에서 한 번의 임시 SGD 업데이트가 다른 seen 기법에 주는 영향을 측정합니다.",
        "Train subset으로 업데이트 방향을 계산하고 source-disjoint validation subset에서 평가합니다.",
        "3회는 다른 train subset 반복이며 학습 seed 반복/신뢰구간이 아닙니다.",
        "AUC 변화는 양수가 개선, balanced CE 변화는 음수가 개선입니다. Real FPR을 함께 확인하세요.",
        "real_only는 공통 Real 성분, uniform_all6는 같은 subset들의 균등 공동 업데이트 대조군입니다.",
        "학습 당시 AdamW를 재개한 것이 아니라 momentum/weight decay 없는 SGD 방향 검사입니다.",
        "best checkpoint의 국소 효과로 학습 초기부터 충돌이 있었다고 단정할 수 없습니다.", "",
        "작은 한 번의 업데이트에서 AUC가 그대로일 수 있습니다. CE 변화와 gradient/update 크기도 함께 확인하세요.", "",
        "## Family 간 평균 영향 (같은 method 자체 업데이트 제외)", "",
        "| selection | source family | target family | ΔAUC | Δbalanced CE | ΔReal FPR |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in family.itertuples(index=False):
        report.append(f"| {row.selection} | {row.source_family} | {row.target_family} | {row.delta_auc:+.6f} | {row.delta_balanced_ce:+.6f} | {row.delta_real_fpr:+.6f} |")
    report += ["", "핵심 파일: method_interaction_summary.csv, family_interaction_summary.csv, gradient_alignment.csv.",
               "음의 gradient cosine만으로 실패 원인을 확정하지 않습니다. 실제 validation 변화와 Real-only 대조군을 함께 봅니다.",
               "기존 checkpoint·manifest·dataset은 변경하지 않으며 unseen·WildDeepfake를 사용하지 않습니다."]
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n")
    write_json(args.output_root / "diagnostic_summary.json", {"protocol": protocol,
        "selections": args.selections, "completed_selections": len(results),
        "temporary_updates": len(rows) // 6, "probe_target_rows": len(rows),
        "new_model_trained": False, "unseen_used": False, "wilddeepfake_used": False})


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", nargs="+", choices=[f"S{i}" for i in range(1, 7)], default=["S2", "S3", "S4"])
    for name in ("data-root", "manifest-root", "rotation-runs-root", "s1-fixed-runs-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, help="Existing Drive trainval TAR directory; used only for preparation")
    parser.add_argument("--config", type=Path, default=root / "configs/family_rotation_v1/selections.json")
    parser.add_argument("--protocol-config", type=Path, default=root / "configs/family_rotation_joint_update_diagnostic_v1/protocol.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if len(set(args.selections)) != len(args.selections):
        parser.error("Duplicate selections")
    if args.batch_size < 1 or args.update_batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size/workers")
    if args.prepare_only and args.archive_root is None:
        parser.error("--archive-root is required with --prepare-only")
    return args


def main():
    args = parse_args()
    config, protocol = json.loads(args.config.read_text()), json.loads(args.protocol_config.read_text())
    if (protocol["training_protocol"] != "fixed_q95" or protocol["training_seed"] != 42
            or protocol["unseen_used"] or protocol["wilddeepfake_used"]):
        raise ValueError("Expected seed42 fixed-Q95 seen-only protocol")
    for key in ("repeats", "update_real_per_repeat", "update_fake_per_method_per_repeat", "probe_real", "probe_fake_per_method"):
        if not isinstance(protocol[key], int) or protocol[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not protocol["learning_rates"] or any(not math.isfinite(value) or value <= 0 for value in protocol["learning_rates"]):
        raise ValueError("Invalid learning rates")
    sources = read_sources(args, config)
    if args.prepare_only:
        prepare_data(args, config, protocol, sources)
        return
    samples = {s: load_samples(args, s, value, protocol) for s, value in sources.items()}
    if args.check_only:
        print("All diagnostic subsets and checkpoint configs verified.", flush=True)
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime")
    torch.manual_seed(protocol["sampling_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results, alignments = [], []
    for selection, source in sources.items():
        rows, gradients = run_selection(args, config, protocol, selection, source, samples[selection])
        results.append(rows)
        alignments.append(gradients)
        save_aggregate(args, protocol, results, alignments)
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(f"Done. Results: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
