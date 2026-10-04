#!/usr/bin/env python3
"""Analyze saved adaptive predictions and weights; CPU only, no inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def operating_point(real, fake, cap):
    """Most permissive empirical threshold with FPR <= cap; reject score ties."""
    real = np.sort(np.asarray(real, dtype=float))
    cutoff = real[len(real) - int(np.floor(cap * len(real))) - 1]
    return float(np.mean(np.asarray(fake) > cutoff)), float(np.mean(real > cutoff)), float(cutoff)


def auc(real, fake):
    real = np.sort(np.asarray(real, dtype=float))
    lower = np.searchsorted(real, fake, side="left")
    upper = np.searchsorted(real, fake, side="right")
    return float(np.mean((lower + 0.5 * (upper - lower)) / len(real)))


def analyze_predictions(frame, selection, seed, strategy, selected):
    expected = {"original", "Deepfakes", "Face2Face", *selected["all"]}
    if set(frame.method) != expected or not frame.groupby("method").size().eq(2000).all():
        raise ValueError("Expected all 21 groups, 2000 samples each")
    if frame.sample_id.duplicated().any() or not frame.split.eq("test").all():
        raise ValueError("Duplicate IDs or non-test samples")
    expected_label = np.where(frame.method.eq("original"), "real", "fake")
    if not np.array_equal(frame.label.to_numpy(), expected_label):
        raise ValueError("Unexpected Real/Fake labels")
    probabilities = frame.fake_probability.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid scores")
    real_frame = frame[frame.method.eq("original")]
    real = real_frame.fake_probability.to_numpy()
    rows = []
    for method, group in frame[~frame.method.eq("original")].groupby("method"):
        fake = group.fake_probability.to_numpy()
        row = dict(selection=selection, training_seed=seed, strategy=strategy, method=method,
            role="ffpp" if method in {"Deepfakes", "Face2Face"} else "seen" if method in selected["seen"] else "unseen",
            auc=auc(real, fake), tpr_fixed05=float(group.predicted_class.eq(1).mean()),
            score_mean=float(fake.mean()), score_median=float(np.median(fake)))
        for cap, name in ((0.01, "1"), (0.05, "5")):
            tpr, fpr, cutoff = operating_point(real, fake, cap)
            row[f"tpr_at_fpr{name}"] = tpr
            row[f"achieved_fpr{name}"] = fpr
            row[f"real_cutoff{name}"] = cutoff
        rows.append(row)
    methods = pd.DataFrame(rows)
    unseen = methods[methods.role.eq("unseen")]
    if len(unseen) != 12:
        raise ValueError("Expected selection-common unseen12")
    model = dict(selection=selection, training_seed=seed, strategy=strategy,
        unseen_auc=float(unseen.auc.mean()), unseen_tpr_fpr1=float(unseen.tpr_at_fpr1.mean()),
        unseen_tpr_fpr5=float(unseen.tpr_at_fpr5.mean()), unseen_tpr_fixed05=float(unseen.tpr_fixed05.mean()),
        real_fpr_fixed05=float(real_frame.predicted_class.eq(1).mean()), real_score_mean=float(real.mean()))
    for name in ("1", "5"):
        model[f"achieved_fpr{name}"] = float(methods[f"achieved_fpr{name}"].iloc[0])
        model[f"real_cutoff{name}"] = float(methods[f"real_cutoff{name}"].iloc[0])
    return model, methods


def summarize_weights(log, best_epoch, selection, seed):
    log = log[log.epoch_position.lt(best_epoch)].copy()
    if log.empty or not log.groupby("refresh").size().eq(6).all():
        raise ValueError("Expected six weights per pre-best refresh")
    common = dict(selection=selection, training_seed=seed, best_epoch=best_epoch)
    active = log.groupby("refresh").loss_weight.apply(lambda v: (v - 1).abs().gt(1e-6).any())
    summary = dict(**common, refresh_count=len(active), nonuniform_refresh_fraction=float(active.mean()),
        mean_absolute_weight_change=float((log.loss_weight - 1).abs().mean()),
        positive_gain_fraction=float(log.other_five_relative_gain_vs_uniform.gt(0).mean()),
        positive_utility_fraction=float(log.utility.gt(0).mean()),
        above_margin_fraction=float(log.ema_utility.gt(0.001).mean()),
        real_penalty_veto_fraction=float((log.other_five_relative_gain_vs_uniform.gt(0) & log.utility.le(0)).mean()),
        mean_gain=float(log.other_five_relative_gain_vs_uniform.mean()),
        mean_penalty=float(log.excess_real_penalty.mean()))
    weights = log.groupby("method").agg(mean_weight=("loss_weight", "mean"),
        min_weight=("loss_weight", "min"), max_weight=("loss_weight", "max"),
        mean_gain=("other_five_relative_gain_vs_uniform", "mean"),
        mean_penalty=("excess_real_penalty", "mean"), mean_utility=("utility", "mean"),
        mean_ema=("ema_utility", "mean")).reset_index()
    for key, value in common.items():
        weights[key] = value
    return summary, weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-roots", type=Path, nargs="+", required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--selections", nargs="+", default=[f"S{i}" for i in range(1, 7)])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/family_rotation_v1/selections.json")
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Choose an empty diagnostic output directory")
    cfg = json.loads(a.config.read_text())
    summaries = [(root, pd.read_csv(root/"model_summary.csv")) for root in a.evaluation_roots]
    models, methods, logs, weights, shifts, provenance = [], [], [], [], [], []
    for selection in a.selections:
        selected = dict(all=[m for f in cfg["families"].values() for m in f.values()],
            seen=[cfg["families"][f][l] for f in ("FS", "FR", "EFS") for l in cfg["selections"][selection]])
        for seed in a.seeds:
            pair = {}
            for strategy in ("uniform", "complementarity"):
                candidates = []
                for root, summary in summaries:
                    part = summary[(summary.selection == selection) & (summary.training_seed == seed) & (summary.strategy == strategy)]
                    path = root/selection.lower()/strategy/"q95/predictions.csv"
                    if len(part) == 1 and path.is_file():
                        candidates.append((root, path, part.iloc[0]))
                if not candidates:
                    raise FileNotFoundError(f"{selection}/seed{seed}/{strategy}: no saved predictions")
                root, path, reference = candidates[0]
                run = a.runs_root/f"xception_{selection.lower()}_m7_q95_{strategy}_adaptive_v1_seed{seed}"
                saved = json.loads((run/"config.json").read_text())
                identity = json.loads(path.with_name("metrics.json").read_text())["identity"]
                adaptive = saved["adaptive_learning"]
                if (saved["seed"] != seed or adaptive["strategy"] != strategy
                        or identity["manifest_sha256"] != saved["manifest_sha256"]
                        or identity["roles_sha256"] != adaptive["validation_roles_sha256"]
                        or identity["preprocessing"] != "fixed_q95"):
                    raise ValueError(f"Run/evaluation identity mismatch: {path}")
                frame = pd.read_csv(path).sort_values("sample_id").reset_index(drop=True)
                model, method = analyze_predictions(frame, selection, seed, strategy, selected)
                if abs(model["unseen_auc"] - reference.df40_unseen_macro_auc) > 1e-6:
                    raise ValueError(f"Saved summary/prediction AUC differs: {path}")
                models.append(model); methods.append(method); pair[strategy] = frame
                provenance.append(dict(selection=selection, training_seed=seed, strategy=strategy,
                    predictions=str(path), run=str(run), checkpoint_sha256=identity["checkpoint_sha256"]))
                if strategy == "complementarity":
                    compute = pd.read_csv(root/"compute_summary.csv")
                    compute = compute[(compute.selection == selection) & (compute.strategy == strategy)]
                    if len(compute) != 1:
                        raise ValueError("Missing/duplicate best epoch in compute summary")
                    summary, weight = summarize_weights(pd.read_csv(run/"adaptive_weight_history.csv"),
                        int(compute.iloc[0].best_epoch), selection, seed)
                    logs.append(summary); weights.append(weight)
                print(f"Analyzed {selection}/{seed}/{strategy}", flush=True)
            u, c = pair["uniform"], pair["complementarity"]
            if not u[["sample_id", "method", "label"]].equals(c[["sample_id", "method", "label"]]):
                raise ValueError("Paired sample identities differ")
            paired = u[["method", "fake_probability"]].copy()
            paired["delta"] = c.fake_probability - u.fake_probability
            for method, group in paired.groupby("method"):
                shifts.append(dict(selection=selection, training_seed=seed, method=method,
                    mean_delta=float(group.delta.mean()), median_delta=float(group.delta.median()),
                    score_increased_fraction=float(group.delta.gt(0).mean())))
    model = pd.DataFrame(models)
    method = pd.concat(methods, ignore_index=True)
    comparison = model.pivot(index=["selection", "training_seed"], columns="strategy",
        values=["unseen_auc", "unseen_tpr_fpr1", "unseen_tpr_fpr5"])
    comparison.columns = ["_".join(c) for c in comparison.columns]
    comparison = comparison.reset_index()
    for metric in ("unseen_auc", "unseen_tpr_fpr1", "unseen_tpr_fpr5"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_complementarity"] - comparison[f"{metric}_uniform"]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in dict(model_summary=model, method_summary=method, matched_fpr_comparison=comparison,
            weight_log_summary=pd.DataFrame(logs), method_weight_summary=pd.concat(weights, ignore_index=True),
            paired_score_shift=pd.DataFrame(shifts), source_files=pd.DataFrame(provenance)).items():
        table.to_csv(a.output_dir/f"{name}.csv", index=False)
    (a.output_dir/"diagnostic_protocol.json").write_text(json.dumps(dict(
        mode="diagnostic_only_saved_predictions", selections=a.selections, seeds=a.seeds,
        fpr_caps=[0.01, 0.05], cutoff_rule="fake_score > boundary_real_score; empirical FPR <= cap",
        threshold_role="test ROC description only; not deployment or model selection",
        no_retraining=True, weight_log_scope="refreshes before best epoch end",
        utility_interpretation="temporary seen-meta update; not observed final validation improvement",
        package_versions=dict(numpy=np.__version__, pandas=pd.__version__)), indent=2)+"\n")
    print(comparison.to_string(index=False), flush=True)
    print("Saved:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
