#!/usr/bin/env python3
"""Aggregate S1-S6 transfer matrices into bounded per-selection loss weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_baseline import write_json


CONDITIONS = (
    "canonical_256_jpeg_q95",
    "canonical_256_jpeg_q90",
    "canonical_256_jpeg_q75",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--uniform-floor", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.uniform_floor <= 1 or args.temperature <= 0:
        raise ValueError("Invalid weighting parameters")
    records = []
    selection_methods = {}
    for number in range(1, 7):
        selection = f"S{number}"
        candidates = sorted(args.evaluation_root.glob(f"s{number}*_q95_q90_q75"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one evaluation directory for {selection}: {candidates}")
        root = candidates[0]
        matrices = []
        for condition in CONDITIONS:
            path = root / f"transfer_matrix_{condition}.csv"
            matrix = pd.read_csv(path, index_col=0)
            if set(matrix.index) != set(matrix.columns):
                raise RuntimeError(f"Non-square method matrix: {path}")
            matrices.append(matrix)
        methods = list(matrices[0].index)
        selection_methods[selection] = methods
        for source in methods:
            values = []
            for matrix in matrices:
                values.extend(float(matrix.loc[source, target]) for target in methods if target != source)
            records.append({
                "selection": selection,
                "method": source,
                "transfer_utility": float(np.mean(values)),
                "worst_transfer_auc": float(np.min(values)),
                "evaluation_dir": str(root),
            })
    table = pd.DataFrame(records)
    global_utility = table.groupby("method")["transfer_utility"].mean().to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for selection, methods in selection_methods.items():
        utility = np.asarray([global_utility[m] for m in methods], dtype=float)
        logits = (utility - utility.mean()) / args.temperature
        soft = np.exp(logits - logits.max())
        soft = len(methods) * soft / soft.sum()
        weights = args.uniform_floor + (1.0 - args.uniform_floor) * soft
        weights = weights / weights.mean()
        payload = {
            "protocol": "family_rotation_transfer_loss_weighting_v1",
            "selection": selection,
            "derivation": "mean off-diagonal AUC over Q95/Q90/Q75 and all available rotations",
            "uniform_floor": args.uniform_floor,
            "temperature": args.temperature,
            "weights": {m: float(w) for m, w in zip(methods, weights)},
            "utilities": {m: float(u) for m, u in zip(methods, utility)},
        }
        write_json(args.output_dir / f"{selection.lower()}_weights.json", payload)
        outputs.extend({"selection": selection, "method": m, "utility": u, "loss_weight": w}
                       for m, u, w in zip(methods, utility, weights))
    table.to_csv(args.output_dir / "transfer_utility_observations.csv", index=False)
    pd.DataFrame(outputs).to_csv(args.output_dir / "selection_loss_weights.csv", index=False)
    print(pd.DataFrame(outputs).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
