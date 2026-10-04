#!/usr/bin/env python3
"""Bounded CPU smoke check for augmented adaptive training, resume and notebooks."""

import ast
import contextlib
import copy
import io
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from family_rotation_adaptive_augmentation import (
    CONFIG, ROOT, adaptive_config_path, check_training_arguments, load_experiment, validate_checkpoint_config,
)
from family_rotation_adaptive_learning import AdaptiveController, ProbeDataset
from diagnose_xception_family_rotation_corruptions import corrupt
from evaluate_checkpoint import evaluate_with_predictions
from evaluate_xception_family_rotation_adaptive_augmentation import write_summaries
from train_xception_family_rotation_adaptive_augmentation import command_for
from train_xception_letterbox import make_loaders, parse_args
from train_baseline import set_seed
from xception_preprocessing import Letterbox, RandomGaussianNoise, RandomJpegRoundTrip


def main():
    torch.set_num_threads(1)
    settings = load_experiment()
    protocol = json.loads(adaptive_config_path(settings).read_text())
    with tempfile.TemporaryDirectory(prefix="adaptive_aug_check_") as folder:
        root = Path(folder)
        args = SimpleNamespace(output_root=root, data_root=root, protocol_config=CONFIG, seed=42,
            epochs=15, batch_size=16, workers=0, learning_rate=1e-4, weight_decay=1e-4,
            patience=3, log_interval_seconds=60, verbose=False)
        parsed = []
        for augmentation in settings["augmentations"]:
            for strategy in settings["strategies"]:
                _, command = command_for(args, settings, "S2", strategy, augmentation, root/"manifest.csv", root/"roles.csv")
                with contextlib.redirect_stdout(io.StringIO()):
                    old = sys.argv
                    try:
                        sys.argv = command[2:]
                        value = parse_args()
                    finally:
                        sys.argv = old
                assert check_training_arguments(value, settings) == augmentation
                assert value.quiet and value.train_noise_probability == (0.25 if augmentation.endswith("noise") else 0)
                parsed.append(value)
        methods = ["original", "A", "B", "C", "D", "E", "F"]
        rows = []
        for split in ("train", "val"):
            for index, method in enumerate(methods):
                path = f"{split}_{method}.png"
                image = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
                if method == "F":
                    image = np.tile(image, (32, 32, 1))  # Native1024 EFS is valid.
                Image.fromarray(image).save(root/path)
                rows.append({"sample_id": f"{split}-{method}", "split": split, "label": "real" if index == 0 else "fake",
                    "method": method, "source_path": path, "validation_role": "meta",
                    "video_id": f"{split}-{index}", "group_id": f"{split}-{index}"})
        frame = pd.DataFrame(rows)
        train, val = [frame[frame.split.eq(s)].copy() for s in ("train", "val")]
        data = {"input_size": (3, 299, 299), "interpolation": "bicubic", "mean": (0.5,)*3, "std": (0.5,)*3}
        validation_tensors = []
        for value in parsed:
            loader, _, training_transform, validation_transform = make_loaders(train, val, root, data, value)
            ops = training_transform.transforms
            assert isinstance(ops[1], Letterbox)
            if value.train_noise_probability:
                assert isinstance(ops[2], RandomGaussianNoise) and ops[2].max_std == 5
                assert isinstance(ops[3], RandomJpegRoundTrip)
            else:
                assert isinstance(ops[2], RandomJpegRoundTrip)
            batch = next(iter(loader))
            assert tuple(batch["image"].shape[1:]) == (3, 299, 299)
            with Image.open(root/"val_F.png") as image:
                validation_tensors.append(validation_transform(image.convert("RGB")))
        assert all(torch.equal(validation_tensors[0], tensor) for tensor in validation_tensors)
        print("PASS: four CLI variants, augmentation order, native1024 and identical Q95 validation")

        source = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
        noise = lambda std: np.asarray(corrupt(source, {"kind": "gaussian_noise", "strength": std}, "sample", 20261005))
        assert np.array_equal(noise(10), noise(10))
        for strength in (2, 5, 10):
            observed = (noise(strength).astype(float) - 128).std()
            assert abs(observed - strength) < .1
        assert np.array_equal(np.asarray(source), np.full((256, 256, 3), 128))
        print("PASS: deterministic paired noise2/5/10, uint8 units, source unchanged")

        small_protocol = copy.deepcopy(protocol)
        for key in ("probe_real", "probe_fake_per_method", "update_real", "update_fake_per_method"):
            small_protocol[key] = 1
        small_transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])

        class TrainingDataset(ProbeDataset):
            def __getitem__(self, index):
                image, label = super().__getitem__(index)
                return {"image": image, "label": torch.tensor(label), "method": self.frame.iloc[index].method}

        def build(output, seed):
            output.mkdir()
            set_seed(seed)
            model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3),
                torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            loader = DataLoader(TrainingDataset(train, root, small_transform), batch_size=2, shuffle=True,
                generator=torch.Generator().manual_seed(seed))
            controller = AdaptiveController("complementarity", small_protocol, train, val, root,
                small_transform, output, workers=0, quiet=True)
            return model, optimizer, loader, controller

        model, optimizer, loader, controller = build(root/"first", 42)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            controller.train_epoch(model, loader, optimizer, scaler, torch.device("cpu"), 1)
            state = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "controller": controller.state_dict(),
                "generator": loader.generator.get_state(), "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()})
            controller.train_epoch(model, loader, optimizer, scaler, torch.device("cpu"), 2)
            resumed, opt, resumed_loader, restored = build(root/"resume", 999)
            resumed.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            restored.load_state_dict(state["controller"])
            resumed_loader.generator.set_state(state["generator"])
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            restored.train_epoch(resumed, resumed_loader, opt, scaler, torch.device("cpu"), 2)
        assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in model.state_dict().items())
        assert controller.weights == restored.weights and controller.refreshes == restored.refreshes
        print("PASS: quiet complementarity train, BN-safe probes and exact epoch-boundary CPU resume")

        class TestDataset(TrainingDataset):
            def __getitem__(self, index):
                return {**super().__getitem__(index), "index": index}

        metrics, labels, predictions, scores = evaluate_with_predictions(model,
            DataLoader(TestDataset(val, root, small_transform), batch_size=2), val,
            torch.nn.CrossEntropyLoss(), torch.device("cpu"), progress_enabled=False)
        assert len(scores) == len(val) and np.isfinite(metrics["roc_auc"])
        print("PASS: quiet ordered CPU evaluation")

        older = {"seed": 42, "epochs": 15, "resume": None, "quiet": False}
        saved = {**older, "resume": str(root/"last.pt"), "quiet": True}
        validate_checkpoint_config(older, saved)
        try:
            validate_checkpoint_config(older, {**saved, "epochs": 16})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Scientific checkpoint mismatch was accepted")
        print("PASS: resumed run may keep older best.pt, scientific mismatches still rejected")

        summary_rows = []
        for augmentation in settings["augmentations"]:
            for strategy in settings["strategies"]:
                for condition in settings["evaluation_conditions"]:
                    summary_rows.append({"selection": "S2", "augmentation": augmentation, "strategy": strategy,
                        "training_seed": 42, "evaluation_condition": condition["name"],
                        "jpeg_quality": condition["jpeg_quality"], "noise_std": condition["noise_std"],
                        "df40_unseen_macro_auc": .8 + .01*(strategy == "complementarity"),
                        "real_fpr": .01, "df40_seen_macro_auc": .99, "ffpp_reference_macro_auc": .9})
        identity = {"selections": ["S2"], "strategies": settings["strategies"],
                    "augmentations": settings["augmentations"], "conditions": settings["evaluation_conditions"]}
        output = root/"summary"
        output.mkdir()
        compute = [{"selection": "S2", "augmentation": "mixed_jpeg", "strategy": "uniform", "training_seed": 42}]
        write_summaries(output, summary_rows, [], compute, identity)
        write_summaries(output, summary_rows, [], [], identity)
        comparison = pd.read_csv(output/"baseline_comparison.csv")
        assert len(comparison) == 16 and np.allclose(comparison.delta_df40_unseen_macro_auc, .01)
        assert json.loads((output/"evaluation_summary.json").read_text())["complete"]
        assert len(pd.read_csv(output/"compute_summary.csv")) == 1
        print("PASS: matched baseline comparisons, completion counts and compute reuse")

    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    for path in sorted((ROOT/"docs/colab/method_pilots").glob("train_evaluate_m7_jpeg_noise_*_seed42.ipynb")):
        notebook = json.loads(path.read_text())
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert len(code) == 7
        for cell in code:
            ast.parse(shell.input_transformer_manager.transform_cell("".join(cell["source"])))
    print("PASS: both seven-cell Colab notebooks, including IPython shell syntax")


if __name__ == "__main__":
    main()
