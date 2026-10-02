"""Online method loss weighting with held-out, source-disjoint validation probes."""

import contextlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diagnose_xception_family_rotation_joint_updates import mean_gradient
from prepare_family_rotation_adaptive_learning import atomic_csv


@contextlib.contextmanager
def preserve_rng():
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def ranked_subset(frame, count, seed, tag):
    import hashlib
    rank = frame.sample_id.map(lambda s: hashlib.sha256(f"{seed}|{tag}|{s}".encode()).hexdigest())
    if len(frame) < count:
        raise ValueError(f"{tag}: need {count}, found {len(frame)}")
    return frame.assign(_rank=rank).sort_values(["_rank", "sample_id"]).head(count).drop(columns="_rank").copy()


def safe_weights(scores, protocol, strategy):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite adaptive utility")
    if strategy == "complementarity":
        scores = np.maximum(scores - protocol["minimum_complementarity_gain"], 0)
        if scores.max() == 0:
            return np.ones(len(scores))
    logits = np.maximum((scores - scores.max()) / protocol["softmax_temperature"], -50)
    mass = np.exp(logits)
    mass /= mass.sum()
    # Capped-simplex allocation; all methods retain the same uniform floor.
    floor, cap = protocol["uniform_floor"], protocol["maximum_method_weight"]
    n = len(scores)
    result = np.full(n, floor)
    capacity = np.full(n, cap - floor)
    remaining = n * (1 - floor)
    active = np.ones(n, dtype=bool)
    while remaining > 1e-12:
        distribution = mass[active] / mass[active].sum()
        allocation = remaining * distribution
        indices = np.flatnonzero(active)
        saturated = allocation >= capacity[indices]
        if not saturated.any():
            result[indices] += allocation
            break
        chosen = indices[saturated]
        result[chosen] += capacity[chosen]
        remaining -= capacity[chosen].sum()
        active[chosen] = False
    if not np.isclose(result.mean(), 1) or (result < floor - 1e-9).any() or (result > cap + 1e-9).any():
        raise RuntimeError("Invalid bounded loss weights")
    return result


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, frame, root, transform):
        self.frame = frame.reset_index(drop=True)
        self.root, self.transform = root, transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from PIL import Image
        row = self.frame.iloc[index]
        with Image.open(self.root / row.source_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, 0 if row.label == "real" else 1


class AdaptiveController:
    def __init__(self, strategy, protocol, train, roles, root, transform, output, batch_size=32, workers=2):
        self.strategy, self.protocol = strategy, protocol
        self.methods = sorted(train.loc[train.label == "fake", "method"].unique())
        if len(self.methods) != 6 or train.loc[train.label == "fake"].groupby("method").size().nunique() != 1:
            raise ValueError("Adaptive M7 requires six equally sized fake methods")
        self.train, self.root, self.transform, self.output = train, root, transform, output
        self.batch_size, self.workers = batch_size, workers
        meta = roles[roles.validation_role == "meta"]
        self.probe = pd.concat([ranked_subset(meta[meta.method == method],
            protocol["probe_real"] if method == "original" else protocol["probe_fake_per_method"],
            protocol["role_seed"], f"probe/{method}") for method in ["original", *self.methods]], ignore_index=True)
        path = output / "adaptive_probe_samples.csv"
        if path.exists() and not pd.read_csv(path, dtype=str, keep_default_na=False).equals(self.probe.astype(str).reset_index(drop=True)):
            raise RuntimeError("Existing adaptive probe sample IDs differ")
        if not path.exists():
            atomic_csv(path, self.probe)
        self.weights = {method: 1.0 for method in self.methods}
        self.ema = np.zeros(6)
        self.refreshes = 0
        self.next_fraction = protocol["warmup_epoch_fraction"]
        self.log = []
        self.seconds = 0.0
        self.extra_forward_images = 0
        self.extra_backward_images = 0

    def state_dict(self):
        return {"weights": self.weights, "ema": self.ema.tolist(), "refreshes": self.refreshes,
                "next_fraction": self.next_fraction, "log": self.log, "seconds": self.seconds,
                "extra_forward_images": self.extra_forward_images, "extra_backward_images": self.extra_backward_images}

    def load_state_dict(self, state):
        self.weights = state["weights"]
        self.ema = np.asarray(state["ema"], dtype=float)
        self.refreshes = state["refreshes"]
        self.next_fraction = state["next_fraction"]
        self.log = state["log"]
        self.seconds = state["seconds"]
        self.extra_forward_images = state["extra_forward_images"]
        self.extra_backward_images = state["extra_backward_images"]
        if set(self.weights) != set(self.methods):
            raise RuntimeError("Resumed adaptive method membership differs")
        if self.log:
            atomic_csv(self.output / "adaptive_weight_history.csv", pd.DataFrame(self.log))

    def loader(self, frame, batch_size=None):
        return DataLoader(ProbeDataset(frame, self.root, self.transform),
            batch_size=batch_size or self.batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=False,
            generator=torch.Generator().manual_seed(self.protocol["role_seed"]))

    @torch.no_grad()
    def losses(self, model, device, label):
        model.eval()
        losses = []
        for images, targets in tqdm(self.loader(self.probe), desc=label, leave=False):
            logits = model(images.to(device))
            loss = F.cross_entropy(logits.float(), targets.to(device), reduction="none")
            losses.extend(loss.cpu().tolist())
        self.extra_forward_images += len(self.probe)
        values = np.asarray(losses)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite adaptive probe CE")
        return {method: float(values[self.probe.method.eq(method).to_numpy()].mean())
                for method in ["original", *self.methods]}

    def refresh(self, model, device, fraction):
        if self.strategy == "uniform":
            return
        start = time.monotonic()
        print(f"\nAdaptive refresh {self.refreshes + 1}: {self.strategy}, epoch position {fraction:.3f}", flush=True)
        training_modes = {module: module.training for module in model.modules()}
        with preserve_rng():
            try:
                base = self.losses(model, device, "Adaptive meta-validation")
                floor = self.protocol["loss_denominator_floor"]
                if self.strategy == "difficulty":
                    fake = np.array([base[m] for m in self.methods])
                    scores = fake / max(float(fake.mean()), floor)
                    branches = {}
                else:
                    scores, branches = self.complementarity(model, device, base)
                decay = self.protocol["ema_decay"]
                self.ema = scores if self.refreshes == 0 else decay * self.ema + (1 - decay) * scores
                values = safe_weights(self.ema, self.protocol, self.strategy)
                self.weights = dict(zip(self.methods, values.tolist()))
                for i, method in enumerate(self.methods):
                    self.log.append({"refresh": self.refreshes + 1, "epoch_position": fraction,
                        "method": method, "utility": float(scores[i]), "ema_utility": float(self.ema[i]),
                        "loss_weight": self.weights[method], "probe_fake_ce": base[method],
                        "probe_real_ce": base["original"], **branches.get(method, {})})
                self.refreshes += 1
                atomic_csv(self.output / "adaptive_weight_history.csv", pd.DataFrame(self.log))
                print("Loss weights:", {m: round(w, 3) for m, w in self.weights.items()}, flush=True)
            finally:
                for module, mode in training_modes.items():
                    module.training = mode
                model.zero_grad(set_to_none=True)
                self.seconds += time.monotonic() - start

    def complementarity(self, model, device, base):
        p = self.protocol
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        original = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
        # mean_gradient uses FP32 and eval mode, keeping dropout and BN buffers frozen.
        buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        gradients = {}
        for method in ["original", *self.methods]:
            part = ranked_subset(self.train[self.train.method == method],
                p["update_real"] if method == "original" else p["update_fake_per_method"],
                p["role_seed"], f"refresh{self.refreshes}/{method}")
            gradients[method] = mean_gradient(model, self.loader(part, 8), device, f"Direction {method}")
            self.extra_forward_images += len(part)
            self.extra_backward_images += len(part)
        real = gradients.pop("original")
        uniform_fake = torch.stack(list(gradients.values())).mean(0)

        def probe(fake, label):
            gradient = 0.5 * (real + fake)
            step = p["temporary_sgd_learning_rate"] * gradient.to(device)
            relative = float(step.double().norm() / original.double().norm().clamp_min(1e-12))
            if not torch.isfinite(step).all() or relative > p["maximum_relative_temporary_update"]:
                raise FloatingPointError(f"Unsafe temporary adaptive step: {relative}")
            try:
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters(original - step, parameters)
                return self.losses(model, device, label), relative
            finally:
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters(original.clone(), parameters)

        try:
            uniform, _ = probe(uniform_fake, "Probe uniform-all-six")
            scores, details = [], {}
            for method in self.methods:
                after, norm = probe(gradients[method], f"Probe source {method}")
                gains, target_gains = [], {}
                for target in self.methods:
                    if target == method:
                        continue
                    denominator = max(0.5 * (base["original"] + base[target]), p["loss_denominator_floor"])
                    gain = 0.5 * (uniform["original"] + uniform[target] - after["original"] - after[target]) / denominator
                    gains.append(gain)
                    target_gains[target] = gain
                penalty = max(after["original"] - uniform["original"], 0) / max(base["original"], p["loss_denominator_floor"])
                utility = float(np.mean(gains)) - p["real_penalty"] * penalty
                scores.append(utility)
                details[method] = {"other_five_relative_gain_vs_uniform": float(np.mean(gains)),
                    "excess_real_penalty": penalty, "temporary_relative_update_norm": norm,
                    "source_fake_ce_after": after[method], "real_ce_after": after["original"],
                    "uniform_real_ce_after": uniform["original"],
                    "target_relative_gain_vs_uniform_json": json.dumps(target_gains, sort_keys=True)}
            return np.asarray(scores), details
        finally:
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(original, parameters)
                for name, buffer in model.named_buffers():
                    if not torch.equal(buffer, buffers[name]):
                        raise RuntimeError(f"Temporary adaptive update modified BN buffer: {name}")

    def train_epoch(self, model, loader, optimizer, scaler, device, epoch):
        model.train()
        total_loss = total_correct = total_samples = 0
        progress = tqdm(loader, desc="Train", leave=True, dynamic_ncols=True)
        for index, batch in enumerate(progress):
            fraction = epoch - 1 + index / len(loader)
            if self.strategy != "uniform" and fraction + 1e-12 >= self.next_fraction:
                self.refresh(model, device, fraction)
                self.next_fraction += self.protocol["refresh_epoch_fraction"]
            images, labels = batch["image"].to(device), batch["label"].to(device)
            weight = torch.tensor([self.weights.get(method, 1) for method in batch["method"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                losses = F.cross_entropy(logits, labels, reduction="none")
                # Mean=1 weights preserve expected Real/Fake objective scale at equal quotas.
                loss = (losses * weight).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weighted training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            size = labels.size(0)
            total_loss += float(loss.detach()) * size
            total_correct += int((logits.argmax(1) == labels).sum())
            total_samples += size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}
