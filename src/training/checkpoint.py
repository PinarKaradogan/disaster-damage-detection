from pathlib import Path
import json
import random

import numpy as np
import torch

from src.training.artifacts import atomic_output


class EarlyStopping:
    """Strictly higher validation macro F1 improves; ties consume patience."""

    def __init__(self, patience=5):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score):
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError("Validation macro F1 must be finite")
        improved = score > self.best
        if improved:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def save_checkpoint(path, model, *, epoch, best_val_macro_f1, config, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state_dict=model.state_dict(), epoch=epoch,
                   best_val_macro_f1=best_val_macro_f1, config=config,
                   model_name=config["model_name"])
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    atomic_save(path, payload)


def atomic_save(path, payload):
    with atomic_output(path, binary=True) as handle:
        torch.save(payload, handle)


def capture_rng(train_loader, val_loader):
    numpy = np.random.get_state()
    generators = {}
    for name, loader in (("train", train_loader), ("val", val_loader)):
        for kind, source in (("loader", loader), ("sampler", getattr(loader, "sampler", None))):
            generator = getattr(source, "generator", None)
            if generator is not None:
                generators[f"{name}_{kind}"] = generator.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": generators}


def restore_rng(state, train_loader, val_loader):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, loader in (("train", train_loader), ("val", val_loader)):
        for kind, source in (("loader", loader), ("sampler", getattr(loader, "sampler", None))):
            key = f"{name}_{kind}"
            generator = getattr(source, "generator", None)
            if (key in state["generators"]) != (generator is not None):
                raise ValueError(f"Resume generator mismatch: {key}")
            if generator is not None:
                generator.set_state(state["generators"][key])


def normalize_training_metadata(metadata):
    """Missing mode in historical metadata means full, never head_only."""
    normalized = dict(metadata)
    normalized.setdefault("training_mode", "full")
    if normalized["training_mode"] == "full" and "parameter_count" in normalized:
        normalized.setdefault("trainable_parameter_count", normalized["parameter_count"])
    return normalized


def read_resume_checkpoint(path, metadata):
    """Never fall back to unsafe pickle or infer missing legacy training state."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Cannot safely load resume checkpoint: {path}") from exc
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "epoch", "best_val_macro_f1", "early_stopping",
                "rng", "config", "history", "best_checkpoint", "best_metrics",
                "training_time_seconds", "finished"}
    if not isinstance(payload, dict) or payload.get("resume_schema_version") != 1 or not required <= payload.keys():
        raise ValueError("Checkpoint lacks complete resume state (legacy/invalid checkpoint)")
    # JSON canonicalization makes tuples from torch and lists from config.json equivalent.
    checkpoint_metadata = normalize_training_metadata(payload["config"])
    metadata = normalize_training_metadata(metadata)
    if json.dumps(checkpoint_metadata, sort_keys=True) != json.dumps(metadata, sort_keys=True):
        changed = sorted(key for key in set(checkpoint_metadata) | set(metadata)
                         if json.dumps(checkpoint_metadata.get(key), sort_keys=True)
                         != json.dumps(metadata.get(key), sort_keys=True))
        raise ValueError(f"Resume configuration mismatch: {', '.join(changed)}")
    epoch = payload["epoch"]
    if not isinstance(epoch, int) or not 1 <= epoch <= metadata["epochs"]:
        raise ValueError("Invalid completed epoch in resume checkpoint")
    if [row["epoch"] for row in payload["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Resume checkpoint history is not contiguous")
    best = payload["best_metrics"]
    stopping = payload["early_stopping"]
    if (stopping["best"] != payload["best_val_macro_f1"]
            or stopping["best"] != best["best_val_macro_f1"]
            or not 1 <= best["best_epoch"] <= epoch
            or payload["best_checkpoint"]["epoch"] != best["best_epoch"]
            or payload["best_checkpoint"]["best_val_macro_f1"] != stopping["best"]
            or stopping["bad_epochs"] != epoch - best["best_epoch"]
            or payload["scheduler_state_dict"].get("T_max") != metadata["epochs"]
            or payload["scheduler_state_dict"].get("last_epoch") != epoch):
        raise ValueError("Inconsistent resume checkpoint training state")
    if payload["finished"] or epoch >= metadata["epochs"]:
        raise ValueError("Experiment already completed; resume refused")
    if stopping["bad_epochs"] >= metadata["early_stopping_patience"]:
        raise ValueError("Experiment already reached early stopping; resume refused")
    return payload


def load_checkpoint(path, model, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return payload
