"""Versioned metadata and run orchestration. No test-split access."""
import platform
import json
import sys
from pathlib import Path

import timm
import torch

from src.data.constants import LABEL_MAP
from src.data.dataset import ROOT
from src.models import create_model
from src.training.artifacts import file_sha256, write_json
from src.training.data import IMAGENET_MEAN, IMAGENET_STD, build_loaders
from src.training.engine import fit
from src.training.checkpoint import normalize_training_metadata, read_resume_checkpoint
from src.training.losses import make_criterion
from src.training.utils import get_device, set_training_seed


def run_experiment(config, *, manifest_path, experiment_name, loss="weighted_ce", sampler="none",
                   smoke=False, results_root=ROOT / "results", runs_root=ROOT / "runs",
                   project_root=ROOT, device=None, pretrained=True, resume=False, session_epochs=None):
    if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in {".", ".."}:
        raise ValueError("experiment_name must be a single directory name")
    if loss not in {"ce", "weighted_ce", "focal"} or sampler not in {"none", "weighted"}:
        raise ValueError("Unknown loss or sampler")
    if sampler == "weighted" and loss != "ce":
        raise ValueError("Weighted sampler requires standard CE")
    if min(config.batch_size, config.smoke_batch_size, config.epochs) < 1 or config.num_workers < 0:
        raise ValueError("Batch size/epochs must be positive; workers must be nonnegative")
    if session_epochs is not None and session_epochs < 1:
        raise ValueError("session_epochs must be positive")
    output_dir, log_dir = Path(results_root) / experiment_name, Path(runs_root) / experiment_name
    stored_metadata = None
    if resume:
        if not (output_dir / "last.pth").is_file():
            raise ValueError("Resume requires a complete last.pth; legacy checkpoints are unsupported")
        stored_metadata = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") not in {"paused", "failed", "running"}:
            raise ValueError("Experiment already completed or invalid status; resume refused")
    else:
        for directory in [output_dir, log_dir]:
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing experiment: {directory}")
    set_training_seed(config.seed)
    device = get_device() if device is None else torch.device(device)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = Path(project_root) / manifest_path
    data = build_loaders(config, manifest_path, sampler=sampler, smoke=smoke, project_root=project_root)
    model = create_model(config.model_name, num_classes=config.num_classes,
                         pretrained=pretrained and not resume, training_mode=config.training_mode)
    criterion = make_criterion(loss, sampler, data.class_weights)
    actual_epochs = min(2, config.epochs) if smoke else config.epochs
    metadata = {
        **config.to_dict(), "schema_version": 1, "model": config.model_name,
        "experiment_name": experiment_name, "evaluation_split": "val", "smoke": smoke,
        "batch_size": config.smoke_batch_size if smoke else config.batch_size,
        "epochs": actual_epochs, "requested_epochs": config.epochs,
        "loss": loss, "focal_gamma": 2.0 if loss == "focal" else None, "sampler": sampler,
        "class_weights": data.class_weights.tolist() if loss == "weighted_ce" or sampler == "weighted" else None,
        "class_weight_formula": "inverse_train_frequency / mean(inverse_train_frequency)",
        "train_class_counts": data.train_class_counts, "split_counts": data.split_counts,
        "label_map": LABEL_MAP, "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "augmentation": {"train": {"horizontal_flip_p": 0.5, "vertical_flip_p": 0.5,
                                    "rotation_degrees": 15}, "val": None},
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "scheduler_t_max": actual_epochs, "amp_requested": config.amp,
        "amp": config.amp and device.type == "cuda", "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pretrained": pretrained if config.model_name == "resnet34_early_fusion" else False,
        "pretrained_adaptation": "RGB weights repeated twice, scaled by 0.5" if config.model_name == "resnet34_early_fusion" and pretrained else None,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad),
        "samples_per_epoch": {"train": len(data.train.dataset), "val": len(data.val.dataset)},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "timm": timm.__version__},
    }
    if config.model_name == "resnet34_early_fusion":
        metadata["pretrained_source"] = (stored_metadata["pretrained_source"] if resume
                                         else model.backbone.pretrained_cfg)
    resume_state = None
    if resume:
        resume_state = read_resume_checkpoint(output_dir / "last.pth", metadata)
        if (json.dumps(normalize_training_metadata(stored_metadata), sort_keys=True)
                != json.dumps(normalize_training_metadata(resume_state["config"]), sort_keys=True)):
            raise ValueError("Resume config.json differs from checkpoint configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(output_dir / "config.json", metadata)
    write_json(output_dir / "status.json", {"status": "running"})
    print(f"parameters total={metadata['parameter_count']} "
          f"trainable={metadata['trainable_parameter_count']} training_mode={config.training_mode}", file=sys.stderr, flush=True)
    try:
        metrics, history = fit(model, data.train, data.val, criterion, config, device,
                               output_dir=output_dir, log_dir=log_dir, metadata=metadata,
                               epochs=actual_epochs, session_epochs=session_epochs,
                               resume_state=resume_state)
        metrics.update(schema_version=1, evaluation_split="val", smoke=smoke,
                       experiment_name=experiment_name, training_mode=config.training_mode,
                       parameter_count=metadata["parameter_count"],
                       trainable_parameter_count=metadata["trainable_parameter_count"])
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "status.json", {"status": metrics["status"]})
    except BaseException as exc:
        write_json(output_dir / "status.json", {"status": "failed", "error": str(exc)})
        raise
    return metrics
