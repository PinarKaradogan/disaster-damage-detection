"""Evaluate an interrupted experiment's best checkpoint on validation only; never train."""
import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.constants import LABEL_MAP
from src.data.dataset import DamagePatchDataset, ROOT
from src.models import create_model
from src.training.artifacts import file_sha256
from src.training.checkpoint import load_checkpoint
from src.training.data import IMAGENET_MEAN, IMAGENET_STD, deterministic_transform
from src.training.engine import run_epoch
from src.training.losses import make_criterion
from src.training.utils import set_training_seed


def evaluate_validation(experiment_dir, *, output, project_root=ROOT):
    experiment_dir, output = Path(experiment_dir).resolve(), Path(output).resolve()
    if output == experiment_dir or experiment_dir in output.parents:
        raise ValueError("Write the evaluation report outside the source experiment directory")
    if output.suffix.lower() != ".json" or output.name == "metrics.json":
        raise ValueError("Use a separate .json evaluation report, never metrics.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    metadata = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((experiment_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"running", "failed", "paused"}:
        raise ValueError("This tool requires an unfinished experiment")
    if metadata["evaluation_split"] != "val" or metadata["smoke"]:
        raise ValueError("Only full validation experiments are supported")
    if metadata["num_classes"] != 4 or metadata["label_map"] != LABEL_MAP:
        raise ValueError("Unsupported class architecture/label map")
    expected_normalization = {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    if metadata["normalization"] != expected_normalization or metadata["augmentation"]["val"] is not None:
        raise ValueError("Unsupported original validation preprocessing")
    if (metadata["batch_size"] < 1 or metadata["input_size"] < 8
            or metadata["num_workers"] < 0):
        raise ValueError("Invalid recorded loader settings")
    device = torch.device(metadata["device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only recorded CPU/CUDA policies are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Original CUDA policy requires CUDA; refusing silent CPU/AMP fallback")
    amp = metadata["amp"]
    if not isinstance(amp, bool) or (amp and device.type != "cuda"):
        raise ValueError("Invalid original AMP policy")
    manifest = Path(metadata["manifest_path"])
    if not manifest.is_absolute():
        manifest = Path(project_root) / manifest
    if not manifest.is_file() or file_sha256(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Manifest missing or SHA256 mismatch")

    set_training_seed(metadata["seed"])
    model = create_model(metadata["model_name"], num_classes=metadata["num_classes"], pretrained=False,
                         training_mode=metadata.get("training_mode", "full"))
    checkpoint_path = experiment_dir / "best.pth"
    try:
        checkpoint = load_checkpoint(checkpoint_path, model, device="cpu")
    except Exception as exc:
        raise ValueError("Cannot safely load best.pth or model architecture mismatch") from exc
    if checkpoint.get("model_name") != metadata["model_name"]:
        raise ValueError("Checkpoint model_name mismatch")
    if json.dumps(checkpoint.get("config"), sort_keys=True) != json.dumps(metadata, sort_keys=True):
        raise ValueError("Checkpoint configuration differs from config.json")
    epoch, recorded_f1 = checkpoint["epoch"], checkpoint["best_val_macro_f1"]
    if not isinstance(epoch, int) or not 1 <= epoch <= metadata["epochs"]:
        raise ValueError("Invalid checkpoint epoch")
    if not isinstance(recorded_f1, (int, float)) or not math.isfinite(recorded_f1) or not 0 <= recorded_f1 <= 1:
        raise ValueError("Invalid recorded checkpoint Macro F1")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    # Legacy checkpoints can contain AdamW tensors; release them before loading images.
    del checkpoint
    criterion = make_criterion(
        metadata["loss"], metadata["sampler"],
        torch.tensor(metadata["class_weights"], dtype=torch.float32)
        if metadata["loss"] == "weighted_ce" else None,
    )
    if metadata["loss"] == "focal" and metadata["focal_gamma"] != 2.0:
        raise ValueError("Unsupported recorded focal gamma")
    dataset = DamagePatchDataset(manifest, split="val", project_root=project_root,
                                 transform=deterministic_transform(metadata["input_size"]))
    if len(dataset) != metadata["split_counts"]["val"]:
        raise ValueError("Validation sample count differs from config.json")
    loader = DataLoader(dataset, batch_size=metadata["batch_size"], shuffle=False,
                        num_workers=metadata["num_workers"])
    last_epoch = None
    history_path = experiment_dir / "history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            last_epoch = int(rows[-1]["epoch"])
    model.to(device)
    started = time.perf_counter()
    # run_epoch switches to eval and disables gradients when no optimizer is passed.
    metrics = run_epoch(model, loader, criterion, device, amp=amp)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    difference = metrics["macro_f1"] - recorded_f1
    tolerance = 1e-6
    report = dict(
        schema_version=1, report_type="interrupted_experiment_validation",
        interrupted_experiment=True, source_status=status["status"],
        experiment_dir=str(experiment_dir), experiment_name=metadata["experiment_name"],
        evaluation_split="val", test_used=False, training_performed=False,
        checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=epoch, last_history_epoch=last_epoch,
        manifest_path=str(manifest.resolve()), manifest_sha256=metadata["manifest_sha256"],
        model_name=metadata["model_name"], loss=metadata["loss"], sampler=metadata["sampler"],
        batch_size=metadata["batch_size"], input_size=metadata["input_size"],
        num_workers=metadata["num_workers"], device=str(device), amp=amp,
        normalization=metadata["normalization"], validation_samples=len(dataset),
        label_map=LABEL_MAP, confusion_matrix_axes="rows=true, columns=predicted; label IDs 0..3",
        validation=metrics, recorded_val_macro_f1=recorded_f1,
        macro_f1_comparison=dict(signed_difference=difference, absolute_difference=abs(difference),
                                 tolerance=tolerance, matches_within_tolerance=abs(difference) <= tolerance),
        evaluation_time_seconds=time.perf_counter() - started,
        versions=dict(python=platform.python_version(), torch=str(torch.__version__)),
        original_versions=metadata["versions"],
    )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a competing writer after the initial check.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_validation(args.experiment_dir, output=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
