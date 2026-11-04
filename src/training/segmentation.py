"""Phase 5 U-Net building localization training (train/val only).

Reuses the Phase 4 conventions: atomic best.pth / last.pth, full resume state
(model, optimizer, scheduler, AMP scaler, RNG, early stopping, history), refusal to
overwrite an existing experiment, TensorBoard, versioned config/metrics/status JSON.
Model selection: maximum validation global pixel Dice.
"""
import copy
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import ROOT
from src.data.segmentation_dataset import IMAGENET_MEAN, IMAGENET_STD, BuildingSegmentationDataset, read_split_ids
from src.evaluation.segmentation_metrics import SegmentationMeter
from src.models.unet import create_unet, pretrained_source
from src.training.artifacts import file_sha256, write_history, write_json
from src.training.checkpoint import EarlyStopping, atomic_save, capture_rng, restore_rng
from src.training.engine import _cpu_snapshot
from src.training.utils import get_device, set_training_seed

HISTORY_FIELDS = ("epoch", "train_loss", "train_dice", "val_loss", "val_dice", "val_iou", "learning_rate")


@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "unet_resnet34"
    seed: int = 42
    image_size: int = 1024
    tile_size: int = 512
    tiles_per_image: int = 4
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    early_stopping_patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler_eta_min: float = 1e-6
    amp: bool = True
    threshold: float = 0.5
    bce_weight: float = 1.0
    dice_weight: float = 1.0

    def to_dict(self):
        return asdict(self)


class BCEDiceLoss(nn.Module):
    """bce_weight * BCEWithLogits + dice_weight * (1 - soft Dice over the whole batch)."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth

    def forward(self, logits, masks):
        logits, masks = logits.float(), masks.float()
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * masks).sum()
        dice = (2 * intersection + self.smooth) / (probabilities.sum() + masks.sum() + self.smooth)
        return self.bce_weight * self.bce(logits, masks) + self.dice_weight * (1 - dice)


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None, amp=False, threshold=0.5):
    training = optimizer is not None
    phase = "train" if training else "val"
    started = time.perf_counter()
    print(f"[{phase}] epoch_start completed_batches=0 elapsed=0.0s waiting_for_first_batch", file=sys.stderr, flush=True)
    model.train(training)
    meter = SegmentationMeter(threshold)
    loss_total, samples = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index == 1:
                print(f"[{phase}] first_batch_received completed_batches=0 "
                      f"elapsed={time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; training stopped")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_total += loss.item() * len(images)
            samples += len(images)
            meter.update(logits.detach(), masks, batch["image_id"])
            if batch_index % 100 == 0:
                print(f"[{phase}] completed_batches={batch_index} "
                      f"elapsed={time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    if not samples:
        raise ValueError("Loader must contain at least one sample")
    return {"loss": loss_total / samples, **meter.compute()}


def build_loaders(config, *, project_root=ROOT, smoke=False):
    train_ids = read_split_ids("train", project_root=project_root)
    val_ids = read_split_ids("val", project_root=project_root)
    if smoke:  # pipeline check only; never reported as a result
        train_ids, val_ids = train_ids[:4], val_ids[:2]
    common = dict(project_root=project_root, tile_size=config.tile_size, image_size=config.image_size,
                  seed=config.seed)
    train = BuildingSegmentationDataset("train", image_ids=train_ids, tiles_per_image=config.tiles_per_image, **common)
    val = BuildingSegmentationDataset("val", image_ids=val_ids, **common)
    kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers)
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(config.seed), **kwargs)
    val_loader = DataLoader(val, shuffle=False, **kwargs)
    return train_loader, val_loader


def _validate_resume(payload, metadata):
    required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                "epoch", "early_stopping", "rng", "config", "history", "best_checkpoint", "best_metrics",
                "training_time_seconds", "finished"}
    if not isinstance(payload, dict) or payload.get("resume_schema_version") != 1 or not required <= payload.keys():
        raise ValueError("Checkpoint lacks complete resume state")
    if json.dumps(payload["config"], sort_keys=True) != json.dumps(metadata, sort_keys=True):
        changed = sorted(k for k in set(payload["config"]) | set(metadata)
                         if json.dumps(payload["config"].get(k), sort_keys=True) != json.dumps(metadata.get(k), sort_keys=True))
        raise ValueError(f"Resume configuration mismatch: {', '.join(changed)}")
    epoch = payload["epoch"]
    if [row["epoch"] for row in payload["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Resume checkpoint history is not contiguous")
    if payload["finished"] or epoch >= metadata["epochs"]:
        raise ValueError("Experiment already completed; resume refused")
    if payload["early_stopping"]["bad_epochs"] >= metadata["early_stopping_patience"]:
        raise ValueError("Experiment already reached early stopping; resume refused")


def fit(model, train_loader, val_loader, criterion, config, device, *, output_dir, log_dir, metadata,
        epochs, session_epochs=None, resume_state=None):
    output_dir = Path(output_dir)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.scheduler_eta_min)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopping = EarlyStopping(config.early_stopping_patience)
    history, best_metrics, best_checkpoint = [], None, None
    start_epoch, elapsed, stop = 1, 0.0, False
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        stopping.best = resume_state["early_stopping"]["best"]
        stopping.bad_epochs = resume_state["early_stopping"]["bad_epochs"]
        history = copy.deepcopy(resume_state["history"])
        best_metrics, best_checkpoint = resume_state["best_metrics"], resume_state["best_checkpoint"]
        elapsed, start_epoch = resume_state["training_time_seconds"], resume_state["epoch"] + 1
        restore_rng(resume_state["rng"], train_loader, val_loader)
        warnings.warn("Resume: rebuilding history.csv and best.pth from committed last.pth", UserWarning)
        atomic_save(output_dir / "best.pth", best_checkpoint)
        write_history(output_dir / "history.csv", history, HISTORY_FIELDS)
    end_epoch = min(epochs, start_epoch + session_epochs - 1) if session_epochs else epochs
    started = time.perf_counter()
    with SummaryWriter(log_dir=str(log_dir), purge_step=start_epoch if resume_state else None) as writer:
        for epoch in range(start_epoch, end_epoch + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler,
                              amp=use_amp, threshold=config.threshold)
            val = run_epoch(model, val_loader, criterion, device, amp=use_amp, threshold=config.threshold)
            history.append(dict(epoch=epoch, train_loss=train["loss"], train_dice=train["dice"],
                                val_loss=val["loss"], val_dice=val["dice"], val_iou=val["iou"],
                                learning_rate=learning_rate))
            for split, metrics in (("train", train), ("val", val)):
                writer.add_scalar(f"{split}/loss", metrics["loss"], epoch)
                writer.add_scalar(f"{split}/dice", metrics["dice"], epoch)
            writer.add_scalar("val/iou", val["iou"], epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)
            improved, stop = stopping.update(val["dice"])
            if improved:
                best_metrics = dict(best_epoch=epoch, best_val_dice=val["dice"],
                                    val={k: v for k, v in val.items()})
                best_checkpoint = _cpu_snapshot(dict(model_state_dict=model.state_dict(), epoch=epoch,
                                                     best_val_dice=val["dice"], config=metadata,
                                                     model_name=config.model_name))
            scheduler.step()
            atomic_save(output_dir / "last.pth", dict(
                resume_schema_version=1, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(), epoch=epoch,
                early_stopping=dict(best=stopping.best, bad_epochs=stopping.bad_epochs),
                rng=capture_rng(train_loader, val_loader), config=metadata, history=history,
                best_checkpoint=best_checkpoint, best_metrics=best_metrics,
                training_time_seconds=elapsed + time.perf_counter() - started, finished=stop or epoch == epochs))
            if improved:
                atomic_save(output_dir / "best.pth", best_checkpoint)
            write_history(output_dir / "history.csv", history, HISTORY_FIELDS)
            writer.flush()
            print(f"epoch={epoch} train_loss={train['loss']:.4f} val_loss={val['loss']:.4f} "
                  f"val_dice={val['dice']:.4f} val_iou={val['iou']:.4f}", file=sys.stderr, flush=True)
            if stop:
                break
    return {**best_metrics, "training_time_seconds": elapsed + time.perf_counter() - started,
            "epochs_completed": len(history), "early_stopped": stop,
            "status": "completed" if stop or len(history) == epochs else "paused"}


def run_segmentation_experiment(config, *, experiment_name, smoke=False, resume=False, session_epochs=None,
                                project_root=ROOT, results_root=None, runs_root=None, device=None,
                                pretrained=True):
    if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in {".", ".."}:
        raise ValueError("experiment_name must be a single directory name")
    if min(config.batch_size, config.epochs, config.tiles_per_image) < 1 or config.num_workers < 0:
        raise ValueError("batch size, epochs and tiles_per_image must be positive; workers nonnegative")
    if session_epochs is not None and session_epochs < 1:
        raise ValueError("session_epochs must be positive")
    project_root = Path(project_root)
    output_dir = Path(results_root or project_root / "results") / experiment_name
    log_dir = Path(runs_root or project_root / "runs") / experiment_name
    if resume:
        if not (output_dir / "last.pth").is_file():
            raise ValueError("Resume requires a complete last.pth")
        status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("status") not in {"paused", "failed", "running"}:
            raise ValueError("Experiment already completed or invalid status; resume refused")
    else:
        for directory in (output_dir, log_dir):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing experiment: {directory}")
    set_training_seed(config.seed)
    device = get_device() if device is None else torch.device(device)
    train_loader, val_loader = build_loaders(config, project_root=project_root, smoke=smoke)
    model = create_unet(config.model_name, pretrained=pretrained and not resume)
    criterion = BCEDiceLoss(config.bce_weight, config.dice_weight)
    epochs = min(2, config.epochs) if smoke else config.epochs
    split_dir = project_root / "splits/standard"
    metadata = {
        **config.to_dict(), "schema_version": 1, "task": "building_segmentation",
        "experiment_name": experiment_name, "evaluation_split": "val", "smoke": smoke, "epochs": epochs,
        "requested_epochs": config.epochs, "selection_metric": "val global pixel Dice",
        "input": "PRE-disaster RGB image", "target": "data/train/targets/<id>_pre_disaster_target.png (0/1)",
        "split_files_sha256": {s: file_sha256(split_dir / f"{s}_images.txt") for s in ("train", "val")},
        "split_counts": {"train_images": len(train_loader.dataset.image_ids),
                         "val_images": len(val_loader.dataset.image_ids),
                         "train_tiles_per_epoch": len(train_loader.dataset), "val_tiles": len(val_loader.dataset)},
        "augmentation": {"train": "random tile crop + identical rot90(k in 0..3) + horizontal flip p=0.5",
                         "val": "fixed non-overlapping tile grid, no augmentation"},
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "loss": "bce_weight * BCEWithLogits + dice_weight * (1 - batch soft Dice, smooth=1)",
        "optimizer": "adamw", "scheduler": "cosine", "scheduler_t_max": epochs,
        "amp": config.amp and device.type == "cuda", "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pretrained": pretrained, "pretrained_source": pretrained_source(config.model_name) if pretrained else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
    }
    import segmentation_models_pytorch as smp
    metadata["versions"]["segmentation_models_pytorch"] = smp.__version__
    resume_state = None
    if resume:
        stored = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        if json.dumps(stored, sort_keys=True) != json.dumps(metadata, sort_keys=True):
            raise ValueError("Resume config.json differs from the current configuration")
        resume_state = torch.load(output_dir / "last.pth", map_location="cpu", weights_only=True)
        _validate_resume(resume_state, metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(output_dir / "config.json", metadata)
    write_json(output_dir / "status.json", {"status": "running"})
    print(f"parameters total={metadata['parameter_count']} train_tiles={metadata['split_counts']['train_tiles_per_epoch']} "
          f"val_tiles={metadata['split_counts']['val_tiles']}", file=sys.stderr, flush=True)
    try:
        metrics = fit(model, train_loader, val_loader, criterion, config, device, output_dir=output_dir,
                      log_dir=log_dir, metadata=metadata, epochs=epochs, session_epochs=session_epochs,
                      resume_state=resume_state)
        metrics.update(schema_version=1, evaluation_split="val", smoke=smoke, experiment_name=experiment_name,
                       parameter_count=metadata["parameter_count"])
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "status.json", {"status": metrics["status"]})
    except BaseException as exc:
        write_json(output_dir / "status.json", {"status": "failed", "error": str(exc)})
        raise
    return metrics
