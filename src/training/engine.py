"""One paired-model engine for CPU tests and CUDA AMP training."""
import time
import copy
import sys
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.training.artifacts import write_history
from src.training.checkpoint import EarlyStopping, atomic_save, capture_rng, restore_rng
from src.training.metrics import classification_metrics


def _cpu_snapshot(state):
    """Deep-copy state containers, copying tensors directly to independent CPU storage."""
    memo = {}

    def collect(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.detach().to(device="cpu", copy=True)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(state)
    # Preserve state_dict metadata and container types without CUDA deepcopy allocations.
    return copy.deepcopy(state, memo)


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    phase = "train" if training else "val"
    started = time.perf_counter()
    print(f"[{phase}] epoch_start completed_batches=0 elapsed=0.0s waiting_for_first_batch", file=sys.stderr, flush=True)
    model.train(training)
    criterion.to(device)
    targets, predictions = [], []
    loss_total, denominator = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index == 1:
                print(f"[{phase}] first_batch_received completed_batches=0 "
                      f"elapsed={time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
            pre = batch["pre"].to(device)
            post = batch["post"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(pre, post)
                loss = criterion(logits, labels)
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
            # Weighted CE uses sum(target weights), not batch size, as denominator.
            mass = float(labels.numel())
            if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
                mass = criterion.weight[labels].sum().item()
            loss_total += loss.item() * mass
            denominator += mass
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if batch_index % 100 == 0:
                print(f"[{phase}] completed_batches={batch_index} "
                      f"elapsed={time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    if not targets:
        raise ValueError("Loader must contain at least one sample")
    return {"loss": loss_total / denominator, **classification_metrics(targets, predictions)}


def fit(model, train_loader, val_loader, criterion, config, device, *, output_dir, log_dir,
        metadata, epochs=None, session_epochs=None, resume_state=None):
    if config.optimizer != "adamw" or config.scheduler != "cosine":
        raise ValueError("Phase 4 supports AdamW and cosine scheduler")
    epochs = config.epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if session_epochs is not None and session_epochs < 1:
        raise ValueError("session_epochs must be positive")
    output_dir = Path(output_dir)
    if getattr(model, "training_mode", "full") != config.training_mode:
        raise ValueError("Model training_mode differs from training config")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.scheduler_eta_min,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopping = EarlyStopping(config.early_stopping_patience)
    history, best_metrics = [], None
    best_checkpoint = None
    start_epoch, elapsed, stop = 1, 0.0, False
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        stopping.best = resume_state["early_stopping"]["best"]
        stopping.bad_epochs = resume_state["early_stopping"]["bad_epochs"]
        history = copy.deepcopy(resume_state["history"])
        best_metrics = resume_state["best_metrics"]
        best_checkpoint = resume_state["best_checkpoint"]
        elapsed = resume_state["training_time_seconds"]
        start_epoch = resume_state["epoch"] + 1
        restore_rng(resume_state["rng"], train_loader, val_loader)
        # last.pth commits an epoch before derived best/history files are published.
        warnings.warn("Resume: rebuilding history.csv and best.pth from committed last.pth", UserWarning)
        atomic_save(output_dir / "best.pth", best_checkpoint)
        write_history(output_dir / "history.csv", history)
    end_epoch = min(epochs, start_epoch + session_epochs - 1) if session_epochs else epochs
    started = time.perf_counter()
    with SummaryWriter(log_dir=str(log_dir), purge_step=start_epoch if resume_state else None) as writer:
        for epoch in range(start_epoch, end_epoch + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train = run_epoch(model, train_loader, criterion, device,
                              optimizer=optimizer, scaler=scaler, amp=use_amp)
            val = run_epoch(model, val_loader, criterion, device, amp=use_amp)
            row = dict(epoch=epoch, train_loss=train["loss"], train_macro_f1=train["macro_f1"],
                       val_loss=val["loss"], val_macro_f1=val["macro_f1"], learning_rate=learning_rate)
            history.append(row)
            for split, metrics in [("train", train), ("val", val)]:
                writer.add_scalar(f"{split}/loss", metrics["loss"], epoch)
                writer.add_scalar(f"{split}/macro_f1", metrics["macro_f1"], epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)
            improved, stop = stopping.update(val["macro_f1"])
            if improved:
                best_metrics = dict(best_epoch=epoch, best_val_macro_f1=val["macro_f1"],
                                    class_f1=val["class_f1"], accuracy=val["accuracy"],
                                    val_loss=val["loss"])
                best_checkpoint = _cpu_snapshot(dict(
                    model_state_dict=model.state_dict(), epoch=epoch,
                    best_val_macro_f1=val["macro_f1"], config=metadata,
                    model_name=config.model_name, optimizer_state_dict=optimizer.state_dict()))
            scheduler.step()
            atomic_save(output_dir / "last.pth", dict(
                resume_schema_version=1, model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(), epoch=epoch,
                best_val_macro_f1=stopping.best,
                early_stopping=dict(best=stopping.best, bad_epochs=stopping.bad_epochs),
                rng=capture_rng(train_loader, val_loader), config=metadata, history=history,
                best_checkpoint=best_checkpoint, best_metrics=best_metrics,
                training_time_seconds=elapsed + time.perf_counter() - started,
                finished=stop or epoch == epochs))
            if improved:
                atomic_save(output_dir / "best.pth", best_checkpoint)
            write_history(output_dir / "history.csv", history)
            writer.flush()
            print(f"epoch={epoch} train_loss={train['loss']:.4f} "
                  f"val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}", file=sys.stderr, flush=True)
            if stop:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**best_metrics, "training_time_seconds": elapsed + time.perf_counter() - started,
            "epochs_completed": len(history), "early_stopped": stop,
            "status": "completed" if stop or len(history) == epochs else "paused"}, history
