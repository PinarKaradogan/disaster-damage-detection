"""One paired-model engine for CPU tests and CUDA AMP training."""
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.training.artifacts import write_history
from src.training.checkpoint import EarlyStopping, save_checkpoint
from src.training.metrics import classification_metrics


def run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    criterion.to(device)
    targets, predictions = [], []
    loss_total, denominator = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
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
    if not targets:
        raise ValueError("Loader must contain at least one sample")
    return {"loss": loss_total / denominator, **classification_metrics(targets, predictions)}


def fit(model, train_loader, val_loader, criterion, config, device, *, output_dir, log_dir,
        metadata, epochs=None):
    if config.optimizer != "adamw" or config.scheduler != "cosine":
        raise ValueError("Phase 4 supports AdamW and cosine scheduler")
    epochs = config.epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=config.scheduler_eta_min,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopping = EarlyStopping(config.early_stopping_patience)
    history, best_metrics = [], None
    started = time.perf_counter()
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for epoch in range(1, epochs + 1):
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
                save_checkpoint(output_dir / "best.pth", model, epoch=epoch,
                                best_val_macro_f1=val["macro_f1"], config=metadata,
                                optimizer=optimizer)
            scheduler.step()
            # Persist after each epoch so interrupted long runs retain their history.
            write_history(output_dir / "history.csv", history)
            writer.flush()
            print(f"epoch={epoch} train_loss={train['loss']:.4f} "
                  f"val_loss={val['loss']:.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
            if stop:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {**best_metrics, "training_time_seconds": time.perf_counter() - started,
            "epochs_completed": len(history), "early_stopped": stop}, history
