from pathlib import Path

import torch


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
    torch.save(payload, path)


def load_checkpoint(path, model, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return payload
