"""Small, dependency-free experiment artifact writers."""
import csv
import hashlib
import json
from pathlib import Path


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_history(path, history):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1", "learning_rate",
        ])
        writer.writeheader()
        writer.writerows(history)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
