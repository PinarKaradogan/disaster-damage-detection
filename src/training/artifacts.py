"""Small, dependency-free experiment artifact writers."""
import csv
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output(path, *, binary=False):
    """Flush a sibling temporary file before atomically replacing the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
        with os.fdopen(fd, "wb" if binary else "w", **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, values):
    with atomic_output(path) as handle:
        handle.write(json.dumps(values, indent=2, allow_nan=False) + "\n")


CLASSIFICATION_HISTORY_FIELDS = (
    "epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1", "learning_rate",
)


def write_history(path, history, fieldnames=CLASSIFICATION_HISTORY_FIELDS):
    with atomic_output(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(history)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
