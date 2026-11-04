import csv
import json

import pytest

from src.evaluation.audit_run import audit_run
from src.training.artifacts import file_sha256, write_json

LABELS = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}


def make_metadata_run(tmp_path, *, val_f1=(0.5, 0.7, 0.6, 0.6), best_epoch=2, patience=2):
    manifest = tmp_path / "patches.csv"
    rows = [("train", d) for d in LABELS for _ in range(2)] + [("val", "no-damage")] * 3
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "damage"])
        writer.writerows(rows)
    run = tmp_path / "results" / "run"
    run.mkdir(parents=True)
    config = dict(
        smoke=False, evaluation_split="val", manifest_path="C:/elsewhere/patches.csv",
        manifest_sha256=file_sha256(manifest), label_map=LABELS, early_stopping_patience=patience,
        split_counts={"train": 8, "val": 3}, train_class_counts=[2, 2, 2, 2],
        versions={"python": "3.13"}, num_workers=2,
    )
    metrics = dict(smoke=False, evaluation_split="val", best_epoch=best_epoch, best_val_macro_f1=max(val_f1),
                   class_f1=[0.1] * 4, accuracy=0.5, epochs_completed=len(val_f1), early_stopped=True)
    write_json(run / "config.json", config)
    write_json(run / "metrics.json", metrics)
    write_json(run / "status.json", {"status": "completed"})
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "val_macro_f1"])
        writer.writerows((i + 1, f) for i, f in enumerate(val_f1))
    return manifest


def test_metadata_only_audit_passes_without_checkpoints(tmp_path):
    manifest = make_metadata_run(tmp_path)
    report = audit_run("run", output=tmp_path / "audit.json", project_root=tmp_path,
                       metadata_only=True, manifest=manifest)
    assert report["all_passed"] is True and report["metadata_only"] is True
    assert report["test_used"] is False and report["training_performed"] is False
    assert json.loads((tmp_path / "audit.json").read_text())["all_passed"] is True


def test_metadata_only_audit_flags_inconsistent_best_epoch(tmp_path):
    manifest = make_metadata_run(tmp_path, best_epoch=3)
    report = audit_run("run", output=tmp_path / "audit.json", project_root=tmp_path,
                       metadata_only=True, manifest=manifest)
    assert report["all_passed"] is False
    assert report["checks"]["history_best_epoch_matches_metrics"] is False


def test_audit_refuses_to_overwrite_and_requires_files(tmp_path):
    manifest = make_metadata_run(tmp_path)
    (tmp_path / "audit.json").write_text("{}")
    with pytest.raises(FileExistsError):
        audit_run("run", output=tmp_path / "audit.json", project_root=tmp_path,
                  metadata_only=True, manifest=manifest)
    with pytest.raises(FileNotFoundError):
        audit_run("run", output=tmp_path / "other.json", project_root=tmp_path, manifest=manifest)
