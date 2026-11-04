"""Read-only audit of a completed Phase 4 run and validation re-evaluation of best.pth.

python -m src.evaluation.audit_run --experiment <name> --output results/phase4_audit/audit_<name>.json

Never trains, never opens test patches and never writes into the experiment folder.
Re-evaluation uses the recorded device (CUDA runs require CUDA).
"""
import argparse
import csv
import json
import math
import multiprocessing
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.constants import LABEL_MAP
from src.data.dataset import ROOT, DamagePatchDataset
from src.models import create_model
from src.training.artifacts import file_sha256
from src.training.checkpoint import load_checkpoint
from src.training.data import deterministic_transform
from src.training.engine import run_epoch
from src.training.losses import make_criterion

METADATA_FILES = ("config.json", "metrics.json", "history.csv", "status.json")
REQUIRED_FILES = ("best.pth", "last.pth", *METADATA_FILES)
REEVAL_TOLERANCE = 1e-4


def _passed(value):
    if isinstance(value, dict):
        return all(value.values())
    if isinstance(value, bool):
        return value
    return isinstance(value, int) and value > 0


def audit_run(name, *, output, num_workers=4, project_root=ROOT, metadata_only=False, manifest=None):
    """Audit results/<name>.

    metadata_only=True is for runs whose checkpoints were not shared (e.g. from another
    machine): checkpoint integrity and best.pth re-evaluation are skipped, all metadata
    consistency checks still run. ``manifest`` overrides the recorded (machine-specific)
    manifest path; its SHA-256 must still match the recorded value.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    run_dir = Path(project_root) / "results" / name
    if output.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Write the audit report outside the experiment folder")
    required = METADATA_FILES if metadata_only else REQUIRED_FILES
    checks = {"files_present": {f: (run_dir / f).is_file() for f in required}}
    if not all(checks["files_present"].values()):
        raise FileNotFoundError(f"Missing run files: {checks['files_present']}")
    if not metadata_only:
        checks["tensorboard_events"] = len(list((Path(project_root) / "runs" / name).glob("events.out.tfevents.*")))

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))

    manifest = Path(manifest) if manifest is not None else Path(cfg["manifest_path"])
    checks["status_completed"] = status["status"] == "completed"
    checks["not_smoke"] = cfg["smoke"] is False and met["smoke"] is False
    checks["evaluation_split_val"] = cfg["evaluation_split"] == "val" == met["evaluation_split"]
    checks["manifest_sha256_matches_file"] = file_sha256(manifest) == cfg["manifest_sha256"]
    checks["label_map_matches_code"] = cfg["label_map"] == LABEL_MAP
    checks["history_rows_equal_epochs_completed"] = len(history) == met["epochs_completed"]
    val_f1 = [float(row["val_macro_f1"]) for row in history]
    best_idx = max(range(len(val_f1)), key=lambda i: (val_f1[i], -i))  # ties keep the earlier epoch
    checks["history_best_epoch_matches_metrics"] = int(history[best_idx]["epoch"]) == met["best_epoch"]
    checks["history_best_f1_matches_metrics"] = math.isclose(val_f1[best_idx], met["best_val_macro_f1"], abs_tol=1e-12)
    later = val_f1[best_idx + 1:]
    checks["early_stop_consistent"] = (
        met["early_stopped"] is True and len(later) == cfg["early_stopping_patience"]
        and all(v <= met["best_val_macro_f1"] for v in later)
    )

    if metadata_only:
        splits = pd.read_csv(manifest, usecols=["split", "damage"])
        val_labels = splits.loc[splits["split"] == "val", "damage"]
        train_labels = splits.loc[splits["split"] == "train", "damage"].map(LABEL_MAP)
        checks["val_count_matches_config"] = len(val_labels) == cfg["split_counts"]["val"]
        checks["train_class_counts_match_manifest"] = (
            [int((train_labels == i).sum()) for i in range(4)] == cfg["train_class_counts"])
        report = dict(
            experiment=name, test_used=False, training_performed=False, metadata_only=True,
            checks=checks, all_passed=all(_passed(v) for v in checks.values()),
            recorded=dict(best_epoch=met["best_epoch"], macro_f1=met["best_val_macro_f1"],
                          class_f1=met["class_f1"], accuracy=met["accuracy"]),
            not_verified=["checkpoint integrity", "best.pth re-evaluation (checkpoints not shared)"],
            versions_recorded=cfg["versions"], num_workers_recorded=cfg["num_workers"],
        )
        return _write_report(output, report)

    best = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    last = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=False)
    checks["best_ckpt_epoch_matches"] = best["epoch"] == met["best_epoch"]
    checks["best_ckpt_config_equals_config_json"] = (
        json.dumps(best["config"], sort_keys=True) == json.dumps(cfg, sort_keys=True))
    checks["last_ckpt_epoch_is_final"] = last["epoch"] == met["epochs_completed"]
    checks["last_ckpt_finished_flag"] = bool(last.get("finished"))
    checks["last_ckpt_has_optimizer_rng_earlystop"] = all(
        key in last for key in ("optimizer_state_dict", "rng", "early_stopping"))
    embedded = last.get("best_checkpoint")
    checks["last_embedded_best_equals_best_pth"] = isinstance(embedded, dict) and all(
        torch.equal(embedded["model_state_dict"][k], v) for k, v in best["model_state_dict"].items())
    del last, best

    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Recorded CUDA run requires CUDA for re-evaluation")
    model = create_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=False,
                         training_mode=cfg["training_mode"])
    load_checkpoint(run_dir / "best.pth", model, device="cpu")
    weights = torch.tensor(cfg["class_weights"], dtype=torch.float32) if cfg["loss"] == "weighted_ce" else None
    criterion = make_criterion(cfg["loss"], cfg["sampler"], weights)
    dataset = DamagePatchDataset(manifest, split="val", project_root=project_root,
                                 transform=deterministic_transform(cfg["input_size"]))
    checks["val_count_matches_config"] = len(dataset) == cfg["split_counts"]["val"]
    checks["val_split_only"] = bool(dataset.df["split"].eq("val").all())
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=num_workers)
    model.to(device)
    metrics = run_epoch(model, loader, criterion, device, amp=cfg["amp"])
    difference = metrics["macro_f1"] - met["best_val_macro_f1"]
    checks["reeval_macro_f1_matches_metrics_1e-4"] = abs(difference) <= REEVAL_TOLERANCE

    keep = ("macro_f1", "class_f1", "class_precision", "class_recall", "confusion_matrix", "accuracy", "loss")
    report = dict(
        experiment=name, test_used=False, training_performed=False, checks=checks,
        all_passed=all(_passed(v) for v in checks.values()),
        recorded=dict(best_epoch=met["best_epoch"], macro_f1=met["best_val_macro_f1"], class_f1=met["class_f1"]),
        reevaluated={k: metrics[k] for k in keep if k in metrics},
        reeval_minus_recorded=difference,
        confusion_matrix_axes="rows=true, columns=predicted; label IDs 0..3",
        versions_recorded=cfg["versions"], num_workers_recorded=cfg["num_workers"],
    )
    return _write_report(output, report)


def _write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true",
                        help="skip checkpoint checks and re-evaluation (checkpoints not available)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="local manifest path when the recorded path is from another machine")
    args = parser.parse_args(argv)
    report = audit_run(args.experiment, output=args.output, num_workers=args.num_workers,
                       metadata_only=args.metadata_only, manifest=args.manifest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
