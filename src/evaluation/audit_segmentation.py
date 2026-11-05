"""Read-only audit of a completed U-Net run (validation only; never trains, never opens test).

python -m src.evaluation.audit_segmentation --experiment <name> --output results/phase5_audit/audit_<name>.json

Checks file completeness, split-file hashes, history/metrics/early-stopping consistency and
best/last checkpoint integrity; re-evaluates best.pth on the training-time validation tiles
(must reproduce the recorded Dice) and additionally on full 1024x1024 validation images
(what Phase 5.4 integration uses). Per-image full-image metrics are written to a CSV next
to the report for failure analysis.
"""
import argparse
import csv
import json
import math
import multiprocessing
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ROOT
from src.data.segmentation_dataset import load_pair, normalize, read_split_ids
from src.evaluation.segmentation_metrics import SegmentationMeter
from src.models.unet import create_unet
from src.training.artifacts import file_sha256
from src.training.segmentation import BCEDiceLoss, SegmentationConfig, build_loaders, run_epoch

REQUIRED_FILES = ("best.pth", "last.pth", "config.json", "metrics.json", "history.csv", "status.json")
REEVAL_TOLERANCE = 1e-4


class FullImageDataset(Dataset):
    def __init__(self, image_ids, *, project_root=ROOT, image_size=1024):
        self.image_ids, self.project_root, self.image_size = list(image_ids), project_root, image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image, mask = load_pair(image_id, project_root=self.project_root, image_size=self.image_size)
        return {"image": normalize(image), "mask": torch.from_numpy(mask.copy()).float().unsqueeze(0),
                "image_id": image_id}


def config_from_metadata(metadata):
    names = {f.name for f in fields(SegmentationConfig)}
    return SegmentationConfig(**{k: v for k, v in metadata.items() if k in names})


@torch.no_grad()
def evaluate_full_images(model, image_ids, device, *, project_root=ROOT, image_size=1024, threshold=0.5,
                         num_workers=4, amp=True):
    loader = DataLoader(FullImageDataset(image_ids, project_root=project_root, image_size=image_size),
                        batch_size=2, shuffle=False, num_workers=num_workers)
    model.eval()
    meter = SegmentationMeter(threshold)
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch["image"].to(device))
        meter.update(logits.float(), batch["mask"].to(device), batch["image_id"])
    per_image = [dict(image_id=i, gt_pixels=c[3], tp=c[0], fp=c[1], fn=c[2],
                      dice=(2 * c[0] / (2 * c[0] + c[1] + c[2])) if (2 * c[0] + c[1] + c[2]) else 1.0)
                 for i, c in meter.per_image.items()]
    return meter.compute(), per_image


def audit_segmentation(name, *, output, project_root=ROOT, num_workers=4):
    output = Path(output)
    per_image_csv = output.with_name(output.stem + "_per_image_full.csv")
    if output.exists() or per_image_csv.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    run_dir = Path(project_root) / "results" / name
    checks = {"files_present": {f: (run_dir / f).is_file() for f in REQUIRED_FILES}}
    if not all(checks["files_present"].values()):
        raise FileNotFoundError(f"Missing run files: {checks['files_present']}")
    checks["tensorboard_events"] = len(list((Path(project_root) / "runs" / name).glob("events.out.tfevents.*")))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))

    split_dir = Path(project_root) / "splits/standard"
    checks["status_completed"] = status["status"] == "completed"
    checks["not_smoke"] = cfg["smoke"] is False and met["smoke"] is False
    checks["task_and_split"] = cfg["task"] == "building_segmentation" and cfg["evaluation_split"] == "val"
    checks["split_files_unchanged"] = all(
        file_sha256(split_dir / f"{s}_images.txt") == cfg["split_files_sha256"][s] for s in ("train", "val"))
    checks["history_contiguous"] = [int(r["epoch"]) for r in history] == list(range(1, len(history) + 1))
    checks["history_rows_equal_epochs_completed"] = len(history) == met["epochs_completed"]
    dice = [float(r["val_dice"]) for r in history]
    best_idx = max(range(len(dice)), key=lambda i: (dice[i], -i))
    checks["history_best_matches_metrics"] = (int(history[best_idx]["epoch"]) == met["best_epoch"]
                                              and math.isclose(dice[best_idx], met["best_val_dice"], abs_tol=1e-12))
    later = dice[best_idx + 1:]
    checks["stopping_consistent"] = (
        (met["early_stopped"] and len(later) == cfg["early_stopping_patience"] and all(v <= dice[best_idx] for v in later))
        or (not met["early_stopped"] and len(history) == cfg["epochs"]))

    best = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True)
    last = torch.load(run_dir / "last.pth", map_location="cpu", weights_only=True)
    checks["best_ckpt_epoch_matches"] = best["epoch"] == met["best_epoch"]
    checks["best_ckpt_config_equals_config_json"] = (
        json.dumps(best["config"], sort_keys=True) == json.dumps(cfg, sort_keys=True))
    checks["last_ckpt_final_and_finished"] = last["epoch"] == met["epochs_completed"] and bool(last["finished"])
    embedded = last["best_checkpoint"]["model_state_dict"]
    checks["last_embedded_best_equals_best_pth"] = all(
        torch.equal(embedded[k], v) for k, v in best["model_state_dict"].items())
    del last

    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Recorded CUDA run requires CUDA for re-evaluation")
    model = create_unet(cfg["model_name"], pretrained=False)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    config = config_from_metadata({**cfg, "num_workers": num_workers})
    _, val_loader = build_loaders(config, project_root=project_root)
    checks["val_images_match_config"] = len(val_loader.dataset.image_ids) == cfg["split_counts"]["val_images"]
    tiles = run_epoch(model, val_loader, BCEDiceLoss(cfg["bce_weight"], cfg["dice_weight"]), device,
                      amp=cfg["amp"], threshold=cfg["threshold"])
    difference = tiles["dice"] - met["best_val_dice"]
    checks["reeval_tile_dice_matches_metrics_1e-4"] = abs(difference) <= REEVAL_TOLERANCE

    full, per_image = evaluate_full_images(model, read_split_ids("val", project_root=project_root), device,
                                           project_root=project_root, image_size=cfg["image_size"],
                                           threshold=cfg["threshold"], num_workers=num_workers, amp=cfg["amp"])

    def passed(value):
        if isinstance(value, dict):
            return all(value.values())
        if isinstance(value, bool):
            return value
        return isinstance(value, int) and value > 0

    report = dict(experiment=name, test_used=False, training_performed=False, checks=checks,
                  all_passed=all(passed(v) for v in checks.values()),
                  recorded=dict(best_epoch=met["best_epoch"], best_val_dice=met["best_val_dice"]),
                  reevaluated_val_tiles={k: v for k, v in tiles.items()}, reeval_minus_recorded=difference,
                  full_image_val=full, full_image_note="Same best.pth applied to whole 1024x1024 val images "
                  "(no tiling); reported separately from the selection metric.",
                  per_image_csv=per_image_csv.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with per_image_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "gt_pixels", "tp", "fp", "fn", "dice"])
        writer.writeheader()
        writer.writerows(sorted(per_image, key=lambda row: row["image_id"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(audit_segmentation(args.experiment, output=args.output, num_workers=args.num_workers), indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
