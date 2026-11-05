"""Phase 6 one-shot TEST evaluation (never trains, never tunes).

python -m src.evaluation.evaluate_test classifier   --experiment <run>
python -m src.evaluation.evaluate_test segmentation --experiment <unet run>
python -m src.evaluation.evaluate_test two-stage    [--threshold 0.9 --split-erosion 4]

Guards (all must pass before any test file is read):
- results/phase6_protocol.md says "Durum: **onaylandı**" (a draft cannot open the test set);
- it is committed and has no uncommitted changes; its commit is recorded;
- the run's best.pth SHA-256 equals the value pre-registered in the protocol;
- the output file does not exist yet (each evaluation runs once; nothing is overwritten).
Outputs go to results/phase6_test/: <kind>_<name>.json (+ a per-building / per-image CSV).
"""
import argparse
import csv
import json
import multiprocessing
import platform
import re
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.constants import LABEL_MAP
from src.data.dataset import ROOT, DamagePatchDataset
from src.evaluation.audit_segmentation import evaluate_full_images
from src.evaluation.two_stage import DEFAULT_CLASSIFIER, DEFAULT_UNET, MIN_AREA, evaluate_images, harmonic_mean
from src.models import create_model
from src.models.unet import create_unet
from src.training.artifacts import file_sha256, write_json
from src.training.data import deterministic_transform
from src.training.metrics import classification_metrics

PROTOCOL = "results/phase6_protocol.md"
OUTPUT_DIR = "results/phase6_test"
TEST_SPLIT_FILE = "splits/standard/test_images.txt"
APPROVED_MARKER = "Durum: **onaylandı**"
SHA_ROW = re.compile(r"\|\s*`(?P<run>[^`]+)`[^|]*\|\s*`(?P<sha>[0-9a-f]{64})`")


# ------------------------------------------------------------------ guards
def protocol_commit(project_root=ROOT, *, check_git=True):
    """Protocol must be committed and clean; returns (commit hash, registered run -> sha)."""
    path = Path(project_root) / PROTOCOL
    if not path.is_file():
        raise FileNotFoundError(f"Missing test protocol: {path}")
    text = path.read_text(encoding="utf-8")
    if APPROVED_MARKER not in text:
        raise RuntimeError(f"Test protocol is not approved yet (expected the line '{APPROVED_MARKER}...')")
    registered = {m["run"]: m["sha"] for m in SHA_ROW.finditer(text)}
    if not check_git:
        return None, registered

    def git(*args):
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=False)

    if git("ls-files", "--error-unmatch", PROTOCOL).returncode != 0:
        raise RuntimeError("Test protocol is not committed; commit it before opening the test set")
    if git("diff", "--quiet", "HEAD", "--", PROTOCOL).returncode != 0:
        raise RuntimeError("Test protocol has uncommitted changes; commit them first")
    commit = git("log", "-1", "--format=%H", "--", PROTOCOL).stdout.strip()
    return commit, registered


def verify_checkpoint(run, checkpoint, registered):
    if run not in registered:
        raise ValueError(f"{run} is not pre-registered with a SHA-256 in {PROTOCOL}")
    digest = file_sha256(checkpoint)
    if digest != registered[run]:
        raise ValueError(f"{run}: best.pth SHA-256 {digest} differs from the protocol {registered[run]}")
    return digest


def output_path(kind, name, project_root=ROOT):
    path = Path(project_root) / OUTPUT_DIR / f"{kind}_{name}.json"
    if path.exists():
        raise FileExistsError(f"Test evaluation already done (one-shot): {path}")
    return path


def test_image_ids(project_root=ROOT):
    ids = (Path(project_root) / TEST_SPLIT_FILE).read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate test image ids")
    return ids


def environment():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


# ------------------------------------------------------------------ evaluations
def load_classifier(run_dir, *, project_root=ROOT, device=None):
    """best.pth + config.json of a classifier run (config and manifest consistency checked)."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True)
    if json.dumps(checkpoint["config"], sort_keys=True) != json.dumps(config, sort_keys=True):
        raise ValueError("Checkpoint configuration differs from config.json")
    manifest = Path(config["manifest_path"])
    if not manifest.is_file():
        manifest = Path(project_root) / "data/processed/patches_full/patches.csv"
    if file_sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("Manifest SHA-256 differs from the training run")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(config["model_name"], num_classes=config["num_classes"], pretrained=False,
                         training_mode=config.get("training_mode", "full"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, checkpoint, manifest, device


@torch.no_grad()
def predict_split(model, config, manifest, split, device, *, project_root=ROOT, num_workers=4):
    """Deterministic predictions on one manifest split, in manifest row order."""
    dataset = DamagePatchDataset(manifest, split=split, project_root=project_root,
                                 transform=deterministic_transform(config["input_size"]))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
    amp = config["amp"] and device.type == "cuda"
    labels, predictions, probabilities = [], [], []
    started = time.perf_counter()
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(batch["pre"].to(device), batch["post"].to(device))
        probs = torch.softmax(logits.float(), dim=1).cpu()
        labels.extend(batch["label"].tolist())
        predictions.extend(probs.argmax(1).tolist())
        probabilities.extend(probs.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return dataset.df.reset_index(drop=True), labels, predictions, probabilities, time.perf_counter() - started, amp


PREDICTION_COLUMNS = ["image_id", "uid", "disaster", "holdout_source", "label", "pred", "p0", "p1", "p2", "p3"]


def write_predictions(path, frame, labels, predictions, probabilities):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        for (_, row), label, pred, probs in zip(frame.iterrows(), labels, predictions, probabilities):
            holdout = row.get("holdout_source", "")
            writer.writerow([row["image_id"], row["uid"], row.get("disaster", ""),
                             "" if not isinstance(holdout, str) else holdout, label, pred,
                             *[f"{p:.5f}" for p in probs]])


def evaluate_classifier(run, *, project_root=ROOT, num_workers=4, check_git=True, device=None):
    commit, registered = protocol_commit(project_root, check_git=check_git)
    run_dir = Path(project_root) / "results" / run
    out = output_path("classifier", run, project_root)
    sha = verify_checkpoint(run, run_dir / "best.pth", registered)
    model, config, checkpoint, manifest, device = load_classifier(run_dir, project_root=project_root, device=device)
    frame, labels, predictions, probabilities, seconds, amp = predict_split(
        model, config, manifest, "test", device, project_root=project_root, num_workers=num_workers)
    metrics = classification_metrics(labels, predictions)
    report = dict(
        kind="classifier", run=run, model_name=config["model_name"], evaluation_split="test", test_used=True,
        protocol=PROTOCOL, protocol_commit=commit, best_pth_sha256=sha, best_epoch=checkpoint["epoch"],
        val_macro_f1=checkpoint["best_val_macro_f1"], test_samples=len(frame), label_map=LABEL_MAP,
        manifest=str(manifest), manifest_sha256=config["manifest_sha256"],
        test=metrics, damage_f1_harmonic=harmonic_mean(metrics["class_f1"]),
        parameter_count=sum(p.numel() for p in model.parameters()),
        inference=dict(seconds=seconds, pairs_per_second=len(frame) / seconds, batch_size=config["batch_size"],
                       amp=amp, num_workers=num_workers, includes_data_loading=True),
        training_time_seconds=json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["training_time_seconds"],
        environment=environment())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(out.with_suffix(".csv"), frame, labels, predictions, probabilities)
    write_json(out, report)
    return report


def evaluate_segmentation(run, *, project_root=ROOT, num_workers=4, check_git=True, device=None):
    commit, registered = protocol_commit(project_root, check_git=check_git)
    run_dir = Path(project_root) / "results" / run
    out = output_path("segmentation", run, project_root)
    sha = verify_checkpoint(run, run_dir / "best.pth", registered)
    checkpoint = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_unet(config["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    image_ids = test_image_ids(project_root)
    started = time.perf_counter()
    metrics, per_image = evaluate_full_images(model, image_ids, device, project_root=project_root,
                                              image_size=config["image_size"], threshold=config["threshold"],
                                              num_workers=num_workers, amp=config["amp"], source="test")
    seconds = time.perf_counter() - started
    report = dict(kind="segmentation", run=run, evaluation_split="test", test_used=True, protocol=PROTOCOL,
                  protocol_commit=commit, best_pth_sha256=sha, best_epoch=checkpoint["epoch"],
                  val_dice=checkpoint["best_val_dice"], test_images=len(image_ids), test=metrics,
                  inference=dict(seconds=seconds, seconds_per_image=seconds / len(image_ids),
                                 includes_data_loading=True),
                  environment=environment())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "gt_pixels", "tp", "fp", "fn", "dice"])
        writer.writeheader()
        writer.writerows(sorted(per_image, key=lambda row: row["image_id"]))
    write_json(out, report)
    return report


def evaluate_two_stage(*, threshold, split_erosion, min_area=MIN_AREA, project_root=ROOT, check_git=True,
                       device=None):
    commit, registered = protocol_commit(project_root, check_git=check_git)
    name = f"t{threshold:g}_e{split_erosion}".replace(".", "")
    out = output_path("two_stage", name, project_root)
    unet_run, classifier_run = Path(DEFAULT_UNET).parent.name, Path(DEFAULT_CLASSIFIER).parent.name
    shas = {run: verify_checkpoint(run, Path(project_root) / "results" / run / "best.pth", registered)
            for run in (unet_run, classifier_run)}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_ids = test_image_ids(project_root)
    started = time.perf_counter()
    report, frame = evaluate_images(device, image_ids, split="test", project_root=project_root,
                                    threshold=threshold, min_area=min_area, split_erosion=split_erosion)
    seconds = time.perf_counter() - started
    segmentation = Path(project_root) / OUTPUT_DIR / f"segmentation_{unet_run}.json"
    xview2 = None
    if segmentation.is_file():
        f1_loc = json.loads(segmentation.read_text(encoding="utf-8"))["test"]["dice"]
        xview2 = dict(f1_localization_pixel=f1_loc,
                      end_to_end=0.3 * f1_loc + 0.7 * report["end_to_end_missed_as_false_negative"]["damage_f1_harmonic"],
                      oracle=0.3 * f1_loc + 0.7 * report["oracle_damage_f1_harmonic"],
                      note="xView2-STYLE, building-level damage F1; not the official pixel-level xView2 score")
    report.update(kind="two_stage", protocol=PROTOCOL, protocol_commit=commit, best_pth_sha256=shas,
                  xview2_style=xview2, inference=dict(seconds=seconds, seconds_per_image=seconds / len(image_ids),
                                                      includes_oracle_pass=True),
                  environment=environment())
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    write_json(out, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("classifier", "segmentation"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", required=True)
        p.add_argument("--num-workers", type=int, default=4)
    p = sub.add_parser("two-stage")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--split-erosion", type=int, default=4)
    p.add_argument("--min-area", type=int, default=MIN_AREA)
    args = parser.parse_args(argv)
    if args.command == "classifier":
        report = evaluate_classifier(args.experiment, num_workers=args.num_workers)
        summary = dict(macro_f1=report["test"]["macro_f1"], class_f1=report["test"]["class_f1"])
    elif args.command == "segmentation":
        report = evaluate_segmentation(args.experiment, num_workers=args.num_workers)
        summary = {k: report["test"][k] for k in ("dice", "iou", "precision", "recall")}
    else:
        report = evaluate_two_stage(threshold=args.threshold, split_erosion=args.split_erosion, min_area=args.min_area)
        summary = dict(localization=report["localization"], detection_rate=report["detection_rate"],
                       xview2_style=report["xview2_style"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
