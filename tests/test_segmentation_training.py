"""Phase 5 U-Net model, loss, metrics and training/resume checks on CPU; no downloads."""
import json

import pytest
import torch

from src.evaluation.segmentation_metrics import SegmentationMeter
from src.models.unet import create_unet
from src.train_segmentation import build_parser
from src.training.segmentation import BCEDiceLoss, SegmentationConfig, run_segmentation_experiment
from test_segmentation_data import make_project
from test_training_resume import equal_nested


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def logits_from(pred):
    return torch.where(pred.bool(), torch.tensor(10.0), torch.tensor(-10.0))


def test_meter_global_and_per_image_metrics():
    meter = SegmentationMeter()
    truth = torch.zeros(3, 1, 2, 2)
    pred = torch.zeros(3, 1, 2, 2)
    truth[0, 0, 0] = 1                # image a: 2 building pixels
    pred[0, 0, 0, 0] = 1              # 1 TP, 1 FN
    pred[0, 0, 1, 1] = 1              # 1 FP
    pred[1, 0, 0, 0] = 1              # image b (no buildings): 1 FP -> false-positive image
    # image c: no buildings, no prediction
    meter.update(logits_from(pred), truth, ["a", "b", "c"])
    result = meter.compute()
    assert result["pixel_counts"] == {"tp": 1, "fp": 2, "fn": 1, "tn": 8}
    assert result["dice"] == pytest.approx(2 * 1 / (2 * 1 + 2 + 1))
    assert result["iou"] == pytest.approx(1 / 4)
    assert result["per_image_dice_nonempty"] == pytest.approx(2 / 4)
    assert result["empty_image_false_positive_rate"] == pytest.approx(0.5)
    assert (result["images"], result["images_with_buildings"], result["images_without_buildings"]) == (3, 1, 2)


def test_meter_accumulates_tiles_of_the_same_image():
    meter = SegmentationMeter()
    truth = torch.ones(2, 1, 2, 2)
    meter.update(logits_from(truth), truth, ["x", "x"])
    result = meter.compute()
    assert result["images"] == 1 and result["dice"] == 1.0 and result["per_image_dice_nonempty"] == 1.0
    assert result["empty_image_false_positive_rate"] is None


def test_meter_all_empty_defines_perfect_dice():
    meter = SegmentationMeter()
    empty = torch.zeros(1, 1, 4, 4)
    meter.update(logits_from(empty), empty, ["e"])
    result = meter.compute()
    assert result["dice"] == 1.0 and result["iou"] == 1.0 and result["per_image_dice_nonempty"] is None
    with pytest.raises(ValueError):
        SegmentationMeter(threshold=1.5)


def test_bce_dice_loss_orders_predictions_and_has_gradients():
    loss = BCEDiceLoss()
    masks = (torch.rand(2, 1, 8, 8) > 0.5).float()
    good = loss(logits_from(masks), masks)
    bad = loss(logits_from(1 - masks), masks)
    assert good < 0.01 and bad > 1.5
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    loss(logits, masks).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_unet_output_shape_and_backward():
    model = create_unet(pretrained=False)
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 1, 64, 64) and torch.isfinite(logits).all()
    BCEDiceLoss()(logits, torch.zeros_like(logits)).backward()
    first = next(p for p in model.encoder.parameters() if p.grad is not None)
    assert torch.isfinite(first.grad).all()
    with pytest.raises(ValueError, match="Unknown"):
        create_unet("unet_vgg", pretrained=False)


def test_cli_defaults():
    args = build_parser().parse_args(["--experiment-name", "x"])
    assert (args.model, args.batch_size, args.num_workers, args.epochs) == ("unet_resnet34", 8, 4, 30)


def run(project, tmp_path, name, **kwargs):
    config = SegmentationConfig(image_size=64, tile_size=32, tiles_per_image=2, batch_size=2, num_workers=0,
                                epochs=3, amp=False, early_stopping_patience=10)
    return run_segmentation_experiment(config, experiment_name=name, project_root=project,
                                       results_root=tmp_path / "results", runs_root=tmp_path / "runs",
                                       device="cpu", pretrained=False, **kwargs)


def test_experiment_artifacts_resume_and_overwrite_protection(tmp_path):
    project = make_project(tmp_path / "project", {"train": ["a_00000001", "a_00000002"], "val": ["b_00000003"]})
    metrics = run(project, tmp_path, "full")
    output = tmp_path / "results/full"
    for file in ("best.pth", "last.pth", "config.json", "metrics.json", "history.csv", "status.json"):
        assert (output / file).is_file()
    config = json.loads((output / "config.json").read_text())
    assert config["task"] == "building_segmentation" and config["evaluation_split"] == "val"
    assert config["split_counts"] == {"train_images": 2, "val_images": 1, "train_tiles_per_epoch": 4, "val_tiles": 4}
    assert set(config["split_files_sha256"]) == {"train", "val"}
    assert metrics["status"] == "completed" and 0 <= metrics["best_val_dice"] <= 1
    assert metrics["val"]["images"] == 1
    header = (output / "history.csv").read_text().splitlines()[0]
    assert header == "epoch,train_loss,train_dice,val_loss,val_dice,val_iou,learning_rate"

    run(project, tmp_path, "split", session_epochs=1)
    assert json.loads((tmp_path / "results/split/status.json").read_text())["status"] == "paused"
    run(project, tmp_path, "split", resume=True)
    uninterrupted = torch.load(output / "last.pth", weights_only=True)
    resumed = torch.load(tmp_path / "results/split/last.pth", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "history",
                "early_stopping", "best_metrics"):
        equal_nested(uninterrupted[key], resumed[key])
    best = torch.load(output / "best.pth", weights_only=True)
    assert best["epoch"] == metrics["best_epoch"] and best["model_name"] == "unet_resnet34"

    with pytest.raises(FileExistsError):
        run(project, tmp_path, "full")
    with pytest.raises(ValueError, match="completed"):
        run(project, tmp_path, "full", resume=True)
