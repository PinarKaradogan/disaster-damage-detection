"""Phase 6 per-disaster analysis and Grad-CAM checks on synthetic data (CPU, no downloads)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.disaster_analysis import (
    DISASTER_TYPE, analyse_classifier, analyse_segmentation, group_metrics, markdown_table,
)
from src.evaluation.gradcam import GradCAM, select_samples
from src.models import create_model


def test_group_metrics_support_rule():
    labels = [0] * 40 + [1] * 35 + [2] * 5            # class 2 below 30, class 3 absent
    preds = [0] * 40 + [1] * 30 + [0] * 5 + [2] * 5
    row = group_metrics(labels, preds)
    assert row["buildings"] == 80 and row["support"]["major-damage"] == 5
    assert row["class_f1"]["major-damage"] == "n < 30" and row["class_f1"]["destroyed"] == "n < 30"
    assert row["classes_included"] == 2
    f1_no = 2 * 40 / (2 * 40 + 5)                        # 5 minor predicted as no-damage
    f1_minor = 2 * 30 / (2 * 30 + 5)
    assert row["present_class_macro_f1"] == pytest.approx((f1_no + f1_minor) / 2)


def test_small_group_reports_accuracy_only():
    row = group_metrics([0] * 10 + [3] * 5, [0] * 15)
    assert row["class_f1"] is None and row["present_class_macro_f1"] is None
    assert row["accuracy"] == pytest.approx(10 / 15) and "accuracy only" in row["note"]


def test_analyse_classifier_groups_by_disaster_type_and_holdout(tmp_path):
    rows = []
    for disaster, n in (("socal-fire", 40), ("santa-rosa-wildfire", 40), ("guatemala-volcano", 12)):
        for i in range(n):
            rows.append(dict(image_id=f"{disaster}_{i:08d}", uid=f"{disaster}{i}", disaster=disaster,
                             holdout_source="challenge_test" if i % 2 else "train_source",
                             label=i % 2, pred=0))
    path = tmp_path / "preds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    report = analyse_classifier(path)
    assert set(report["by_disaster"]) == {"socal-fire", "santa-rosa-wildfire", "guatemala-volcano"}
    assert report["by_type"]["fire"]["buildings"] == 80
    assert report["by_disaster"]["guatemala-volcano"]["class_f1"] is None
    assert set(report["by_holdout_source"]) == {"challenge_test", "train_source"}
    table = markdown_table(report)
    socal = next(line for line in table.splitlines() if line.startswith("| socal-fire |"))
    assert socal.startswith("| socal-fire | fire | 40 |")
    assert socal.count("n < 30") == 4                     # 20 buildings per class < 30
    assert next(l for l in table.splitlines() if l.startswith("| guatemala-volcano")).count("–") == 5


def test_unknown_disaster_is_rejected(tmp_path):
    path = tmp_path / "preds.csv"
    pd.DataFrame([dict(image_id="new-disaster_00000001", uid="u", disaster="new-disaster", label=0, pred=0)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="type mapping"):
        analyse_classifier(path)
    assert DISASTER_TYPE["hurricane-matthew"] == "wind"


def test_analyse_segmentation_sums_pixels_per_disaster(tmp_path):
    path = tmp_path / "per_image.csv"
    pd.DataFrame([dict(image_id="socal-fire_00000001", gt_pixels=10, tp=8, fp=2, fn=2, dice=0.8),
                  dict(image_id="socal-fire_00000002", gt_pixels=0, tp=0, fp=4, fn=0, dice=0.0),
                  dict(image_id="palu-tsunami_00000001", gt_pixels=5, tp=5, fp=0, fn=0, dice=1.0)]).to_csv(path, index=False)
    report = analyse_segmentation(path)
    assert report["by_disaster"]["socal-fire"]["dice"] == pytest.approx(16 / (16 + 6 + 2))
    assert report["by_disaster"]["socal-fire"]["images_with_buildings"] == 1
    assert report["by_type"]["tsunami"]["dice"] == 1.0


def test_select_samples_is_fixed_and_balanced():
    rng = np.random.RandomState(0)
    frame = pd.DataFrame(dict(image_id=[f"x_{i}" for i in range(400)], uid=[f"u{i:03d}" for i in range(400)],
                              label=rng.randint(0, 4, 400), pred=rng.randint(0, 4, 400)))
    first = select_samples(frame)
    again = select_samples(frame.sample(frac=1, random_state=3))     # input order must not matter
    assert list(first["uid"]) == list(again["uid"])
    assert len(first) == 32
    for label in range(4):
        group = first[first["label"] == label]
        assert group["correct"].sum() == 4 and (~group["correct"]).sum() == 4
        assert ((group["pred"] == label) == group["correct"]).all()


def test_gradcam_map_shape_range_and_target():
    torch.manual_seed(0)
    model = create_model("resnet34_early_fusion", pretrained=False)
    cam_tool = GradCAM(model, model.backbone.layer4)
    pre, post = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    cam, predicted, target = cam_tool(pre, post)
    assert cam.shape == (2, 64, 64) and torch.isfinite(cam).all()
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-6 and torch.equal(predicted, target)
    forced = torch.tensor([3, 1])
    _, _, used = cam_tool(pre, post, target=forced)
    assert torch.equal(used, forced)
