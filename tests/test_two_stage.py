"""Phase 5.4 two-stage integration checks on synthetic data; no xBD data, no downloads."""
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.evaluation.two_stage import (
    DAMAGE_COLORS, components, crop_pairs, draw_damage, match_buildings, overlap_matrix, run_two_stage,
)
from src.preprocessing.extract_patches import patch_geometry


def test_components_bboxes_and_min_area():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:15, 10:30] = 0.9      # building A: rows 5..14, cols 10..29
    probability[40:50, 40:45] = 0.8     # building B
    probability[60, 60] = 0.99          # 1-pixel speckle -> dropped
    probability[20:22, 50:52] = 0.4     # below threshold -> ignored
    labels, boxes = components(probability, threshold=0.5, min_area=10)
    assert len(boxes) == 2 and labels.max() == 2
    a = next(b for b in boxes if b["area"] == 200)
    assert (a["x_min"], a["y_min"], a["x_max"], a["y_max"]) == (10.0, 5.0, 30.0, 15.0)
    assert labels[60, 60] == 0 and labels[21, 51] == 0
    assert {b["component"] for b in boxes} == {1, 2}
    with pytest.raises(ValueError):
        components(probability, threshold=1.0)


def test_diagonal_neighbours_stay_separate_with_4_connectivity():
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[0:3, 0:3] = 1
    probability[3:6, 3:6] = 1           # touches the first block only at a corner
    _, boxes = components(probability, min_area=1)
    assert len(boxes) == 2


def test_crop_pairs_use_phase3_rule_on_same_coordinates():
    pre = np.zeros((200, 200, 3), dtype=np.uint8)
    post = np.zeros((200, 200, 3), dtype=np.uint8)
    pre[100:110, 50:70] = (255, 0, 0)
    post[100:110, 50:70] = (0, 0, 255)
    box = dict(x_min=50.0, y_min=100.0, x_max=70.0, y_max=110.0)
    (pre_patch, post_patch, crop), = crop_pairs(Image.fromarray(pre), Image.fromarray(post), [box])
    assert crop == patch_geometry(50, 100, 70, 110, 20, 10)
    assert pre_patch.size == post_patch.size == (64, 64)   # side = max(64, 2 * 20)
    a, b = np.array(pre_patch), np.array(post_patch)
    assert np.array_equal(a[..., 0] > 0, b[..., 2] > 0)    # building at identical patch pixels
    assert (a[..., 0] > 0).sum() == 200


def test_match_buildings_localization_and_merge_counting():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[0:4, 0:4] = 1                    # matched well
    gt[10:14, 0:4] = 2                  # merged with 3 into one component
    gt[10:14, 5:9] = 3
    gt[16:20, 16:20] = 4                # missed
    pred = np.zeros_like(gt)
    pred[0:4, 0:4] = 1
    pred[10:14, 0:9] = 2                # one blob over buildings 2 and 3
    pred[0:3, 15:18] = 3                # false positive
    joint, pred_area, gt_area = overlap_matrix(pred, gt)
    assert joint[1, 1] == 16 and pred_area[2] == 36 and gt_area[4] == 16
    match = match_buildings(pred, gt)
    # IoU(1,1)=1; blob 2 vs building 2 = 16/36 < 0.5 -> not a TP
    assert (match["tp"], match["fp"], match["fn"]) == (1, 2, 3)
    assert match["assigned"] == {1: 1, 2: 2, 3: 2}      # buildings 2 and 3 take the blob's damage
    assert match["merged_components"] == 1 and match["buildings_in_merged"] == 2


def test_draw_damage_colours_component_pixels():
    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[4:8, 4:8] = 1
    out = np.array(draw_damage(image, labels, [dict(component=1, predicted_class=3, x_min=4, y_min=4,
                                                    x_max=8, y_max=8)], alpha=1.0))
    assert tuple(out[5, 5]) == DAMAGE_COLORS[3]
    assert tuple(out[12, 12]) == (0, 0, 0)


class FixedUNet(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("logits", torch.where(torch.as_tensor(mask), 8.0, -8.0)[None, None])

    def forward(self, images):
        return self.logits.expand(len(images), -1, -1, -1)


class BrightnessClassifier(nn.Module):
    """Predicts 'destroyed' when POST is darker than PRE inside the crop, else 'no-damage'."""

    def forward(self, pre, post):
        darker = (pre.mean((1, 2, 3)) - post.mean((1, 2, 3))) > 0.1
        logits = torch.zeros(len(pre), 4)
        logits[darker, 3] = 5
        logits[~darker, 0] = 5
        return logits


def test_run_two_stage_end_to_end_with_stub_models():
    pre = np.full((128, 128, 3), 200, dtype=np.uint8)
    post = pre.copy()
    mask = np.zeros((128, 128), dtype=bool)
    mask[10:30, 10:30] = True           # intact building
    mask[80:100, 80:110] = True         # destroyed building: dark in POST
    post[70:110, 70:120] = 20
    labels, detections = run_two_stage(Image.fromarray(pre), Image.fromarray(post), FixedUNet(mask),
                                       BrightnessClassifier(), torch.device("cpu"), input_size=32)
    assert len(detections) == 2 and labels.shape == (128, 128)
    by_position = {(d["x_min"], d["y_min"]): d["predicted_damage"] for d in detections}
    assert by_position == {(10.0, 10.0): "no-damage", (80.0, 80.0): "destroyed"}
    assert all(len(d["probabilities"]) == 4 for d in detections)
    with pytest.raises(ValueError, match="differ"):
        run_two_stage(Image.fromarray(pre), Image.fromarray(post[:64]), FixedUNet(mask),
                      BrightnessClassifier(), torch.device("cpu"))


def test_split_touching_separates_bridged_buildings_and_keeps_small_ones():
    from src.evaluation.two_stage import split_touching
    binary = np.zeros((40, 60), dtype=bool)
    binary[5:25, 5:25] = True           # building A
    binary[5:25, 35:55] = True          # building B
    binary[14:16, 25:35] = True         # 2-pixel bridge joining A and B
    binary[32:35, 2:5] = True           # small separate building (vanishes after erosion)
    labels, count = split_touching(binary, erosion=0)
    assert count == 2                    # A+B merged, plus the small one
    labels, count = split_touching(binary, erosion=2)
    assert count == 3
    assert labels[15, 10] != labels[15, 45] and labels[15, 10] > 0 and labels[15, 45] > 0
    assert labels[33, 3] > 0 and labels[33, 3] not in (labels[15, 10], labels[15, 45])
    assert np.array_equal(labels > 0, binary)            # no pixel gained or lost
    for value in range(1, count + 1):                     # each label is one connected piece
        from scipy import ndimage
        assert ndimage.label(labels == value)[1] == 1


def test_split_never_assigns_pixels_across_background():
    from src.evaluation.two_stage import split_touching
    binary = np.zeros((30, 30), dtype=bool)
    binary[2:22, 2:22] = True           # big building with a core
    binary[2:4, 25:28] = True           # thin building, no core, near the big one
    labels, count = split_touching(binary, erosion=3)
    assert count == 2 and labels[3, 26] != labels[10, 10]


def test_components_with_split_returns_separate_boxes():
    probability = np.zeros((40, 60), dtype=np.float32)
    probability[5:25, 5:25] = 0.9
    probability[5:25, 35:55] = 0.9
    probability[14:16, 25:35] = 0.9
    _, merged = components(probability, min_area=10)
    _, split = components(probability, min_area=10, split_erosion=2)
    assert len(merged) == 1 and len(split) == 2
