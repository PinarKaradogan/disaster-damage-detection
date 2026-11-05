"""Binary building-segmentation metrics (validation only).

Primary (model selection): GLOBAL pixel Dice = 2TP / (2TP + FP + FN), i.e. pixel F1,
accumulated over every validation pixel. IoU = TP / (TP + FP + FN).

Empty-mask policy (about 18% of xBD images contain no building):
- Global metrics: empty images only contribute FP pixels; if a metric's denominator is 0
  (no building and no prediction anywhere) Dice/IoU are defined as 1.0.
- Per-image Dice is averaged over images WITH buildings only (``per_image_dice_nonempty``).
- Images WITHOUT buildings are summarized separately as the fraction where the model
  predicts any building pixel (``empty_image_false_positive_rate``).
These numbers are not comparable with the four-class damage Macro-F1.
"""
from collections import defaultdict

import torch


def _ratio(numerator, denominator, empty_value):
    return float(numerator / denominator) if denominator else float(empty_value)


class SegmentationMeter:
    def __init__(self, threshold=0.5):
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_image = defaultdict(lambda: [0, 0, 0, 0])  # tp, fp, fn, gt pixels

    @torch.no_grad()
    def update(self, logits, masks, image_ids):
        if logits.shape != masks.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Expected matching [B, 1, H, W] logits and masks")
        predicted = torch.sigmoid(logits.float()) > self.threshold
        truth = masks > 0.5
        dims = (1, 2, 3)
        tp = (predicted & truth).sum(dims)
        fp = (predicted & ~truth).sum(dims)
        fn = (~predicted & truth).sum(dims)
        tn = (~predicted & ~truth).sum(dims)
        for index, image_id in enumerate(image_ids):
            counts = self.per_image[image_id]
            counts[0] += int(tp[index]); counts[1] += int(fp[index])
            counts[2] += int(fn[index]); counts[3] += int(tp[index] + fn[index])
        self.tp += int(tp.sum()); self.fp += int(fp.sum())
        self.fn += int(fn.sum()); self.tn += int(tn.sum())

    def compute(self):
        if not self.per_image:
            raise ValueError("No samples were accumulated")
        tp, fp, fn = self.tp, self.fp, self.fn
        nonempty = [2 * c[0] / (2 * c[0] + c[1] + c[2]) for c in self.per_image.values() if c[3] > 0]
        empty = [c[1] > 0 for c in self.per_image.values() if c[3] == 0]
        return {
            "dice": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
            "iou": _ratio(tp, tp + fp + fn, 1.0),
            "precision": _ratio(tp, tp + fp, 0.0),
            "recall": _ratio(tp, tp + fn, 0.0),
            "pixel_accuracy": _ratio(tp + self.tn, tp + fp + fn + self.tn, 1.0),
            "per_image_dice_nonempty": float(sum(nonempty) / len(nonempty)) if nonempty else None,
            "empty_image_false_positive_rate": float(sum(empty) / len(empty)) if empty else None,
            "images": len(self.per_image), "images_with_buildings": len(nonempty),
            "images_without_buildings": len(empty),
            "pixel_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": self.tn},
            "threshold": self.threshold,
        }
