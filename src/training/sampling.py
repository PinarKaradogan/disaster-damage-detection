"""Imbalance statistics have an explicit train-split boundary."""
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.constants import LABEL_MAP


def train_labels(dataset):
    if dataset.df.empty or not dataset.df["split"].eq("train").all():
        raise ValueError("Imbalance weights require only the train split")
    labels = torch.tensor(dataset.df["damage"].map(LABEL_MAP).tolist(), dtype=torch.long)
    return labels


def class_weights(labels, *, split):
    """w_c = (1 / count_c) / mean(1 / count); order is LABEL_MAP 0..3.

    Missing train classes are rejected rather than silently inventing weights.
    """
    if split != "train":
        raise ValueError("Weights must be calculated from train labels only")
    labels = torch.as_tensor(labels)
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
        raise ValueError("Expected nonempty 1D int64 train labels")
    if (labels < 0).any() or (labels > 3).any():
        raise ValueError("Labels must be in [0, 3]")
    counts = torch.bincount(labels, minlength=4).float()
    if (counts == 0).any():
        raise ValueError("All four classes must be present in train labels")
    inverse = counts.reciprocal()
    return inverse / inverse.mean()


def make_weighted_sampler(labels, *, split, seed=42):
    weights = class_weights(labels, split=split)
    return WeightedRandomSampler(
        weights[torch.as_tensor(labels)], num_samples=len(labels), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
