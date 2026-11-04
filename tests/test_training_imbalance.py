import pytest
import torch
from types import SimpleNamespace
import pandas as pd

from src.training.losses import FocalLoss, make_criterion
from src.training.metrics import classification_metrics
from src.training.sampling import class_weights, make_weighted_sampler, train_labels


def test_known_metrics():
    result = classification_metrics([0, 1, 2, 3], [0, 1, 2, 2])
    assert result["class_f1"] == pytest.approx([1, 1, 2 / 3, 0])
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 0.75
    assert classification_metrics([0], [0])["macro_f1"] == 0.25


def test_class_weights_order_and_train_boundary():
    labels = torch.tensor([0] * 8 + [1] * 4 + [2] * 2 + [3])
    weights = class_weights(labels, split="train")
    assert torch.allclose(weights / weights[0], torch.tensor([1., 2., 4., 8.]))
    assert torch.isfinite(weights).all() and (weights > 0).all()
    assert weights.mean() == 1
    for split in ["val", "test"]:
        with pytest.raises(ValueError):
            class_weights(labels, split=split)
        with pytest.raises(ValueError):
            train_labels(SimpleNamespace(df=pd.DataFrame({"split": [split]})))
    with pytest.raises(ValueError):
        class_weights(torch.tensor([0, 1]), split="train")


def test_sampler_weights_and_seed():
    labels = torch.tensor([0, 0, 1, 2, 3])
    first = make_weighted_sampler(labels, split="train", seed=5)
    second = make_weighted_sampler(labels, split="train", seed=5)
    assert first.weights[2] == first.weights[0] * 2
    assert list(first) == list(second)
    assert first.replacement and first.num_samples == len(labels)


def test_focal_finite_backward_and_confidence():
    logits = torch.tensor([[1000., -1000., 0., 0.], [0., 0., 0., 0.]], requires_grad=True)
    loss = FocalLoss()(logits, torch.tensor([1, 2]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    target = torch.tensor([0])
    assert FocalLoss()(torch.tensor([[8., 0., 0., 0.]]), target) < FocalLoss()(torch.zeros(1, 4), target)
    assert FocalLoss(gamma=0)(logits, torch.tensor([1, 2])).item() == pytest.approx(
        torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).item())


def test_treatments_not_combined():
    with pytest.raises(ValueError):
        make_criterion("weighted_ce", "weighted", torch.ones(4))
    with pytest.raises(ValueError):
        make_criterion("focal", "weighted")
    assert isinstance(make_criterion("ce", "weighted"), torch.nn.CrossEntropyLoss)
