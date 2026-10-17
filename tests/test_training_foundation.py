from dataclasses import FrozenInstanceError
import os
import random

import numpy as np
import pytest
import torch

from src.training.config import TrainingConfig
from src.training.utils import get_device, set_training_seed


def test_config_defaults():
    config = TrainingConfig(model_name="simple_cnn")
    assert config.to_dict() == dict(
        model_name="simple_cnn", seed=42, input_size=128, num_classes=4,
        batch_size=32, smoke_batch_size=8, num_workers=0, epochs=30,
        early_stopping_patience=5, optimizer="adamw", learning_rate=3e-4,
        weight_decay=1e-4, scheduler="cosine", scheduler_eta_min=1e-6,
        amp=True,
    )
    with pytest.raises(TypeError):
        TrainingConfig()


def test_config_dict_is_independent():
    config = TrainingConfig("simple_cnn")
    values = config.to_dict()
    values["seed"] = 9
    assert config.seed == 42


def test_config_frozen():
    with pytest.raises(FrozenInstanceError):
        TrainingConfig("simple_cnn").seed = 1


def test_training_seed_reproducibility():
    def draw():
        return random.random(), np.random.rand(), torch.rand(3)

    set_training_seed(42)
    first = draw()
    set_training_seed(42)
    second = draw()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_device_selection(monkeypatch):
    for available, expected in [(False, "cpu"), (True, "cuda")]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        assert get_device() == torch.device(expected)
