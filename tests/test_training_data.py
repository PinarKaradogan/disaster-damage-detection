from dataclasses import replace

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler

from src.data.constants import LABEL_MAP
from src.training.config import TrainingConfig
from src.training.data import build_loaders


@pytest.fixture
def synthetic_manifest(tmp_path):
    rows = []
    for split in ["train", "val", "test"]:
        for index, damage in enumerate(LABEL_MAP):
            filename = f"{split}_{index}.png"
            Image.new("RGB", (19, 23), (index * 40, 70, 120)).save(tmp_path / filename)
            rows.append(dict(image_id=f"{split}_{index}", uid=str(index), damage=damage,
                             split=split, pre_patch=filename, post_patch=filename))
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return tmp_path, path


@pytest.mark.parametrize("sampler", ["none", "weighted"])
def test_loaders_train_val_only(synthetic_manifest, sampler):
    root, manifest = synthetic_manifest
    config = replace(TrainingConfig("simple_cnn"), batch_size=2, smoke_batch_size=2)
    data = build_loaders(config, manifest, sampler=sampler, project_root=root)
    expected = RandomSampler if sampler == "none" else WeightedRandomSampler
    assert isinstance(data.train.sampler, expected)
    assert isinstance(data.val.sampler, SequentialSampler)
    assert data.train.dataset.paired_transform is not None
    assert data.val.dataset.paired_transform is None
    assert data.train_class_counts == [1, 1, 1, 1]
    assert data.split_counts == {"train": 4, "val": 4}
    batch = next(iter(data.val))
    assert batch["pre"].shape == (2, 3, 128, 128)
    assert torch.equal(batch["pre"], batch["post"])
    assert torch.equal(batch["pre"], next(iter(data.val))["pre"])
    assert batch["label"].dtype == torch.int64


def test_smoke_bounded_seeded(synthetic_manifest):
    root, manifest = synthetic_manifest
    config = replace(TrainingConfig("simple_cnn"), smoke_batch_size=1)
    one = build_loaders(config, manifest, sampler="weighted", smoke=True, project_root=root)
    two = build_loaders(config, manifest, sampler="weighted", smoke=True, project_root=root)
    assert len(one.train) == len(one.val) == 2
    assert one.train.dataset.indices == two.train.dataset.indices
    assert one.val.dataset.indices == two.val.dataset.indices
    assert torch.equal(one.class_weights, torch.ones(4))
