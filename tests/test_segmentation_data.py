"""Phase 5 segmentation data checks on tiny synthetic images; no xBD data needed."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.segmentation_dataset import (
    IMAGENET_MEAN, IMAGENET_STD, BuildingSegmentationDataset, paired_geometric, read_split_ids,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE, TILE = 64, 32


def make_project(tmp_path, ids_by_split, *, mask_values=(0, 1)):
    images, targets = tmp_path / "data/train/images", tmp_path / "data/train/targets"
    images.mkdir(parents=True)
    targets.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for ids in ids_by_split.values():
        for image_id in ids:
            mask = rng.integers(0, 2, (SIZE, SIZE), dtype=np.uint8) * mask_values[1]
            image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            image[..., 0] = mask * 255          # red channel encodes the mask
            image[..., 1] = np.arange(SIZE, dtype=np.uint8)[None, :] * 3  # column gradient
            Image.fromarray(image).save(images / f"{image_id}_pre_disaster.png")
            Image.fromarray(mask).save(targets / f"{image_id}_pre_disaster_target.png")
    split_dir = tmp_path / "splits/standard"
    split_dir.mkdir(parents=True)
    for split, ids in ids_by_split.items():
        (split_dir / f"{split}_images.txt").write_text("\n".join(ids) + "\n")
    return tmp_path


def red_from_normalized(image):
    return image[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path, {"train": ["a_00000001", "a_00000002"], "val": ["b_00000003"]})


def test_val_tiles_are_deterministic_grid_and_cover_image(project):
    ds = BuildingSegmentationDataset("val", project_root=project, tile_size=TILE, image_size=SIZE)
    assert len(ds) == 4 and not ds.augment
    origins = sorted((ds[i]["top"], ds[i]["left"]) for i in range(len(ds)))
    assert origins == [(0, 0), (0, 32), (32, 0), (32, 32)]
    first, again = ds[1], ds[1]
    assert torch.equal(first["image"], again["image"]) and torch.equal(first["mask"], again["mask"])
    mask = np.array(Image.open(project / "data/train/targets/b_00000003_pre_disaster_target.png"))
    assert torch.equal(first["mask"][0], torch.from_numpy(mask[0:32, 32:64]).float())


def test_shapes_dtypes_binary_mask_and_normalization(project):
    sample = BuildingSegmentationDataset("val", project_root=project, tile_size=TILE, image_size=SIZE)[0]
    assert sample["image"].shape == (3, TILE, TILE) and sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (1, TILE, TILE) and set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
    # Normalized red channel decodes back to the stored 0/255 values exactly.
    assert torch.allclose(red_from_normalized(sample["image"]), sample["mask"][0], atol=1e-5)


def test_train_augmentation_moves_image_and_mask_together(project):
    random.seed(1)
    ds = BuildingSegmentationDataset("train", project_root=project, tile_size=TILE, image_size=SIZE)
    assert ds.augment and len(ds) == 2 * 4
    for index in range(len(ds)):
        sample = ds[index]
        assert torch.allclose(red_from_normalized(sample["image"]), sample["mask"][0], atol=1e-5)
        assert 0 <= sample["top"] <= SIZE - TILE and 0 <= sample["left"] <= SIZE - TILE


def test_paired_geometric_is_identical_and_exact():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = image[..., 0].copy()
    seen = set()
    for seed in range(40):
        out_image, out_mask = paired_geometric(image, mask, random.Random(seed))
        assert np.array_equal(out_image[..., 0], out_mask)
        seen.add(out_mask.tobytes())
    assert len(seen) == 8  # 4 rotations x 2 flips, all exact (no interpolation)


def test_test_split_and_bad_inputs_rejected(project):
    with pytest.raises(ValueError, match="train/val"):
        BuildingSegmentationDataset("test", project_root=project, tile_size=TILE, image_size=SIZE)
    with pytest.raises(ValueError, match="multiple"):
        BuildingSegmentationDataset("val", project_root=project, tile_size=48, image_size=SIZE)
    wrong = BuildingSegmentationDataset("val", project_root=project, tile_size=TILE, image_size=128)
    with pytest.raises(ValueError, match="expected 128x128"):
        wrong[0]


def test_nonbinary_mask_rejected(tmp_path):
    project = make_project(tmp_path, {"train": ["a_00000001"], "val": ["b_00000002"]}, mask_values=(0, 2))
    with pytest.raises(ValueError, match="binary"):
        BuildingSegmentationDataset("val", project_root=project, tile_size=TILE, image_size=SIZE)[0]


def test_repository_splits_are_disjoint_image_level():
    # Uses the committed split files, so it also runs in CI without xBD data.
    train = set(read_split_ids("train", project_root=ROOT))
    val = set(read_split_ids("val", project_root=ROOT))
    test = set((ROOT / "splits/standard/test_images.txt").read_text().split())
    assert len(train) == 2380 and len(val) == 419 and len(test) == 933
    assert not (train & val) and not (train & test) and not (val & test)
