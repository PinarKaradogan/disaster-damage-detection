"""Phase 5 building-localization data: PRE image -> binary building mask.

Source: raw xBD images ``data/train/images/<id>_pre_disaster.png`` and the official
``data/train/targets/<id>_pre_disaster_target.png`` masks (0 background, 1 building;
un-classified buildings are buildings for localization). Splits come from the same
image-level files as the classifiers (``splits/standard``), so no validation image of
the classifiers is ever a segmentation training image. Only train/val are supported
in Phase 5; the test split stays untouched.

Train: each image yields ``tiles_per_image`` samples, each a random ``tile_size`` crop
plus random rot90 / horizontal flip applied identically to image and mask (exact for
masks, no interpolation). Val: deterministic non-overlapping grid tiles.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = {"train", "val"}


def read_split_ids(split, *, project_root=ROOT, split_dir="splits/standard"):
    if split not in SPLITS:
        raise ValueError("Phase 5 segmentation supports only train/val (test stays untouched)")
    path = Path(project_root) / split_dir / f"{split}_images.txt"
    ids = path.read_text(encoding="utf-8").split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: empty or duplicate image ids")
    return ids


def source_paths(image_id, *, project_root=ROOT, source="train"):
    # Train and val images both come from the Challenge train set folder ("train");
    # only the Phase 6 one-shot test evaluation reads "test".
    if source not in {"train", "test"}:
        raise ValueError("source must be train or test")
    base = Path(project_root) / "data" / source
    return (base / "images" / f"{image_id}_pre_disaster.png",
            base / "targets" / f"{image_id}_pre_disaster_target.png")


def load_pair(image_id, *, project_root=ROOT, image_size=1024, source="train"):
    image_path, mask_path = source_paths(image_id, project_root=project_root, source=source)
    with Image.open(image_path) as handle:
        image = np.array(handle.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as handle:
        mask = np.array(handle, dtype=np.uint8)
    if image.shape != (image_size, image_size, 3) or mask.shape != (image_size, image_size):
        raise ValueError(f"{image_id}: expected {image_size}x{image_size} image and mask, "
                         f"got {image.shape} / {mask.shape}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{image_id}: pre-disaster target must be binary 0/1")
    return image, mask


def normalize(image_uint8):
    """HWC uint8 -> CHW float tensor with ImageNet mean/std (same as the classifiers)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image_uint8)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def paired_geometric(image, mask, rng):
    """Same rot90 + horizontal flip for HWC image and HW mask."""
    k = rng.randrange(4)
    image, mask = np.rot90(image, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return image, mask


class BuildingSegmentationDataset(Dataset):
    def __init__(self, split, *, project_root=ROOT, tile_size=512, image_size=1024,
                 tiles_per_image=4, augment=None, seed=42, image_ids=None):
        if image_size % tile_size:
            raise ValueError("image_size must be a multiple of tile_size")
        self.split = split
        self.project_root = Path(project_root)
        self.tile_size, self.image_size = tile_size, image_size
        self.image_ids = list(image_ids) if image_ids is not None else read_split_ids(
            split, project_root=project_root)
        if split not in SPLITS:
            raise ValueError("Phase 5 segmentation supports only train/val (test stays untouched)")
        self.augment = (split == "train") if augment is None else augment
        grid = image_size // tile_size
        # Val tiles are a fixed grid; train uses the same count of random crops per image.
        self.tiles_per_image = tiles_per_image if self.augment else grid * grid
        self.grid = grid
        self.seed = seed

    def __len__(self):
        return len(self.image_ids) * self.tiles_per_image

    def tile_origin(self, index, rng):
        tile = index % self.tiles_per_image
        if self.augment:
            limit = self.image_size - self.tile_size
            return rng.randint(0, limit), rng.randint(0, limit)
        row, col = divmod(tile, self.grid)
        return row * self.tile_size, col * self.tile_size

    def __getitem__(self, index):
        image_id = self.image_ids[index // self.tiles_per_image]
        image, mask = load_pair(image_id, project_root=self.project_root, image_size=self.image_size)
        # Train draws come from the worker-seeded global RNG; val is fully deterministic.
        rng = random.Random() if self.augment else random.Random(self.seed + index)
        if self.augment:
            rng.seed(random.getrandbits(64))
        top, left = self.tile_origin(index, rng)
        size = self.tile_size
        image, mask = image[top:top + size, left:left + size], mask[top:top + size, left:left + size]
        if self.augment:
            image, mask = paired_geometric(image, mask, rng)
        return {
            "image": normalize(image),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
            "image_id": image_id, "top": top, "left": left,
        }
