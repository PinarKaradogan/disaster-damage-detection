from pathlib import Path
import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.data.constants import LABEL_MAP


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = {
    "image_id",
    "uid",
    "damage",
    "split",
    "pre_patch",
    "post_patch",
}


class PairedRandomGeometric:
    """
    Apply the SAME random geometric augmentation parameters
    to PRE and POST images.

    No brightness/color augmentation is used here.
    """

    def __init__(
        self,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5,
        rotation_degrees=15,
    ):
        self.horizontal_flip_p = horizontal_flip_p
        self.vertical_flip_p = vertical_flip_p
        self.rotation_degrees = rotation_degrees

    def __call__(self, pre, post):
        do_hflip = random.random() < self.horizontal_flip_p
        do_vflip = random.random() < self.vertical_flip_p

        angle = random.uniform(
            -self.rotation_degrees,
            self.rotation_degrees,
        )

        if do_hflip:
            pre = TF.hflip(pre)
            post = TF.hflip(post)

        if do_vflip:
            pre = TF.vflip(pre)
            post = TF.vflip(post)

        pre = TF.rotate(
            pre,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        post = TF.rotate(
            post,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        return pre, post


class DamagePatchDataset(Dataset):
    """
    Dataset for paired xBD PRE/POST building patches.

    The dataset itself does NOT hard-code resize or normalization.

    paired_transform:
        Used for geometric augmentation that must be identical
        for PRE and POST.

    transform:
        Used afterward for deterministic operations such as
        Resize, ToTensor and Normalize.

    Example:
        dataset = DamagePatchDataset(
            manifest_path="data/processed/patches_full/patches.csv",
            split="train",
            paired_transform=PairedRandomGeometric(),
            transform=my_transform,
        )
    """

    def __init__(
        self,
        manifest_path,
        split=None,
        paired_transform=None,
        transform=None,
        project_root=ROOT,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split is required and must be one of: train, val, test"
            )

        self.project_root = Path(project_root).resolve()

        manifest_path = Path(manifest_path)

        if not manifest_path.is_absolute():
            manifest_path = self.project_root / manifest_path

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        df = pd.read_csv(manifest_path)

        missing = REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"Manifest missing required columns: {sorted(missing)}"
            )

        df = df[df["split"] == split].copy()

        if df.empty:
            raise ValueError(
                f"No samples found for split={split!r}"
            )

        invalid_labels = sorted(
            set(df["damage"].dropna()) - set(LABEL_MAP)
        )

        if invalid_labels:
            raise ValueError(
                f"Unsupported damage labels: {invalid_labels}"
            )

        if df["damage"].isna().any():
            raise ValueError(
                "Manifest contains missing damage labels"
            )

        if df.duplicated(["split", "image_id", "uid"]).any():
            raise ValueError(
                "Manifest contains duplicate split/image_id/uid keys"
            )

        self.df = df.reset_index(drop=True)

        self.paired_transform = paired_transform
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_patch_path(self, relative_path):
        path = Path(relative_path)

        if path.is_absolute():
            raise ValueError(
                f"Patch path must be project-relative: {relative_path}"
            )

        full_path = (self.project_root / path).resolve()

        try:
            full_path.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(
                f"Patch path escapes project root: {relative_path}"
            ) from exc

        if not full_path.is_file():
            raise FileNotFoundError(
                f"Patch not found: {full_path}"
            )

        return full_path

    def __getitem__(self, index):
        row = self.df.iloc[index]

        pre_path = self._resolve_patch_path(
            row["pre_patch"]
        )

        post_path = self._resolve_patch_path(
            row["post_patch"]
        )

        with Image.open(pre_path) as image:
            pre = image.convert("RGB")

        with Image.open(post_path) as image:
            post = image.convert("RGB")

        if pre.size != post.size:
            raise ValueError(
                "PRE/POST patch dimensions differ for "
                f"image_id={row['image_id']} uid={row['uid']}: "
                f"{pre.size} vs {post.size}"
            )

        if self.paired_transform is not None:
            pre, post = self.paired_transform(
                pre,
                post,
            )

        if self.transform is not None:
            pre = self.transform(pre)
            post = self.transform(post)
        else:
            pre = TF.to_tensor(pre)
            post = TF.to_tensor(post)

        label = torch.tensor(
            LABEL_MAP[row["damage"]],
            dtype=torch.long,
        )

        return {
            "pre": pre,
            "post": post,
            "label": label,
            "image_id": row["image_id"],
            "uid": row["uid"],
        }
