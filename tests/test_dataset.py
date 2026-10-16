import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.constants import LABEL_MAP
from src.data.dataset import DamagePatchDataset, PairedRandomGeometric


def make_image(path: Path, width=37, height=29, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.arange(
        width * height * 3,
        dtype=np.uint16,
    ).reshape(height, width, 3)

    data = ((data + offset) % 255).astype(np.uint8)

    Image.fromarray(data, mode="RGB").save(path)


def manifest_row(
    image_id,
    uid,
    damage,
    split,
    pre_patch,
    post_patch,
):
    return {
        "image_id": image_id,
        "uid": uid,
        "damage": damage,
        "split": split,
        "pre_patch": pre_patch,
        "post_patch": post_patch,
    }


def make_project(tmp_path):
    root = tmp_path / "project"

    patch_dir = root / "data" / "processed" / "patches"

    rows = []

    samples = [
        (
            "train_a",
            "uid-a",
            "no-damage",
            "train",
        ),
        (
            "train_b",
            "uid-b",
            "minor-damage",
            "train",
        ),
        (
            "val_a",
            "uid-c",
            "major-damage",
            "val",
        ),
        (
            "test_a",
            "uid-d",
            "destroyed",
            "test",
        ),
    ]

    for i, (image_id, uid, damage, split) in enumerate(samples):
        pre_rel = (
            Path("data")
            / "processed"
            / "patches"
            / split
            / f"{image_id}__{uid}_pre.png"
        )

        post_rel = (
            Path("data")
            / "processed"
            / "patches"
            / split
            / f"{image_id}__{uid}_post.png"
        )

        make_image(
            root / pre_rel,
            width=37,
            height=29,
            offset=i,
        )

        make_image(
            root / post_rel,
            width=37,
            height=29,
            offset=i + 10,
        )

        rows.append(
            manifest_row(
                image_id=image_id,
                uid=uid,
                damage=damage,
                split=split,
                pre_patch=pre_rel.as_posix(),
                post_patch=post_rel.as_posix(),
            )
        )

    manifest = patch_dir / "patches.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        manifest,
        index=False,
    )

    return root, manifest


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path)


def test_label_map_is_fixed():
    assert LABEL_MAP == {
        "no-damage": 0,
        "minor-damage": 1,
        "major-damage": 2,
        "destroyed": 3,
    }


@pytest.mark.parametrize(
    "damage, expected",
    [
        ("no-damage", 0),
        ("minor-damage", 1),
        ("major-damage", 2),
        ("destroyed", 3),
    ],
)
def test_dataset_label_mapping(project, damage, expected):
    root, manifest = project

    df = pd.read_csv(manifest)
    row = df[df["damage"] == damage].copy()

    row.to_csv(
        manifest,
        index=False,
    )

    ds = DamagePatchDataset(
        manifest,
        split=row.iloc[0]["split"],
        project_root=root,
    )

    item = ds[0]

    assert item["label"].dtype == torch.int64
    assert item["label"].item() == expected


def test_dataset_returns_expected_structure(project):
    root, manifest = project

    ds = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    item = ds[0]

    assert set(item.keys()) == {
        "pre",
        "post",
        "label",
        "image_id",
        "uid",
    }

    assert isinstance(item["pre"], torch.Tensor)
    assert isinstance(item["post"], torch.Tensor)

    assert item["pre"].dtype == torch.float32
    assert item["post"].dtype == torch.float32

    assert item["label"].dtype == torch.int64


def test_split_filtering(project):
    root, manifest = project

    train = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    val = DamagePatchDataset(
        manifest,
        split="val",
        project_root=root,
    )

    test = DamagePatchDataset(
        manifest,
        split="test",
        project_root=root,
    )

    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1


def test_dataset_does_not_hardcode_resize(project):
    root, manifest = project

    ds = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    item = ds[0]

    assert item["pre"].shape == (3, 29, 37)
    assert item["post"].shape == (3, 29, 37)


def test_external_resize_transform(project):
    root, manifest = project

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    ds = DamagePatchDataset(
        manifest,
        split="train",
        transform=transform,
        project_root=root,
    )

    item = ds[0]

    assert item["pre"].shape == (3, 128, 128)
    assert item["post"].shape == (3, 128, 128)


def test_paired_augmentation_uses_same_geometry():
    random.seed(42)

    data = np.arange(
        32 * 32 * 3,
        dtype=np.uint16,
    ).reshape(32, 32, 3)

    image = Image.fromarray(
        (data % 255).astype(np.uint8),
        mode="RGB",
    )

    augmentation = PairedRandomGeometric(
        horizontal_flip_p=1.0,
        vertical_flip_p=1.0,
        rotation_degrees=20,
    )

    pre, post = augmentation(
        image,
        image.copy(),
    )

    np.testing.assert_array_equal(
        np.asarray(pre),
        np.asarray(post),
    )


def test_dataloader_batch_shape(project):
    root, manifest = project

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    ds = DamagePatchDataset(
        manifest,
        split="train",
        paired_transform=PairedRandomGeometric(
            horizontal_flip_p=0.5,
            vertical_flip_p=0.5,
            rotation_degrees=15,
        ),
        transform=transform,
        project_root=root,
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["pre"].shape == (
        2,
        3,
        128,
        128,
    )

    assert batch["post"].shape == (
        2,
        3,
        128,
        128,
    )

    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.int64


def test_invalid_damage_label_rejected(project):
    root, manifest = project

    df = pd.read_csv(manifest)
    df.loc[0, "damage"] = "un-classified"

    df.to_csv(
        manifest,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported damage labels",
    ):
        DamagePatchDataset(
            manifest,
            split="train",
            project_root=root,
        )


def test_duplicate_manifest_key_rejected(project):
    root, manifest = project

    df = pd.read_csv(manifest)

    df = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    df.to_csv(
        manifest,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        DamagePatchDataset(
            manifest,
            split="train",
            project_root=root,
        )


def test_missing_patch_rejected_when_sample_loaded(project):
    root, manifest = project

    ds = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    first = ds.df.iloc[0]

    (root / first["pre_patch"]).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Patch not found",
    ):
        ds[0]


def test_absolute_patch_path_rejected(project):
    root, manifest = project

    df = pd.read_csv(manifest)

    df.loc[0, "pre_patch"] = str(
        (root / "absolute.png").resolve()
    )

    df.to_csv(
        manifest,
        index=False,
    )

    ds = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    with pytest.raises(
        ValueError,
        match="project-relative",
    ):
        ds[0]


def test_patch_path_cannot_escape_project_root(project):
    root, manifest = project

    df = pd.read_csv(manifest)

    df.loc[0, "pre_patch"] = "../escape.png"

    df.to_csv(
        manifest,
        index=False,
    )

    ds = DamagePatchDataset(
        manifest,
        split="train",
        project_root=root,
    )

    with pytest.raises(
        ValueError,
        match="escapes project root",
    ):
        ds[0]


def test_invalid_split_rejected(project):
    root, manifest = project

    with pytest.raises(
        ValueError,
        match="split is required and must be one of: train, val, test",
    ):
        DamagePatchDataset(
            manifest,
            split="validation",
            project_root=root,
        )


def test_missing_split_rejected(project):
    root, manifest = project

    with pytest.raises(
        ValueError,
        match="^split is required and must be one of: train, val, test$",
    ):
        DamagePatchDataset(
            manifest,
            project_root=root,
        )
