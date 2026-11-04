"""Small CPU/manifest checks; no xBD or model training."""
import json

import pandas as pd
import pytest

from src.preprocessing.subset_manifest import allocate_counts, create_subset, select_train
from src.training.artifacts import file_sha256
from src.training.config import TrainingConfig
from src.training.data import build_loaders
from test_training_data import synthetic_manifest


@pytest.fixture
def source(synthetic_manifest):
    root, manifest = synthetic_manifest
    original = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    train = original[original["split"].eq("train")]
    copies = []
    for index in range(8):
        group = train.copy()
        group["uid"] = group["uid"] + f"-{index}"
        group["disaster"] = ["a", "a", "b", "b"]
        copies.append(group)
    rare = copies[0].iloc[[0]].copy()
    rare["disaster"], rare["image_id"], rare["uid"] = "rare", "rare_image", "rare_uid"
    other = original[~original["split"].eq("train")].copy()
    other["disaster"] = "a"
    frame = pd.concat([*copies, rare, other], ignore_index=True)
    frame["coordinate"] = "000.123400"
    frame.to_csv(manifest, index=False)
    return root, manifest, frame


def test_exact_deterministic_subset_and_loader_compatibility(source, tmp_path):
    root, manifest, frame = source
    before = manifest.read_bytes()
    output = tmp_path / "subset/patches.csv"
    report = create_subset(manifest, output, target=9)
    subset = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert subset["split"].value_counts().to_dict() == {"train": 9, "val": 4}
    pd.testing.assert_frame_equal(subset[subset["split"].eq("val")].reset_index(drop=True),
                                  frame[frame["split"].eq("val")].reset_index(drop=True))
    selected = subset[subset["split"].eq("train")]
    assert selected["disaster"].eq("rare").sum() == 1
    keys = ["image_id", "uid", "split"]
    joined = selected.merge(frame, on=keys, suffixes=("", "_source"), validate="one_to_one")
    for column in ["pre_patch", "post_patch", "coordinate", "damage", "disaster"]:
        assert joined[column].equals(joined[column + "_source"])
    assert manifest.read_bytes() == before
    assert report["source_sha256"] == file_sha256(manifest)
    assert report["manifest_sha256"] == file_sha256(output)
    assert report["subset_train_val_image_overlap"] == 0
    assert not any(report["source_image_overlaps"].values())
    assert all(row["subset_train"] >= 1 for row in report["strata"])
    images = pd.read_csv(output.parent / "image_distribution.csv")
    assert images["subset_train_patches"].sum() == 9
    assert images["source_train_patches"].sum() == 33
    assert report == json.loads((output.parent / "summary.json").read_text(encoding="utf-8"))
    saved = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.parent.iterdir()}
    assert create_subset(manifest, output, target=9) == report
    assert saved == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.parent.iterdir()}
    # Real Dataset __getitem__ and both loader variants work on unchanged image paths.
    for sampler in ["none", "weighted"]:
        loaders = build_loaders(TrainingConfig("resnet34_early_fusion", batch_size=2, input_size=16),
                                output, project_root=root, sampler=sampler)
        assert len(loaders.train.dataset) == 9 and len(loaders.val.dataset) == 4
        assert next(iter(loaders.train))["pre"].shape == (2, 3, 16, 16)
        assert next(iter(loaders.val))["post"].shape == (2, 3, 16, 16)


def test_seed_row_order_and_test_independence(source):
    _, _, frame = source
    train = frame[frame["split"].eq("train")]
    first, _ = select_train(train, target=9, random_state=42)
    again, _ = select_train(train.sample(frac=1, random_state=3), target=9, random_state=42)
    other, _ = select_train(train, target=9, random_state=7)
    keys = lambda f: set(zip(f["image_id"], f["uid"]))
    assert keys(first) == keys(again) and keys(first) != keys(other)
    with pytest.raises(ValueError, match="train rows only"):
        select_train(frame, target=9)


@pytest.mark.parametrize("sizes,target,expected", [
    ([1, 2, 97], 25, [1, 1, 23]), ([1, 1, 1], 3, [1, 1, 1]),
    ([4, 4, 4, 4], 5, [2, 1, 1, 1]), ([1, 2, 97], 100, [1, 2, 97]),
])
def test_rare_and_tied_quota_allocation(sizes, target, expected):
    assert allocate_counts(sizes, target).tolist() == expected


@pytest.mark.parametrize("sizes,target", [([1, 1, 20], 2), ([1, 2], 4), ([0, 2], 2), ([2], 0)])
def test_impossible_allocation_rejected(sizes, target):
    with pytest.raises(ValueError):
        allocate_counts(sizes, target)


def test_leakage_and_overwrite_protection(source, tmp_path):
    _, manifest, frame = source
    output = tmp_path / "subset/patches.csv"
    output.parent.mkdir()
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        create_subset(manifest, output, target=9)
    assert output.read_text() == "existing"
    frame.loc[frame["split"].eq("val"), "image_id"] = "train_0"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="split leakage"):
        create_subset(manifest, tmp_path / "leaky/patches.csv", target=9)


def test_changing_only_test_rows_does_not_change_selected_rows(source, tmp_path):
    _, manifest, frame = source
    first = tmp_path / "first/patches.csv"
    second = tmp_path / "second/patches.csv"
    create_subset(manifest, first, target=9)
    frame.loc[frame["split"].eq("test"), "damage"] = "not_used_for_sampling"
    frame.loc[frame["split"].eq("test"), "disaster"] = "not_used"
    frame.to_csv(manifest, index=False)
    create_subset(manifest, second, target=9)
    assert first.read_bytes() == second.read_bytes()
