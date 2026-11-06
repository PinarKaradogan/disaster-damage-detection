"""Unseen-disaster manifest checks on a synthetic manifest and split files."""
import pandas as pd
import pytest

from src.preprocessing.unseen_manifest import build_unseen_manifest


def make(tmp_path, *, leak=False):
    rows = []
    for image_id, disaster, split in [("a_1", "a", "train"), ("a_2", "a", "train"), ("h_1", "h", "train"),
                                      ("h_2", "h", "test"), ("a_3", "a", "test")]:
        rows.append(dict(image_id=image_id, uid=image_id + "u", disaster=disaster, damage="no-damage",
                         split=split, pre_patch="x.png", post_patch="x.png"))
    source = tmp_path / "data/processed/patches_full/patches.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(source, index=False)
    split_dir = tmp_path / "splits/unseen_disaster/h"
    split_dir.mkdir(parents=True)
    (split_dir / "train_images.txt").write_text("a_1\n" + ("h_1\n" if leak else ""))
    (split_dir / "val_images.txt").write_text("a_2\n")
    (split_dir / "holdout_train_source_images.txt").write_text("" if leak else "h_1\n")
    (split_dir / "holdout_challenge_test_images.txt").write_text("h_2\n")
    return tmp_path


def test_relabels_rows_and_excludes_holdout_from_training(tmp_path):
    frame, report = build_unseen_manifest("h", project_root=make(tmp_path))
    by_id = frame.set_index("image_id")
    assert by_id.loc["a_1", "split"] == "train" and by_id.loc["a_2", "split"] == "val"
    assert by_id.loc["h_1", "split"] == "test" and by_id.loc["h_1", "holdout_source"] == "train_source"
    assert by_id.loc["h_2", "holdout_source"] == "challenge_test"
    assert "a_3" not in by_id.index                     # other disasters' test rows are dropped
    assert report["rows"] == {"train": 1, "val": 1, "test": 2}
    assert report["disasters_in_trainval"] == ["a"]


def test_holdout_disaster_in_training_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="rows found in train/val"):
        build_unseen_manifest("h", project_root=make(tmp_path, leak=True))
