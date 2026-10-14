"""make_splits'in görüntü bazlı, sızıntısız ve tekrarlanabilir bölme ürettiğinin sentetik testi."""
import pandas as pd
import pytest

from src.preprocessing.make_splits import (check_no_leakage, filter_trainable, make_standard,
                                           make_unseen_disaster)

DAMAGES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def fake_buildings(source_split, disasters, n_images=40, n_buildings=5):
    """Her afet için n_images görüntü, her görüntüde n_buildings bina içeren sahte tablo."""
    rows = []
    for disaster in disasters:
        for i in range(n_images):
            image_id = f"{disaster}_{source_split}_{i:08d}"
            for j in range(n_buildings):
                rows.append({"source_split": source_split, "image_id": image_id, "disaster": disaster,
                             "damage": DAMAGES[(i + j) % 4], "uid": f"{image_id}-{j}",
                             "is_bad_polygon": False, "out_of_bounds": j == 0})
    return pd.DataFrame(rows)


@pytest.fixture
def tables():
    train = fake_buildings("train", ["event-a", "event-b", "event-c"])
    test = fake_buildings("test", ["event-a", "event-b"], n_images=10)
    # Eğitimden çıkarılması gereken satırlar
    train.loc[1, "damage"] = "un-classified"
    train.loc[2, "is_bad_polygon"] = True
    return train, test


def test_filter_trainable_drops_unclassified_and_bad_keeps_out_of_bounds(tables):
    train, _ = tables
    kept = filter_trainable(train)
    assert len(kept) == len(train) - 2
    assert "un-classified" not in set(kept["damage"]) and not kept["is_bad_polygon"].any()
    assert kept["out_of_bounds"].sum() == train["out_of_bounds"].sum()


def test_standard_splits_are_disjoint(tables):
    splits = make_standard(*tables)
    check_no_leakage(splits)
    ids = {name: set(df["image_id"]) for name, df in splits.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]


def test_buildings_of_an_image_stay_together(tables):
    splits = make_standard(*tables)
    owner = pd.concat([df.assign(split=name) for name, df in splits.items()])
    assert (owner.groupby("image_id")["split"].nunique() == 1).all()


def test_val_fraction_is_per_disaster(tables):
    splits = make_standard(*tables)
    val_per_disaster = splits["val"].groupby("disaster")["image_id"].nunique()
    assert val_per_disaster.to_dict() == {"event-a": 6, "event-b": 6, "event-c": 6}  # round(40 * 0.15)


def test_same_seed_same_result(tables):
    first, second = make_standard(*tables), make_standard(*tables)
    for name in first:
        assert sorted(first[name]["image_id"].unique()) == sorted(second[name]["image_id"].unique())
    other = make_standard(*tables, seed=7)
    assert set(other["val"]["image_id"]) != set(first["val"]["image_id"])


def test_unseen_disaster_without_test_set(tables):
    train, _ = tables
    splits = make_unseen_disaster(train, "event-b")
    check_no_leakage(splits)
    assert set(splits) == {"train", "val", "holdout_train_source"}
    assert set(splits["holdout_train_source"]["disaster"]) == {"event-b"}
    assert "event-b" not in set(splits["train"]["disaster"]) | set(splits["val"]["disaster"])


def test_unseen_disaster_with_test_set(tables):
    train, test = tables
    splits = make_unseen_disaster(train, "event-b", test)
    check_no_leakage(splits)
    assert set(splits) == {"train", "val", "holdout_train_source", "holdout_challenge_test"}
    assert set(splits["holdout_challenge_test"]["source_split"]) == {"test"}
    assert set(splits["holdout_challenge_test"]["disaster"]) == {"event-b"}
    # Train/val, test seti verilse de verilmese de aynı kalır
    without_test = make_unseen_disaster(train, "event-b")
    for name in ["train", "val", "holdout_train_source"]:
        assert splits[name].equals(without_test[name])


def test_unseen_disaster_missing_from_test_set_has_no_holdout_challenge_test(tables):
    train, test = tables
    splits = make_unseen_disaster(train, "event-c", test)  # event-c test setinde yok
    assert "holdout_challenge_test" not in splits


def test_leakage_check_catches_shared_image(tables):
    splits = make_standard(*tables)
    leaked = splits["train"].head(1)
    splits["val"] = pd.concat([splits["val"], leaked])
    with pytest.raises(AssertionError):
        check_no_leakage(splits)
