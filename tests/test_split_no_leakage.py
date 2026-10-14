"""make_splits'in all-image, sızıntısız ve tekrarlanabilir bölme ürettiğinin sentetik testi."""
import pandas as pd
import pytest

from src.preprocessing.build_index import build_image_inventory
from src.preprocessing.make_splits import (BUILDING_KEY, IMAGE_KEY, KNOWN_SPLITS, check_no_leakage,
                                           filter_trainable, make_standard, make_unseen_disaster, n_val_for,
                                           split_buildings, strata_table, write_split_files)

DAMAGES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def fake_dataset(source_split, disasters, n_images=40, n_buildings=5, n_empty=2):
    """Her afet için n_images binalı + n_empty binasız görüntü. (görüntü listesi, bina tablosu) döndürür."""
    images, rows = [], []
    for disaster in disasters:
        for i in range(n_images + n_empty):
            image_id = f"{disaster}_{source_split}_{i:08d}"
            images.append({"source_split": source_split, "image_id": image_id,
                           "disaster": disaster, "disaster_type": "wind"})
            if i >= n_images:
                continue  # binasız görüntü
            for j in range(n_buildings):
                rows.append({"source_split": source_split, "image_id": image_id, "disaster": disaster,
                             "damage": DAMAGES[(i + j) % 4], "uid": f"{image_id}-{j}",
                             "is_bad_polygon": False, "out_of_bounds": j == 0})
    return pd.DataFrame(images), pd.DataFrame(rows)


ALL_UNCLASSIFIED = "event-a_train_00000003"   # bütün binaları un-classified
EMPTY_TRAIN = "event-a_train_00000040"        # binasız
ALL_BAD_TEST = "event-b_test_00000001"        # test setinde bütün binaları bozuk
EMPTY_TEST = "event-b_test_00000010"          # test setinde binasız


@pytest.fixture
def data():
    train_images, train_b = fake_dataset("train", ["event-a", "event-b", "event-c"])
    test_images, test_b = fake_dataset("test", ["event-a", "event-b"], n_images=10)
    train_b.loc[train_b["image_id"] == ALL_UNCLASSIFIED, "damage"] = "un-classified"
    train_b.loc[train_b.index[1], "is_bad_polygon"] = True
    test_b.loc[test_b["image_id"] == ALL_BAD_TEST, "is_bad_polygon"] = True
    return {
        "train": build_image_inventory(train_images, train_b),
        "test": build_image_inventory(test_images, test_b),
        "buildings": pd.concat([train_b, test_b], ignore_index=True),
        "train_images": train_images, "train_b": train_b,
    }


def test_filter_trainable_drops_unclassified_and_bad_keeps_out_of_bounds(data):
    train_b = data["train_b"]
    kept = filter_trainable(train_b)
    assert len(kept) == len(train_b) - 5 - 1  # 5 un-classified bina + 1 bozuk poligon
    assert "un-classified" not in set(kept["damage"]) and not kept["is_bad_polygon"].any()
    assert kept["out_of_bounds"].sum() == train_b["out_of_bounds"].sum() - 1


def test_standard_splits_are_disjoint(data):
    splits = make_standard(data["train"], data["test"])
    check_no_leakage(splits, IMAGE_KEY)
    check_no_leakage(split_buildings(splits, data["buildings"]), BUILDING_KEY)
    assert len(splits["train"]) + len(splits["val"]) == len(data["train"])


def test_buildings_of_an_image_stay_together(data):
    building_splits = split_buildings(make_standard(data["train"], data["test"]), data["buildings"])
    owner = pd.concat([df.assign(split=name) for name, df in building_splits.items()])
    assert (owner.groupby("image_id")["split"].nunique() == 1).all()


def test_val_fraction_is_per_disaster(data):
    splits = make_standard(data["train"], data["test"])
    val_per_disaster = splits["val"].groupby("disaster")["image_id"].nunique()
    # Binalı strata: floor(0.15 * 40 + 0.5) = 6, binasız strata: floor(0.15 * 2 + 0.5) = 0
    assert val_per_disaster.to_dict() == {"event-a": 6, "event-b": 6, "event-c": 6}


def test_same_seed_same_result(data):
    first, second = make_standard(data["train"], data["test"]), make_standard(data["train"], data["test"])
    for name in first:
        assert list(first[name]["image_id"]) == list(second[name]["image_id"])
    other = make_standard(data["train"], data["test"], seed=7)
    assert set(other["val"]["image_id"]) != set(first["val"]["image_id"])


@pytest.mark.parametrize("n, expected", [(0, 0), (1, 0), (3, 0), (4, 1), (10, 2), (30, 5), (40, 6), (100, 15)])
def test_n_val_formula_rounds_halves_up(n, expected):
    # 10 → 1.5 → 2 ve 30 → 4.5 → 5: Python round() burada 2 ve 4 verirdi
    assert n_val_for(n) == expected


@pytest.fixture
def strata_inventory():
    """Afet başına 40 binalı + 20 binasız görüntü; küçük afette 6 binalı + 3 binasız."""
    images, buildings = fake_dataset("train", ["event-a", "event-b"], n_images=40, n_empty=20)
    small_images, small_buildings = fake_dataset("train", ["event-s"], n_images=6, n_empty=3)
    return build_image_inventory(pd.concat([images, small_images], ignore_index=True),
                                 pd.concat([buildings, small_buildings], ignore_index=True))


def test_strata_split_about_85_15_within_each_group(strata_inventory):
    splits = make_standard(strata_inventory, strata_inventory.iloc[:0])
    table = strata_table(splits["train"], splits["val"])
    for disaster in ["event-a", "event-b"]:
        row = table.loc[disaster]
        assert (row.val_binali, row.train_binali) == (n_val_for(40), 40 - n_val_for(40)) == (6, 34)
        assert (row.val_binasiz, row.train_binasiz) == (n_val_for(20), 20 - n_val_for(20)) == (3, 17)
        assert abs(row.val_binali / row.binali - 0.15) < 0.02 and abs(row.val_binasiz / row.binasiz - 0.15) < 0.02
    # Küçük afette binasız strata (n=3) validation'a 0 görüntü verir ve işaretlenir
    small = table.loc["event-s"]
    assert (small.val_binali, small.val_binasiz) == (1, 0) and small.val_0_strata == "binasiz"
    assert table.loc["TOPLAM", "toplam"] == len(strata_inventory)


def test_strata_split_is_deterministic(strata_inventory):
    first = make_standard(strata_inventory, strata_inventory.iloc[:0])
    second = make_standard(strata_inventory.sample(frac=1, random_state=0), strata_inventory.iloc[:0])
    for name in ["train", "val"]:
        assert sorted(first[name]["image_id"]) == sorted(second[name]["image_id"])


def test_no_image_is_lost_or_duplicated(strata_inventory):
    splits = make_standard(strata_inventory, strata_inventory.iloc[:0])
    train_ids, val_ids = list(splits["train"]["image_id"]), list(splits["val"]["image_id"])
    assert sorted(train_ids + val_ids) == sorted(strata_inventory["image_id"])
    assert not set(train_ids) & set(val_ids)
    unseen = make_unseen_disaster(strata_inventory, "event-s")
    all_ids = [i for df in unseen.values() for i in df["image_id"]]
    assert sorted(all_ids) == sorted(strata_inventory["image_id"])


def test_stratum_uses_raw_building_count_not_eligibility(data):
    # Bütün binaları un-classified olan görüntü binalı strata'da kalır
    splits = make_standard(data["train"], data["test"])
    row = pd.concat([splits["train"], splits["val"]]).set_index("image_id").loc[ALL_UNCLASSIFIED]
    assert row.n_buildings > 0 and row.n_eligible_buildings == 0
    table = strata_table(splits["train"], splits["val"])
    assert table.loc["event-a", "binali"] == 40 and table.loc["event-a", "binasiz"] == 2


def test_images_without_eligible_buildings_stay_in_train_val(data):
    splits = make_standard(data["train"], data["test"])
    train_val = set(splits["train"]["image_id"]) | set(splits["val"]["image_id"])
    assert {EMPTY_TRAIN, ALL_UNCLASSIFIED} <= train_val
    inventory = data["train"].set_index("image_id")
    assert inventory.loc[EMPTY_TRAIN, "n_buildings"] == 0
    assert inventory.loc[ALL_UNCLASSIFIED, ["n_buildings", "n_eligible_buildings"]].tolist() == [5, 0]


def test_full_challenge_test_lists_are_kept(data):
    standard = make_standard(data["train"], data["test"])
    assert set(standard["test"]["image_id"]) == set(data["test"]["image_id"])
    assert {ALL_BAD_TEST, EMPTY_TEST} <= set(standard["test"]["image_id"])
    unseen = make_unseen_disaster(data["train"], "event-b", data["test"])
    assert {ALL_BAD_TEST, EMPTY_TEST} <= set(unseen["holdout_challenge_test"]["image_id"])


def test_split_assignment_ignores_building_eligibility(data):
    before = make_standard(data["train"], data["test"])
    changed_b = data["train_b"].copy()
    changed_b.loc[changed_b.index[::3], "damage"] = "un-classified"
    changed_b.loc[changed_b.index[1::4], "is_bad_polygon"] = True
    changed = build_image_inventory(data["train_images"], changed_b)
    assert not changed["n_eligible_buildings"].equals(data["train"]["n_eligible_buildings"])
    after = make_standard(changed, data["test"])
    for name in before:
        assert list(before[name]["image_id"]) == list(after[name]["image_id"])


def test_unseen_disaster_without_test_set(data):
    splits = make_unseen_disaster(data["train"], "event-b")
    check_no_leakage(splits)
    assert set(splits) == {"train", "val", "holdout_train_source"}
    assert len(splits["holdout_train_source"]) == 42  # binasız görüntüler dahil
    assert "event-b" not in set(splits["train"]["disaster"]) | set(splits["val"]["disaster"])


def test_unseen_disaster_with_test_set(data):
    splits = make_unseen_disaster(data["train"], "event-b", data["test"])
    check_no_leakage(splits)
    assert set(splits) == {"train", "val", "holdout_train_source", "holdout_challenge_test"}
    assert set(splits["holdout_challenge_test"]["source_split"]) == {"test"}
    assert set(splits["holdout_challenge_test"]["disaster"]) == {"event-b"}
    # Train/val, test seti verilse de verilmese de aynı kalır
    without_test = make_unseen_disaster(data["train"], "event-b")
    for name in ["train", "val", "holdout_train_source"]:
        assert splits[name].equals(without_test[name])


def test_unseen_disaster_missing_from_test_set_has_no_holdout_challenge_test(data):
    splits = make_unseen_disaster(data["train"], "event-c", data["test"])  # event-c test setinde yok
    assert "holdout_challenge_test" not in splits


def test_stale_split_files_are_removed(data, tmp_path):
    readme = tmp_path / "README.md"
    other = tmp_path / "notes.txt"
    readme.write_text("# readme\n", encoding="utf-8")
    other.write_text("başka dosya\n", encoding="utf-8")
    known = KNOWN_SPLITS["unseen-disaster"]

    with_test = make_unseen_disaster(data["train"], "event-b", data["test"])
    write_split_files(with_test, tmp_path, known)
    assert (tmp_path / "holdout_challenge_test_images.txt").exists()

    without_test = make_unseen_disaster(data["train"], "event-b")
    hashes, removed = write_split_files(without_test, tmp_path, known)
    assert removed == ["holdout_challenge_test_images.txt"]
    assert not (tmp_path / "holdout_challenge_test_images.txt").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "README.md", "holdout_train_source_images.txt", "notes.txt", "train_images.txt", "val_images.txt"]
    assert readme.read_text(encoding="utf-8") == "# readme\n"
    assert other.read_text(encoding="utf-8") == "başka dosya\n"


def test_leakage_check_catches_shared_image(data):
    splits = make_standard(data["train"], data["test"])
    splits["val"] = pd.concat([splits["val"], splits["train"].head(1)])
    with pytest.raises(AssertionError):
        check_no_leakage(splits)


def test_leakage_check_catches_train_image_in_challenge_test(data):
    splits = make_standard(data["train"], data["test"])
    # Aynı görüntünün satırı (source_split dahil) aynen test listesine kopyalanır; böylece test
    # IMAGE_KEY ["image_id"] veya ["source_split", "image_id"] olsa da sızıntıyı yakalamalı.
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)])
    with pytest.raises(AssertionError, match="train ve test"):
        check_no_leakage(splits, IMAGE_KEY)
