"""build_index'in kabul kriterleri tutmadığında CSV yazmadığının ve atomik yazımının testi."""
import pandas as pd
import pytest

from src.preprocessing.build_index import (build_tables, check_acceptance, save_outputs, summarize,
                                           summarize_images)


@pytest.fixture
def tables():
    label = {"metadata": {"disaster": "test-event", "disaster_type": "wind"}, "features": {"xy": [
        {"properties": {"uid": "a", "subtype": "no-damage"},
         "wkt": "POLYGON ((100 100, 140 100, 140 130, 100 130, 100 100))"},
        {"properties": {"uid": "b", "subtype": "destroyed"},
         "wkt": "POLYGON ((10 10, 60 10, 60 12, 10 12, 10 10))"},
    ]}}
    empty = {"metadata": {"disaster": "test-event", "disaster_type": "wind"}, "features": {"xy": []}}
    return build_tables([("img_0", label), ("img_1", empty)], "train")


def outputs(tables, tmp_path, expected_buildings=None, expected_images=None):
    buildings, images = tables
    return [(buildings, tmp_path / "buildings_train.csv", summarize, expected_buildings),
            (images, tmp_path / "images_train.csv", summarize_images, expected_images)]


def test_image_inventory_keeps_images_without_buildings(tables):
    _, images = tables
    assert images[["image_id", "n_buildings", "n_eligible_buildings"]].values.tolist() == [
        ["img_0", 2, 1], ["img_1", 0, 0]]
    assert summarize_images(images) == {"görüntü": 2, "binalı": 1, "binasız": 1,
                                        "uygun binası kalmayan": 0, "uygun binalı": 1}


def test_check_acceptance_reports_only_mismatches(tables):
    summary = summarize(tables[0])
    assert check_acceptance(summary, summary) == []
    assert check_acceptance(summary, {"toplam": 2, "bad union": 5}) == [("bad union", 5, 1)]


def test_passing_criteria_writes_csv(tables, tmp_path):
    out = outputs(tables, tmp_path, summarize(tables[0]), summarize_images(tables[1]))
    assert save_outputs(out) == []
    assert len(pd.read_csv(tmp_path / "buildings_train.csv")) == 2
    assert len(pd.read_csv(tmp_path / "images_train.csv")) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["buildings_train.csv", "images_train.csv"]  # .tmp yok


def test_failing_criteria_does_not_write_and_removes_old_csv(tables, tmp_path):
    for name in ["buildings_train.csv", "images_train.csv"]:
        (tmp_path / name).write_text("eski çalıştırmadan kalan dosya\n", encoding="utf-8")
    expected = {**summarize(tables[0]), "toplam": 999}
    assert save_outputs(outputs(tables, tmp_path, expected)) == [("toplam", 999, 2)]
    assert list(tmp_path.iterdir()) == []


def test_failing_image_criteria_writes_neither_csv(tables, tmp_path):
    (tmp_path / "buildings_train.csv").write_text("eski\n", encoding="utf-8")
    out = outputs(tables, tmp_path, summarize(tables[0]), {"görüntü": 2799, "binasız": 516})
    assert save_outputs(out) == [("görüntü", 2799, 2), ("binasız", 516, 1)]
    assert list(tmp_path.iterdir()) == []


def test_failing_criteria_without_old_csv(tables, tmp_path):
    assert save_outputs(outputs(tables, tmp_path, {"invalid": 7})) == [("invalid", 7, 0)]
    assert list(tmp_path.iterdir()) == []


def test_no_criteria_writes_csv(tables, tmp_path):
    assert save_outputs(outputs(tables, tmp_path)) == []
    assert (tmp_path / "buildings_train.csv").exists() and (tmp_path / "images_train.csv").exists()


def test_interrupted_write_leaves_no_partial_file(tables, tmp_path, monkeypatch):
    out = tmp_path / "buildings_train.csv"
    out.write_text("önceki geçerli dosya\n", encoding="utf-8")

    def broken_to_csv(self, path, **kwargs):
        with open(path, "w", encoding="utf-8") as f:
            f.write("yarım satır")
        raise OSError("disk dolu")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_to_csv)
    with pytest.raises(OSError):
        save_outputs(outputs(tables, tmp_path))
    assert out.read_text(encoding="utf-8") == "önceki geçerli dosya\n"
    assert list(tmp_path.iterdir()) == [out]
