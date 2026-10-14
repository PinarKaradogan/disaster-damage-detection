"""build_index ve make_splits main() akışının eksik/boş Challenge test girdileriyle davranışı (CLI düzeyi)."""
import json
import sys

import pandas as pd
import pytest

from src.preprocessing import build_index as bi
from src.preprocessing import make_splits as ms


def fake_tables(source_split, disasters, n_images=8, n_empty=2):
    """Afet başına n_images binalı (2'şer bina) + n_empty binasız görüntü: (envanter, bina tablosu)."""
    images, rows = [], []
    for disaster in disasters:
        for i in range(n_images + n_empty):
            image_id = f"{disaster}_{source_split}_{i:08d}"
            images.append({"source_split": source_split, "image_id": image_id,
                           "disaster": disaster, "disaster_type": "wind"})
            for j in range(2 if i < n_images else 0):
                rows.append({"source_split": source_split, "image_id": image_id, "disaster": disaster,
                             "disaster_type": "wind", "damage": "no-damage", "uid": f"{image_id}-{j}",
                             "is_bad_polygon": False, "out_of_bounds": False})
    buildings = pd.DataFrame(rows)
    return bi.build_image_inventory(pd.DataFrame(images), buildings), buildings


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Geçici bir repo kökü: data/processed içinde train CSV'leri, boş splits/ klasörü."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    images, buildings = fake_tables("train", ["event-a", "event-b"])
    images.to_csv(processed / "images_train.csv", index=False)
    buildings.to_csv(processed / "buildings_train.csv", index=False)
    for module in (bi, ms):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(ms, "PROCESSED_DIR", processed)
    monkeypatch.setattr(ms, "SPLITS_DIR", tmp_path / "splits")
    monkeypatch.setattr(bi, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bi, "OUT_DIR", processed)
    return tmp_path


def run_make_splits(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["make_splits", *args])
    ms.main()


def test_unseen_disaster_runs_with_images_test_but_no_buildings_test(project, monkeypatch, capsys):
    test_images, _ = fake_tables("test", ["event-b"])
    test_images.to_csv(project / "data" / "processed" / "images_test.csv", index=False)  # buildings_test.csv yok

    run_make_splits(monkeypatch, "--mode", "unseen-disaster", "--holdout-disaster", "event-b")

    out_dir = project / "splits" / "unseen_disaster" / "event-b"
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "holdout_train_source_images.txt", "train_images.txt", "val_images.txt"]
    assert "holdout_challenge_test_images.txt yazılmadı" in capsys.readouterr().out


def test_unseen_disaster_writes_holdout_challenge_test_when_test_inputs_exist(project, monkeypatch):
    test_images, test_buildings = fake_tables("test", ["event-b"])
    test_images.to_csv(project / "data" / "processed" / "images_test.csv", index=False)
    test_buildings.to_csv(project / "data" / "processed" / "buildings_test.csv", index=False)

    run_make_splits(monkeypatch, "--mode", "unseen-disaster", "--holdout-disaster", "event-b")

    out = project / "splits" / "unseen_disaster" / "event-b" / "holdout_challenge_test_images.txt"
    assert out.read_text(encoding="utf-8").split() == sorted(test_images["image_id"])


def test_standard_rejects_empty_test_inventory(project, monkeypatch):
    processed = project / "data" / "processed"
    test_images, test_buildings = fake_tables("test", ["event-a"])
    test_images.iloc[:0].to_csv(processed / "images_test.csv", index=False)  # yalnızca başlık satırı
    test_buildings.iloc[:0].to_csv(processed / "buildings_test.csv", index=False)

    with pytest.raises(SystemExit, match="images_test.csv boş"):
        run_make_splits(monkeypatch)
    assert not (project / "splits").exists()


def test_build_index_treats_empty_test_labels_as_missing(project, capsys):
    processed = project / "data" / "processed"
    (project / "data" / "test" / "labels").mkdir(parents=True)  # klasör var, 0 post JSON
    for name in ["buildings_test.csv", "images_test.csv"]:
        (processed / name).write_text("eski boş çıktı\n", encoding="utf-8")

    bi.main()  # data/train/labels da yok: train atlanır, test eksik girdi olarak raporlanır

    out = capsys.readouterr().out
    assert "data/test/labels içinde *_post_disaster.json yok (eksik girdi)" in out
    assert not (processed / "buildings_test.csv").exists() and not (processed / "images_test.csv").exists()


def test_build_index_writes_test_csvs_when_post_json_exists(project):
    labels = project / "data" / "test" / "labels"
    labels.mkdir(parents=True)
    label = {"metadata": {"disaster": "event-a", "disaster_type": "wind"}, "features": {"xy": [
        {"properties": {"uid": "a", "subtype": "no-damage"},
         "wkt": "POLYGON ((100 100, 140 100, 140 130, 100 130, 100 100))"}]}}
    (labels / "event-a_00000000_post_disaster.json").write_text(json.dumps(label), encoding="utf-8")

    bi.main()

    images = pd.read_csv(project / "data" / "processed" / "images_test.csv")
    assert images[["image_id", "n_buildings"]].values.tolist() == [["event-a_00000000", 1]]
