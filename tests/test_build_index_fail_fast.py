"""build_index'in kabul kriterleri tutmadığında CSV yazmadığının ve atomik yazımının testi."""
import pandas as pd
import pytest

from src.preprocessing.build_index import build_index, check_acceptance, save_index, summarize


@pytest.fixture
def df():
    label = {"metadata": {"disaster": "test-event", "disaster_type": "wind"}, "features": {"xy": [
        {"properties": {"uid": "a", "subtype": "no-damage"},
         "wkt": "POLYGON ((100 100, 140 100, 140 130, 100 130, 100 100))"},
        {"properties": {"uid": "b", "subtype": "destroyed"},
         "wkt": "POLYGON ((10 10, 60 10, 60 12, 10 12, 10 10))"},
    ]}}
    return build_index([("img_0", label)], "train")


def test_check_acceptance_reports_only_mismatches(df):
    summary = summarize(df)
    assert check_acceptance(summary, summary) == []
    assert check_acceptance(summary, {"toplam": 2, "bad union": 5}) == [("bad union", 5, 1)]


def test_passing_criteria_writes_csv(df, tmp_path):
    out = tmp_path / "buildings_train.csv"
    assert save_index(df, out, summarize(df)) == []
    assert len(pd.read_csv(out)) == 2
    assert list(tmp_path.iterdir()) == [out]  # .tmp dosyası kalmadı


def test_failing_criteria_does_not_write_and_removes_old_csv(df, tmp_path):
    out = tmp_path / "buildings_train.csv"
    out.write_text("eski çalıştırmadan kalan dosya\n", encoding="utf-8")
    expected = {**summarize(df), "toplam": 999}
    assert save_index(df, out, expected) == [("toplam", 999, 2)]
    assert not out.exists() and list(tmp_path.iterdir()) == []


def test_failing_criteria_without_old_csv(df, tmp_path):
    out = tmp_path / "buildings_train.csv"
    assert save_index(df, out, {"invalid": 7}) == [("invalid", 7, 0)]
    assert not out.exists()


def test_no_criteria_writes_csv(df, tmp_path):
    out = tmp_path / "buildings_test.csv"
    assert save_index(df, out) == [] and out.exists()


def test_interrupted_write_leaves_no_partial_file(df, tmp_path, monkeypatch):
    out = tmp_path / "buildings_train.csv"
    out.write_text("önceki geçerli dosya\n", encoding="utf-8")

    def broken_to_csv(self, path, **kwargs):
        with open(path, "w", encoding="utf-8") as f:
            f.write("yarım satır")
        raise OSError("disk dolu")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_to_csv)
    with pytest.raises(OSError):
        save_index(df, out)
    assert out.read_text(encoding="utf-8") == "önceki geçerli dosya\n"
    assert list(tmp_path.iterdir()) == [out]
