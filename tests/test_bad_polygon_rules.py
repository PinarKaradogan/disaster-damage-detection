"""build_index'teki bozuk poligon ve sınır dışı kurallarının sentetik örneklerle testi."""
import math

from src.preprocessing.build_index import build_index, summarize


def make_label(polygons, disaster="test-event"):
    """polygons: [(uid, wkt, subtype veya None), ...] → xBD benzeri label dict."""
    features = []
    for uid, wkt, subtype in polygons:
        props = {"feature_type": "building", "uid": uid}
        if subtype is not None:
            props["subtype"] = subtype
        features.append({"properties": props, "wkt": wkt})
    return {"metadata": {"disaster": disaster, "disaster_type": "wind"}, "features": {"xy": features}}


NORMAL = "POLYGON ((100 100, 140 100, 140 130, 100 130, 100 100))"     # 40x30
THIN = "POLYGON ((10 10, 60 10, 60 12, 10 12, 10 10))"                # 50x2: kısa kenar < 3
LONG = "POLYGON ((200 200, 300 200, 300 208, 200 208, 200 200))"      # 100x8: oran 12.5
BOWTIE = "POLYGON ((0 0, 20 20, 20 0, 0 20, 0 0))"                    # kendini kesen: geçersiz
OUT = "POLYGON ((1000 500, 1040 500, 1040 540, 1000 540, 1000 500))"  # x_max > 1024
SCI = "POLYGON ((-8.3e-06 697.2, 9.1 675.9, 20.2 680.7, 4.1 718.7, -8.3e-06 697.2))"


def index_of(polygons):
    df = build_index([("img_0", make_label(polygons))], "train")
    return df.set_index("uid")


def test_normal_polygon_is_clean():
    row = index_of([("a", NORMAL, "no-damage")]).loc["a"]
    assert (row.width, row.height, row["size"]) == (40, 30, 40)
    assert math.isclose(row.aspect_ratio, 40 / 30)
    assert row.is_valid and not row.is_bad_polygon and not row.out_of_bounds


def test_thin_polygon_is_bad():
    row = index_of([("a", THIN, "destroyed")]).loc["a"]
    assert row.is_valid and row.is_bad_polygon


def test_long_polygon_is_bad():
    row = index_of([("a", LONG, "no-damage")]).loc["a"]
    assert row.aspect_ratio == 12.5 and row.is_bad_polygon


def test_invalid_polygon_is_bad():
    row = index_of([("a", BOWTIE, "no-damage")]).loc["a"]
    assert not row.is_valid and row.is_bad_polygon


def test_out_of_bounds_is_not_bad():
    row = index_of([("a", OUT, "no-damage")]).loc["a"]
    assert row.out_of_bounds and not row.is_bad_polygon


def test_zero_short_side_gives_infinite_aspect():
    line_like = "POLYGON ((0 5, 50 5, 30 5, 0 5))"  # yüksekliği 0
    row = index_of([("a", line_like, "no-damage")]).loc["a"]
    assert row.height == 0 and math.isinf(row.aspect_ratio) and row.is_bad_polygon


def test_scientific_notation_bounds():
    row = index_of([("a", SCI, "no-damage")]).loc["a"]
    assert math.isclose(row.x_min, -8.3e-06) and row.y_min == 675.9 and row.y_max == 718.7


def test_missing_subtype_becomes_unknown():
    assert index_of([("a", NORMAL, None)]).loc["a", "damage"] == "unknown"


def test_summary_counts_overlapping_rules_once_in_union():
    thin_and_long = "POLYGON ((0 0, 50 0, 50 2, 0 2, 0 0))"  # hem ince hem uzun
    df = build_index([("img_0", make_label([
        ("a", NORMAL, "no-damage"), ("b", THIN, "destroyed"), ("c", LONG, "no-damage"),
        ("d", BOWTIE, "un-classified"), ("e", OUT, "minor-damage"), ("f", thin_and_long, "no-damage"),
    ]))], "train")
    s = summarize(df)
    assert (s["toplam"], s["un-classified hariç"]) == (6, 5)
    assert (s["invalid"], s["short<3"], s["aspect>10"]) == (1, 2, 3)
    assert s["bad union"] == 4 and s["out_of_bounds"] == 1


def test_output_is_sorted_and_deterministic():
    labels = [("img_1", make_label([("z", NORMAL, "no-damage"), ("b", THIN, "no-damage")])),
              ("img_0", make_label([("c", LONG, "no-damage")]))]
    df = build_index(labels, "train")
    assert list(zip(df.image_id, df.uid)) == [("img_0", "c"), ("img_1", "b"), ("img_1", "z")]
    assert df.equals(build_index(reversed(labels), "train"))
