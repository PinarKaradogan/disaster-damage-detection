"""xBD post-disaster etiketlerinden bina indeksi (buildings_<split>.csv) üretir.

Kullanım (repo kökünden):
    python -m src.preprocessing.build_index

Her bina için WKT geometrisi features -> xy -> wkt alanından shapely ile okunur,
sınır kutusu geom.bounds ile hesaplanır. data/<split>/labels yoksa o bölme atlanır.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
SPLITS = ["train", "test"]

IMG_SIZE = 1024
THIN_PX = 3
MAX_ASPECT = 10
SIZE_THRESHOLDS = [150, 200, 256, 300, 400, 500, 600, 700, 800, 900, 1000]

COLUMNS = ["source_split", "image_id", "disaster", "disaster_type", "damage", "uid",
           "x_min", "y_min", "x_max", "y_max", "width", "height", "size",
           "is_valid", "aspect_ratio", "is_bad_polygon", "out_of_bounds"]

# Faz 2'de Pınar ve Bahar'ın bağımsız olarak bulduğu train değerleri
TRAIN_EXPECTED = {
    "toplam": 162787, "un-classified hariç": 159794, "invalid": 1, "short<3": 91,
    "aspect>10": 118, "bad union": 177, "out_of_bounds": 2476,
    **{f">={t}": n for t, n in zip(SIZE_THRESHOLDS, [1189, 492, 245, 148, 41, 14, 7, 3, 1, 1, 1])},
}


def label_to_rows(label, image_id, source_split):
    """Bir post-disaster JSON içeriğini (dict) bina satırlarına çevirir."""
    meta = label["metadata"]
    rows = []
    for b in label["features"]["xy"]:
        geom = shapely_wkt.loads(b["wkt"])
        x_min, y_min, x_max, y_max = geom.bounds
        rows.append({
            "source_split": source_split,
            "image_id": image_id,
            "disaster": meta["disaster"],
            "disaster_type": meta["disaster_type"],
            "damage": b["properties"].get("subtype", "unknown"),
            "uid": b["properties"].get("uid"),
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
            "is_valid": geom.is_valid,
        })
    return rows


def add_geometry_flags(df):
    """Sınır kutusundan boyut, aspect ratio, bozuk poligon ve sınır dışı sütunlarını ekler."""
    df = df.copy()
    df["width"] = df["x_max"] - df["x_min"]
    df["height"] = df["y_max"] - df["y_min"]
    short_side = df[["width", "height"]].min(axis=1)
    long_side = df[["width", "height"]].max(axis=1)
    df["size"] = long_side
    # Kısa kenar 0 ise oran sonsuz kabul edilir
    df["aspect_ratio"] = np.where(short_side > 0, long_side / short_side.where(short_side > 0, 1), np.inf)
    df["is_bad_polygon"] = ~df["is_valid"].astype(bool) | (short_side < THIN_PX) | (df["aspect_ratio"] > MAX_ASPECT)
    # Bilgi amaçlı: bu binalar atılmaz, Faz 3 patch çıkarımında crop 0..1024'e clamp edilir
    df["out_of_bounds"] = ((df["x_min"] < 0) | (df["y_min"] < 0)
                           | (df["x_max"] > IMG_SIZE) | (df["y_max"] > IMG_SIZE))
    return df


def build_index(labels, source_split):
    """labels: [(image_id, label_dict), ...] → sıralı bina tablosu."""
    rows = []
    for image_id, label in labels:
        rows += label_to_rows(label, image_id, source_split)
    df = add_geometry_flags(pd.DataFrame(rows, columns=COLUMNS[:10] + ["is_valid"]))
    return df[COLUMNS].sort_values(["image_id", "uid"]).reset_index(drop=True)


def summarize(df):
    """Konsol raporu ve kabul kontrolü için özet sayılar."""
    short_side = df[["width", "height"]].min(axis=1)
    summary = {
        "toplam": len(df),
        "un-classified hariç": int((df["damage"] != "un-classified").sum()),
        "invalid": int((~df["is_valid"].astype(bool)).sum()),
        "short<3": int((short_side < THIN_PX).sum()),
        "aspect>10": int((df["aspect_ratio"] > MAX_ASPECT).sum()),
        "bad union": int(df["is_bad_polygon"].sum()),
        "out_of_bounds": int(df["out_of_bounds"].sum()),
    }
    for t in SIZE_THRESHOLDS:
        summary[f">={t}"] = int((df["size"] >= t).sum())
    return summary


def read_labels(label_dir):
    for path in sorted(label_dir.glob("*_post_disaster.json")):
        yield path.stem.replace("_post_disaster", ""), json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mismatches = []

    for split in SPLITS:
        label_dir = DATA_DIR / split / "labels"
        print(f"\n=== {split} ===")
        if not label_dir.exists():
            print(f"ATLANDI: data/{split}/labels bulunamadı, buildings_{split}.csv üretilmedi.")
            continue

        df = build_index(read_labels(label_dir), split)
        n_images = len(list(label_dir.glob("*_post_disaster.json")))
        n_unknown = int((df["damage"] == "unknown").sum())
        n_dup = int(df.duplicated(["image_id", "uid"]).sum())
        print(f"Post JSON: {n_images}  |  subtype eksik (damage=unknown): {n_unknown}  |  "
              f"tekrarlanan (image_id, uid): {n_dup}")

        summary = summarize(df)
        expected = TRAIN_EXPECTED if split == "train" else {}
        print(f"{'metrik':22s}{'bulunan':>10s}{'beklenen':>10s}")
        for key, value in summary.items():
            exp = expected.get(key)
            mark = "" if exp is None else ("  OK" if exp == value else "  FARK")
            print(f"{key:22s}{value:>10d}{'' if exp is None else exp:>10}{mark}")
            if exp is not None and exp != value:
                mismatches.append((split, key, exp, value))

        out_path = OUT_DIR / f"buildings_{split}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")
        print(f"\nYazıldı: {out_path.relative_to(ROOT).as_posix()}  ({len(df)} satır)")
        print(f"SHA256 : {sha256(out_path)}")

    if mismatches:
        print("\nKABUL KRİTERİ SAĞLANMADI:")
        for split, key, exp, value in mismatches:
            print(f"  {split} {key}: beklenen {exp}, bulunan {value}")
        sys.exit(1)


if __name__ == "__main__":
    main()
