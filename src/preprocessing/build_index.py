"""xBD post-disaster etiketlerinden bina indeksi ve görüntü envanteri üretir.

Kullanım (repo kökünden):
    python -m src.preprocessing.build_index

Çıktılar (data/<split>/labels yoksa o bölme atlanır):
    data/processed/buildings_<split>.csv  her bina için bir satır
    data/processed/images_<split>.csv     her post JSON için bir satır (binasız görüntüler dahil)

Her bina için WKT geometrisi features -> xy -> wkt alanından shapely ile okunur,
sınır kutusu geom.bounds ile hesaplanır.
Train kabul kriterleri CSV'ler yazılmadan önce kontrol edilir; tutmazsa hiçbir CSV yazılmaz,
eski CSV'ler silinir ve script 1 çıkış koduyla durur. CSV'ler atomik olarak (önce .tmp) yazılır.
"""
import hashlib
import json
import os
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
IMAGE_COLUMNS = ["source_split", "image_id", "disaster", "disaster_type", "n_buildings", "n_eligible_buildings"]

# Faz 2'de Pınar ve Bahar'ın bağımsız olarak bulduğu train değerleri
TRAIN_EXPECTED = {
    "toplam": 162787, "un-classified hariç": 159794, "invalid": 1, "short<3": 91,
    "aspect>10": 118, "bad union": 177, "out_of_bounds": 2476,
    **{f">={t}": n for t, n in zip(SIZE_THRESHOLDS, [1189, 492, 245, 148, 41, 14, 7, 3, 1, 1, 1])},
}
# Faz 3 görüntü evreni raporu (post JSON'lardan doğrudan sayım)
TRAIN_IMAGES_EXPECTED = {
    "görüntü": 2799, "binalı": 2283, "binasız": 516,
    "uygun binası kalmayan": 44, "uygun binalı": 2239,
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
    # Bilgi amaçlı, eleme kuralı değil: bu binalar atılmaz. Patch çıkarımında kesit boyutu korunur,
    # görüntü dışında kalan alan siyah (0) padding ile doldurulur; kesit küçültülmez, en-boy oranı bozulmaz.
    df["out_of_bounds"] = ((df["x_min"] < 0) | (df["y_min"] < 0)
                           | (df["x_max"] > IMG_SIZE) | (df["y_max"] > IMG_SIZE))
    return df


def eligible_mask(buildings):
    """Patch çıkarımına uygun binalar: un-classified değil ve bozuk poligon değil."""
    return (buildings["damage"] != "un-classified") & ~buildings["is_bad_polygon"].astype(bool)


def build_tables(labels, source_split):
    """labels: [(image_id, label_dict), ...] → (sıralı bina tablosu, sıralı görüntü envanteri)."""
    rows, images = [], []
    for image_id, label in labels:
        meta = label["metadata"]
        images.append({"source_split": source_split, "image_id": image_id,
                       "disaster": meta["disaster"], "disaster_type": meta["disaster_type"]})
        rows += label_to_rows(label, image_id, source_split)
    buildings = add_geometry_flags(pd.DataFrame(rows, columns=COLUMNS[:10] + ["is_valid"]))
    buildings = buildings[COLUMNS].sort_values(["image_id", "uid"]).reset_index(drop=True)
    return buildings, build_image_inventory(pd.DataFrame(images, columns=IMAGE_COLUMNS[:4]), buildings)


def build_index(labels, source_split):
    """labels: [(image_id, label_dict), ...] → sıralı bina tablosu."""
    return build_tables(labels, source_split)[0]


def build_image_inventory(images, buildings):
    """Her görüntü için bina ve uygun bina sayısını ekler; binası olmayan görüntüler 0 ile kalır."""
    counts = pd.DataFrame({
        "n_buildings": buildings.groupby("image_id").size(),
        "n_eligible_buildings": eligible_mask(buildings).groupby(buildings["image_id"]).sum(),
    })
    inventory = images.merge(counts, left_on="image_id", right_index=True, how="left")
    inventory[["n_buildings", "n_eligible_buildings"]] = (
        inventory[["n_buildings", "n_eligible_buildings"]].fillna(0).astype(int))
    return inventory[IMAGE_COLUMNS].sort_values("image_id").reset_index(drop=True)


def summarize(df):
    """Bina tablosu için konsol raporu ve kabul kontrolü özet sayıları."""
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


def summarize_images(inventory):
    """Görüntü envanteri için özet sayılar."""
    has_buildings = inventory["n_buildings"] > 0
    has_eligible = inventory["n_eligible_buildings"] > 0
    return {
        "görüntü": len(inventory),
        "binalı": int(has_buildings.sum()),
        "binasız": int((~has_buildings).sum()),
        "uygun binası kalmayan": int((has_buildings & ~has_eligible).sum()),
        "uygun binalı": int(has_eligible.sum()),
    }


def read_labels(label_dir):
    for path in sorted(label_dir.glob("*_post_disaster.json")):
        yield path.stem.replace("_post_disaster", ""), json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_acceptance(summary, expected):
    """Beklenen değerlerle eşleşmeyen metrikleri [(metrik, beklenen, bulunan), ...] olarak döndürür."""
    return [(key, exp, summary.get(key)) for key, exp in expected.items() if summary.get(key) != exp]


def write_csv_atomic(df, path):
    """Önce <ad>.tmp dosyasına yazar, başarılıysa asıl ada taşır; yarım CSV kalmaz."""
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_outputs(outputs):
    """outputs: [(tablo, yol, özet_fonksiyonu, beklenen veya None), ...]

    Önce tüm kabul kriterleri kontrol edilir. Hepsi tutarsa tüm CSV'ler atomik yazılır ve [] döner.
    Biri bile tutmazsa hiçbir CSV yazılmaz, aynı yollardaki eski CSV'ler silinir ve
    eşleşmeyen metrikler döndürülür.
    """
    mismatches = []
    for table, _, summary_fn, expected in outputs:
        mismatches += check_acceptance(summary_fn(table), expected or {})
    if mismatches:
        for _, path, _, _ in outputs:
            if Path(path).exists():
                Path(path).unlink()
                print(f"Eski dosya silindi: {Path(path).name}")
        return mismatches
    for table, path, _, _ in outputs:
        write_csv_atomic(table, path)
    return []


def print_summary(summary, expected):
    print(f"{'metrik':24s}{'bulunan':>10s}{'beklenen':>10s}")
    for key, value in summary.items():
        exp = expected.get(key)
        mark = "" if exp is None else ("  OK" if exp == value else "  FARK")
        print(f"{key:24s}{value:>10d}{'' if exp is None else exp:>10}{mark}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        label_dir = DATA_DIR / split / "labels"
        print(f"\n=== {split} ===")
        if not label_dir.exists() or not any(label_dir.glob("*_post_disaster.json")):
            # Klasörün varlığı yetmez: 0 post JSON eksik girdi sayılır, boş CSV geçerli çıktı gibi bırakılmaz
            reason = "bulunamadı" if not label_dir.exists() else "içinde *_post_disaster.json yok (eksik girdi)"
            print(f"ATLANDI: data/{split}/labels {reason}; buildings_{split}.csv ve images_{split}.csv üretilmedi.")
            for name in [f"buildings_{split}.csv", f"images_{split}.csv"]:
                if (OUT_DIR / name).exists():
                    (OUT_DIR / name).unlink()
                    print(f"Eski dosya silindi: {name}")
            continue

        buildings, images = build_tables(read_labels(label_dir), split)
        n_unknown = int((buildings["damage"] == "unknown").sum())
        n_dup = int(buildings.duplicated(["image_id", "uid"]).sum())
        print(f"subtype eksik (damage=unknown): {n_unknown}  |  tekrarlanan (image_id, uid): {n_dup}")

        expected_buildings = TRAIN_EXPECTED if split == "train" else {}
        expected_images = TRAIN_IMAGES_EXPECTED if split == "train" else {}
        print("\nBinalar")
        print_summary(summarize(buildings), expected_buildings)
        print("\nGörüntüler")
        print_summary(summarize_images(images), expected_images)

        buildings_path = OUT_DIR / f"buildings_{split}.csv"
        images_path = OUT_DIR / f"images_{split}.csv"
        mismatches = save_outputs([
            (buildings, buildings_path, summarize, expected_buildings),
            (images, images_path, summarize_images, expected_images),
        ])
        if mismatches:
            print(f"\nKABUL KRİTERİ SAĞLANMADI, {split} CSV'leri yazılmadı:")
            for key, exp, value in mismatches:
                print(f"  {split} {key}: beklenen {exp}, bulunan {value}")
            sys.exit(1)
        for path, table in [(buildings_path, buildings), (images_path, images)]:
            print(f"\nYazıldı: {path.relative_to(ROOT).as_posix()}  ({len(table)} satır)")
            print(f"SHA256 : {sha256(path)}")


if __name__ == "__main__":
    main()
