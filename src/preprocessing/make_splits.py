"""Görüntü envanterinden image-level train/val/test bölmeleri üretir.

Kullanım (repo kökünden):
    python -m src.preprocessing.make_splits
    python -m src.preprocessing.make_splits --mode unseen-disaster --holdout-disaster hurricane-matthew

Girdiler: data/processed/images_<split>.csv (bölme evreni) ve buildings_<split>.csv (yalnızca rapor için).

Çıktılar:
    splits/standard/{train,val,test}_images.txt
    splits/unseen_disaster/<afet>/{train,val,holdout_train_source,holdout_challenge_test}_images.txt
    (holdout_challenge_test yalnızca images_test.csv varsa ve afet test setinde bulunuyorsa yazılır;
    bu çalıştırmada üretilmeyen eski split dosyaları silinir)

Train/val bölmesi: her afet içinde görüntüler ham bina anotasyonu sayısına göre iki strata'ya ayrılır
(n_buildings > 0 ve n_buildings == 0; damage, un-classified, is_bad_polygon ve patch uygunluğu bu
gruplamayı etkilemez). Her strata için image_id listesi sorted(), seed 42 ile deterministik karıştırılır,
n_val = floor(0.15 * n + 0.5) görüntü validation'a, kalanı train'e gider. Küçük strata'larda n_val 0
olabilir; oran zorla değiştirilmez, bu durum raporda işaretlenir. Unseen-disaster modunda kural holdout
afet hariç görüntülere uygulanır; holdout afet tamamen kendi bölmesine gider.

Split dosyaları image-level'dir ve ilgili Challenge setindeki tüm görüntüleri içerir (binasız görüntüler
dahil). Bina uygunluk filtresi (un-classified, is_bad_polygon) yalnızca patch çıkarma aşamasında uygulanır;
binasız veya uygun binası kalmayan görüntülerden patch çıkmaz. Aynı split sınırları tüm modellerde
(patch sınıflandırıcıları ve U-Net) kullanılır. Challenge test'ten gelen dosyalar yalnızca final
değerlendirmede kullanılır; model ve hiperparametre seçimleri sadece train/val üzerinden yapılır.
out_of_bounds binalar tutulur (patch çıkarımında kesit boyutu korunur, görüntü dışı siyah padding ile doldurulur).
"""
import argparse
import hashlib
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd

from src.preprocessing.build_index import eligible_mask

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"

SEED = 42
VAL_FRACTION = 0.15
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Bölmeler arası sızıntı anahtarları. Train ve Challenge test aynı image_id'yi farklı görüntüler için
# kullanıyorsa IMAGE_KEY = ["source_split", "image_id"] yapılır.
IMAGE_KEY = ["image_id"]
BUILDING_KEY = IMAGE_KEY + ["uid"]

# Script'in her modda üretebileceği bölme adları (eski dosya temizliği yalnızca bunlara dokunur)
KNOWN_SPLITS = {
    "standard": ["train", "val", "test"],
    "unseen-disaster": ["train", "val", "holdout_train_source", "holdout_challenge_test"],
}


def filter_trainable(df):
    """Bina tablosundan un-classified ve bozuk poligonları çıkarır."""
    return df[eligible_mask(df)].reset_index(drop=True)


def n_val_for(n, val_fraction=VAL_FRACTION):
    """Validation görüntü sayısı: n_val = floor(val_fraction * n + 0.5), kesirli sayıyla tam hesaplanır.

    Yarımlar yukarı yuvarlanır (Python round()'un çifte yuvarlamasından farklı). Küçük strata'larda
    0 çıkabilir; oran zorla değiştirilmez.
    """
    return math.floor(Fraction(str(val_fraction)) * n + Fraction(1, 2))


def split_train_val(images, val_fraction=VAL_FRACTION, seed=SEED):
    """Her afet ve her strata (n_buildings > 0 / n_buildings == 0) için ayrı görüntü bazında train/val bölmesi.

    Strata ham bina anotasyonu sayısına göredir; damage ve bina uygunluğu gruplamayı etkilemez.
    (train_ids, val_ids) döndürür.
    """
    train_ids, val_ids = [], []
    has_buildings = images["n_buildings"] > 0
    for disaster in sorted(images["disaster"].unique()):
        for stratum in [True, False]:
            mask = (images["disaster"] == disaster) & (has_buildings == stratum)
            ids = sorted(images.loc[mask, "image_id"].unique())
            # Her (afet, strata) kendi seed'li karıştırıcısını kullanır: biri değişince diğerleri etkilenmez
            random.Random(seed).shuffle(ids)
            n_val = n_val_for(len(ids), val_fraction)
            val_ids += ids[:n_val]
            train_ids += ids[n_val:]
    return sorted(train_ids), sorted(val_ids)


def strata_table(train, val):
    """Afet bazında strata dağılımı; 0 validation örneği çıkan strata'lar 'val_0_strata' sütununda işaretlenir."""
    rows = []
    for disaster in sorted(set(train["disaster"]) | set(val["disaster"])):
        t, v = train[train["disaster"] == disaster], val[val["disaster"] == disaster]
        row = {"disaster": disaster,
               "train_binali": int((t["n_buildings"] > 0).sum()), "train_binasiz": int((t["n_buildings"] == 0).sum()),
               "val_binali": int((v["n_buildings"] > 0).sum()), "val_binasiz": int((v["n_buildings"] == 0).sum())}
        row["binali"] = row["train_binali"] + row["val_binali"]
        row["binasiz"] = row["train_binasiz"] + row["val_binasiz"]
        row["toplam"] = row["binali"] + row["binasiz"]
        row["val_0_strata"] = ", ".join(name for name in ["binali", "binasiz"]
                                        if row[name] > 0 and row[f"val_{name}"] == 0)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("disaster")[
        ["toplam", "binali", "binasiz", "train_binali", "train_binasiz", "val_binali", "val_binasiz", "val_0_strata"]]
    total = table.drop(columns="val_0_strata").sum()
    table.loc["TOPLAM"] = [*total.astype(int), ""]
    return table


def make_standard(train_images, test_images, val_fraction=VAL_FRACTION, seed=SEED):
    """Challenge train'in tüm görüntüleri → train/val, Challenge test'in tüm görüntüleri → test.

    {bölme: görüntü envanteri} döndürür. Bina uygunluğu bölme atamasını etkilemez.
    """
    train_ids, val_ids = split_train_val(train_images, val_fraction, seed)
    return {
        "train": train_images[train_images["image_id"].isin(train_ids)],
        "val": train_images[train_images["image_id"].isin(val_ids)],
        "test": test_images,
    }


def make_unseen_disaster(train_images, holdout_disaster, test_images=None, val_fraction=VAL_FRACTION, seed=SEED):
    """Holdout afetin tüm görüntüleri eğitimden çıkarılır; kalan afetlerin tüm görüntüleri train/val'e bölünür.

    Holdout iki ayrı bölme olarak tutulur (tasarım kararı, review'da değişebilir):
    holdout_train_source = afetin Challenge train görüntüleri, holdout_challenge_test = Challenge test görüntüleri.
    test_images verilmezse ya da afet test setinde yoksa holdout_challenge_test bölmesi üretilmez.
    """
    if holdout_disaster not in set(train_images["disaster"]):
        raise ValueError(f"Bilinmeyen afet: {holdout_disaster}. "
                         f"Seçenekler: {sorted(train_images['disaster'].unique())}")
    is_holdout = train_images["disaster"] == holdout_disaster
    rest = train_images[~is_holdout]
    train_ids, val_ids = split_train_val(rest, val_fraction, seed)
    splits = {
        "train": rest[rest["image_id"].isin(train_ids)],
        "val": rest[rest["image_id"].isin(val_ids)],
        "holdout_train_source": train_images[is_holdout],
    }
    if test_images is not None:
        holdout_challenge_test = test_images[test_images["disaster"] == holdout_disaster]
        if len(holdout_challenge_test):
            splits["holdout_challenge_test"] = holdout_challenge_test
    return splits


def check_no_leakage(splits, key=None):
    """Bölmeler arasında aynı anahtarın (varsayılan IMAGE_KEY) iki bölmede bulunmadığını doğrular."""
    key = key or IMAGE_KEY
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = (set(splits[a][key].itertuples(index=False, name=None))
                      & set(splits[b][key].itertuples(index=False, name=None)))
            assert not shared, f"{a} ve {b} ortak {tuple(key)} içeriyor: {sorted(shared)[:5]}"


def split_buildings(splits, buildings):
    """Her bölmenin uygun binalarını (source_split, image_id) eşleşmesiyle döndürür."""
    eligible = filter_trainable(buildings)
    return {name: eligible.merge(images[["source_split", "image_id"]], on=["source_split", "image_id"])
            for name, images in splits.items()}


def report(splits, building_splits):
    for name, images in splits.items():
        b = building_splits[name]
        has_b = images["n_buildings"] > 0
        print(f"\n=== {name} ===")
        print(f"Görüntü: {len(images)}  |  binalı: {int(has_b.sum())}  |  binasız: {int((~has_b).sum())}  |  "
              f"uygun binalı: {int((images['n_eligible_buildings'] > 0).sum())}")
        print(f"Uygun bina: {len(b)}")
        counts = b["damage"].value_counts().reindex(CLASSES, fill_value=0)
        pct = (counts / max(len(b), 1) * 100).round(1)
        print(pd.DataFrame({"bina": counts, "%": pct}).to_string())
        disasters = images.groupby("disaster").agg(
            goruntu=("image_id", "size"), uygun_binali_goruntu=("n_eligible_buildings", lambda s: int((s > 0).sum())),
            uygun_bina=("n_eligible_buildings", "sum"))
        print(disasters.to_string())


def write_split_files(splits, out_dir, known_names):
    """Her bölme için <bölme>_images.txt yazar (sıralı, satır başına bir image_id).

    known_names içindeki ama bu çalıştırmada üretilmeyen eski split dosyaları silinir;
    klasördeki diğer dosyalara (ör. README.md) dokunulmaz. ({dosya: sha256}, [silinenler]) döndürür.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes, removed = {}, []
    for name, images in splits.items():
        path = out_dir / f"{name}_images.txt"
        ids = sorted(images["image_id"].unique())
        path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in known_names:
        path = out_dir / f"{name}_images.txt"
        if name not in splits and path.exists():
            path.unlink()
            removed.append(path.name)
    return hashes, removed


def load_processed(kind, split, required=True):
    """<kind>_<split>.csv'yi okur. Dosya yoksa: required ise hata ile durur, değilse None döner."""
    path = PROCESSED_DIR / f"{kind}_{split}.csv"
    if not path.exists():
        if not required:
            return None
        sys.exit(f"HATA: {path.relative_to(ROOT).as_posix()} bulunamadı. "
                 f"Önce 'python -m src.preprocessing.build_index' çalıştırın "
                 f"(data/{split} indirilmiş olmalı).")
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["standard", "unseen-disaster"], default="standard")
    parser.add_argument("--holdout-disaster", help="unseen-disaster modunda dışarıda tutulacak afet")
    args = parser.parse_args()
    if args.mode == "unseen-disaster" and not args.holdout_disaster:
        parser.error("--mode unseen-disaster için --holdout-disaster gerekli")

    train_images = load_processed("images", "train")
    # Standart modda test seti zorunlu; unseen-disaster modunda yalnızca holdout_challenge_test için kullanılır
    test_images = load_processed("images", "test", required=args.mode == "standard")
    buildings = [load_processed("buildings", "train")]
    if test_images is not None:
        buildings.append(load_processed("buildings", "test"))
    buildings = pd.concat(buildings, ignore_index=True)

    if args.mode == "standard":
        splits = make_standard(train_images, test_images)
        out_dir = SPLITS_DIR / "standard"
    else:
        splits = make_unseen_disaster(train_images, args.holdout_disaster, test_images)
        out_dir = SPLITS_DIR / "unseen_disaster" / args.holdout_disaster
        if test_images is None:
            print("BİLGİ: images_test.csv yok; holdout yalnızca Challenge train görüntülerinden "
                  "üretildi (holdout_challenge_test_images.txt yazılmadı).")
        elif "holdout_challenge_test" not in splits:
            print(f"BİLGİ: {args.holdout_disaster} test setinde yok; holdout_challenge_test_images.txt yazılmadı.")

    building_splits = split_buildings(splits, buildings)
    check_no_leakage(splits, IMAGE_KEY)
    check_no_leakage(building_splits, BUILDING_KEY)
    print(f"Sızıntı kontrolü: OK ({tuple(IMAGE_KEY)} ve {tuple(BUILDING_KEY)} kümeleri ayrık)")
    print("\nTrain/val strata dağılımı (val_0_strata: validation örneği 0 çıkan strata)")
    print(strata_table(splits["train"], splits["val"]).to_string())
    report(splits, building_splits)

    hashes, removed = write_split_files(splits, out_dir, KNOWN_SPLITS[args.mode])
    for name in removed:
        print(f"Eski split dosyası silindi: {(out_dir / name).relative_to(ROOT).as_posix()}")
    print("\nSHA256:")
    for name, digest in hashes.items():
        print(f"  {(out_dir / name).relative_to(ROOT).as_posix()}  {digest}")


if __name__ == "__main__":
    main()
