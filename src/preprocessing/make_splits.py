"""Bina indeksinden görüntü bazında train/val/test bölmeleri üretir.

Kullanım (repo kökünden):
    python -m src.preprocessing.make_splits
    python -m src.preprocessing.make_splits --mode unseen-disaster --holdout-disaster hurricane-matthew

Çıktılar:
    splits/standard/{train,val,test}_images.txt
    splits/unseen_disaster/<afet>/{train,val,holdout_train_source,holdout_challenge_test}_images.txt
    (holdout_challenge_test yalnızca buildings_test.csv varsa ve afet test setinde bulunuyorsa yazılır)

Bölme görüntü (image_id) bazındadır: aynı görüntünün tüm binaları aynı bölmeye düşer.
Eğitimden çıkarılanlar: damage == "un-classified" ve is_bad_polygon == True.
out_of_bounds binalar tutulur (patch çıkarımında crop clamp edilir).

KURAL: Challenge test'ten gelen bölmeler (standard/test ve holdout_challenge_test) yalnızca final
değerlendirmede kullanılır; model ve hiperparametre seçimleri sadece train/val üzerinden yapılır.
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"

SEED = 42
VAL_FRACTION = 0.15
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def filter_trainable(df):
    """un-classified ve bozuk poligonları çıkarır."""
    keep = (df["damage"] != "un-classified") & ~df["is_bad_polygon"].astype(bool)
    return df[keep].reset_index(drop=True)


def split_train_val(df, val_fraction=VAL_FRACTION, seed=SEED):
    """Her afet için ayrı ayrı, görüntü bazında train/val bölmesi. (train_ids, val_ids) döndürür."""
    train_ids, val_ids = [], []
    for disaster in sorted(df["disaster"].unique()):
        ids = sorted(df.loc[df["disaster"] == disaster, "image_id"].unique())
        # Her afet kendi seed'li karıştırıcısını kullanır: bir afetin eklenip çıkarılması diğerlerini etkilemez
        random.Random(seed).shuffle(ids)
        n_val = round(len(ids) * val_fraction)
        val_ids += ids[:n_val]
        train_ids += ids[n_val:]
    return sorted(train_ids), sorted(val_ids)


def make_standard(train_df, test_df, val_fraction=VAL_FRACTION, seed=SEED):
    """Challenge train → train/val, Challenge test → test. {bölme: bina tablosu} döndürür."""
    train_df, test_df = filter_trainable(train_df), filter_trainable(test_df)
    train_ids, val_ids = split_train_val(train_df, val_fraction, seed)
    return {
        "train": train_df[train_df["image_id"].isin(train_ids)],
        "val": train_df[train_df["image_id"].isin(val_ids)],
        "test": test_df,
    }


def make_unseen_disaster(train_df, holdout_disaster, test_df=None, val_fraction=VAL_FRACTION, seed=SEED):
    """Holdout afetin tüm görüntüleri eğitimden çıkarılır; kalan afetler train/val'e bölünür.

    Holdout iki ayrı bölme olarak tutulur (tasarım kararı, review'da değişebilir):
    holdout_train_source = afetin Challenge train görüntüleri, holdout_challenge_test = Challenge test görüntüleri.
    test_df verilmezse ya da afet test setinde yoksa holdout_challenge_test bölmesi üretilmez.
    """
    train_df = filter_trainable(train_df)
    if holdout_disaster not in set(train_df["disaster"]):
        raise ValueError(f"Bilinmeyen afet: {holdout_disaster}. Seçenekler: {sorted(train_df['disaster'].unique())}")
    is_holdout = train_df["disaster"] == holdout_disaster
    rest = train_df[~is_holdout]
    train_ids, val_ids = split_train_val(rest, val_fraction, seed)
    splits = {
        "train": rest[rest["image_id"].isin(train_ids)],
        "val": rest[rest["image_id"].isin(val_ids)],
        "holdout_train_source": train_df[is_holdout],
    }
    if test_df is not None:
        test_df = filter_trainable(test_df)
        holdout_challenge_test = test_df[test_df["disaster"] == holdout_disaster]
        if len(holdout_challenge_test):
            splits["holdout_challenge_test"] = holdout_challenge_test
    return splits


def check_no_leakage(splits):
    """Bölmeler arasında görüntü veya bina sızıntısı olmadığını doğrular."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared_images = set(splits[a]["image_id"]) & set(splits[b]["image_id"])
            assert not shared_images, f"{a} ve {b} ortak görüntü içeriyor: {sorted(shared_images)[:5]}"
            keys_a = set(zip(splits[a]["image_id"], splits[a]["uid"]))
            keys_b = set(zip(splits[b]["image_id"], splits[b]["uid"]))
            assert not keys_a & keys_b, f"{a} ve {b} ortak (image_id, uid) içeriyor"


def report(splits):
    for name, df in splits.items():
        print(f"\n=== {name}: {df['image_id'].nunique()} görüntü, {len(df)} bina ===")
        counts = df["damage"].value_counts().reindex(CLASSES, fill_value=0)
        pct = (counts / max(len(df), 1) * 100).round(1)
        print(pd.DataFrame({"bina": counts, "%": pct}).to_string())
        disasters = df.groupby("disaster").agg(goruntu=("image_id", "nunique"), bina=("uid", "size"))
        print(disasters.to_string())


def write_split_files(splits, out_dir):
    """Her bölme için <bölme>_images.txt yazar (sıralı, satır başına bir image_id)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, df in splits.items():
        path = out_dir / f"{name}_images.txt"
        ids = sorted(df["image_id"].unique())
        path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_buildings(split, required=True):
    """buildings_<split>.csv'yi okur. Dosya yoksa: required ise hata ile durur, değilse None döner."""
    path = PROCESSED_DIR / f"buildings_{split}.csv"
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

    train_df = load_buildings("train")
    # Standart modda test seti zorunlu; unseen-disaster modunda yalnızca holdout_challenge_test için kullanılır
    test_df = load_buildings("test", required=args.mode == "standard")

    if args.mode == "standard":
        splits = make_standard(train_df, test_df)
        out_dir = SPLITS_DIR / "standard"
    else:
        splits = make_unseen_disaster(train_df, args.holdout_disaster, test_df)
        out_dir = SPLITS_DIR / "unseen_disaster" / args.holdout_disaster
        if test_df is None:
            print("BİLGİ: buildings_test.csv yok; holdout yalnızca Challenge train görüntülerinden "
                  "üretildi (holdout_challenge_test_images.txt yazılmadı).")
        elif "holdout_challenge_test" not in splits:
            print(f"BİLGİ: {args.holdout_disaster} test setinde yok; holdout_challenge_test_images.txt yazılmadı.")

    check_no_leakage(splits)
    print("Sızıntı kontrolü: OK (görüntü ve (image_id, uid) kümeleri ayrık)")
    report(splits)

    print("\nSHA256:")
    for path, digest in write_split_files(splits, out_dir).items():
        print(f"  {path}  {digest}")


if __name__ == "__main__":
    main()
