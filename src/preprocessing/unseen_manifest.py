"""Phase 6 unseen-disaster manifest: relabel existing patch rows by the Phase 3 holdout split files.

python -m src.preprocessing.unseen_manifest --holdout-disaster hurricane-matthew

Reads data/processed/patches_full/patches.csv and splits/unseen_disaster/<disaster>/*.txt and writes
data/processed/patches_unseen_<disaster>/patches.csv (+ report.json). No image is read or copied:
patch paths keep pointing at patches_full. Split column:
- train / val: images listed in the unseen-disaster train/val files (the holdout disaster is absent);
- test: every image of the holdout disaster (``holdout_source`` = train_source | challenge_test).
Rows of other disasters' Challenge test images are dropped (not used by this experiment).
The build fails if any train/val row belongs to the holdout disaster or an image is in two sets.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.dataset import ROOT
from src.training.artifacts import file_sha256, write_json

SOURCE = "data/processed/patches_full/patches.csv"


def read_ids(path):
    ids = Path(path).read_text(encoding="utf-8").split()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image ids in {path}")
    return set(ids)


def build_unseen_manifest(disaster, *, project_root=ROOT, source=SOURCE):
    root = Path(project_root)
    split_dir = root / "splits/unseen_disaster" / disaster
    sets = {name: read_ids(split_dir / f"{name}.txt") for name in
            ("train_images", "val_images", "holdout_train_source_images", "holdout_challenge_test_images")}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f"Image ids shared by {a} and {b}")
    frame = pd.read_csv(root / source)
    assignment = {}
    for name, split, holdout in (("train_images", "train", ""), ("val_images", "val", ""),
                                 ("holdout_train_source_images", "test", "train_source"),
                                 ("holdout_challenge_test_images", "test", "challenge_test")):
        for image_id in sets[name]:
            assignment[image_id] = (split, holdout)
    frame = frame[frame["image_id"].isin(assignment)].copy()
    frame["source_split"] = frame["split"]
    frame["split"] = frame["image_id"].map(lambda i: assignment[i][0])
    frame["holdout_source"] = frame["image_id"].map(lambda i: assignment[i][1])
    trainval = frame[frame["split"].isin(["train", "val"])]
    holdout = frame[frame["split"] == "test"]
    if (trainval["disaster"] == disaster).any():
        raise ValueError(f"{disaster} rows found in train/val")
    if not (holdout["disaster"] == disaster).all():
        raise ValueError("Holdout contains other disasters")
    if (holdout["holdout_source"] == "challenge_test").any() and \
            not (holdout.loc[holdout["holdout_source"] == "challenge_test", "source_split"] == "test").all():
        raise ValueError("Challenge-test holdout rows must come from the Challenge test split")
    report = dict(
        holdout_disaster=disaster, source=source, source_sha256=file_sha256(root / source),
        split_files_sha256={n: file_sha256(split_dir / f"{n}.txt") for n in names},
        rows={s: int((frame["split"] == s).sum()) for s in ("train", "val", "test")},
        holdout_rows={k: int((holdout["holdout_source"] == k).sum()) for k in ("train_source", "challenge_test")},
        images={s: int(frame.loc[frame["split"] == s, "image_id"].nunique()) for s in ("train", "val", "test")},
        trainval_contains_holdout_disaster=False,
        disasters_in_trainval=sorted(trainval["disaster"].unique().tolist()),
        damage_counts={s: frame[frame["split"] == s]["damage"].value_counts().sort_index().to_dict()
                       for s in ("train", "val", "test")},
    )
    return frame.reset_index(drop=True), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-disaster", required=True)
    args = parser.parse_args(argv)
    out_dir = ROOT / "data/processed" / f"patches_unseen_{args.holdout_disaster}"
    if (out_dir / "patches.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir / 'patches.csv'}")
    frame, report = build_unseen_manifest(args.holdout_disaster)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "patches.csv", index=False)
    report["output_sha256"] = file_sha256(out_dir / "patches.csv")
    write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
