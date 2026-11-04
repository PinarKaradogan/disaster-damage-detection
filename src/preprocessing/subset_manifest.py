"""Seeded patch-level (disaster, damage) train subset; preserve all validation rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import LABEL_MAP
from src.data.dataset import REQUIRED_COLUMNS, ROOT
from src.training.artifacts import file_sha256
from src.training.config import TrainingConfig
from src.training.data import build_loaders


def allocate_counts(counts, target):
    """Closest proportional integer quotas, bounded by [1, stratum size].

    Start at floor(ideal), clipped to one. Add/remove the largest deficit/excess
    until the exact target is reached. Integer arithmetic and sorted strata break ties.
    """
    sizes = np.asarray(counts, dtype=np.int64)
    if sizes.ndim != 1 or not len(sizes) or (sizes < 1).any():
        raise ValueError("Strata must have positive sizes")
    if not isinstance(target, int) or not len(sizes) <= target <= int(sizes.sum()):
        raise ValueError("Target must fit the train population and represent every stratum")
    total = int(sizes.sum())
    quotas = np.maximum(1, sizes * target // total)
    while int(quotas.sum()) < target:
        candidates = np.flatnonzero(quotas < sizes)
        index = candidates[np.argmax((sizes * target - quotas * total)[candidates])]
        quotas[index] += 1
    while int(quotas.sum()) > target:
        candidates = np.flatnonzero(quotas > 1)
        index = candidates[np.argmax((quotas * total - sizes * target)[candidates])]
        quotas[index] -= 1
    return quotas


def select_train(train, *, target, random_state=42):
    if train.empty or not train["split"].eq("train").all():
        raise ValueError("Sampling requires train rows only")
    # Canonical population order makes selected keys independent of input row order.
    ordered = train.sort_values(["disaster", "damage", "image_id", "uid"])
    groups = list(ordered.groupby(["disaster", "damage"], sort=True))
    quotas = allocate_counts([len(group) for _, group in groups], target)
    rng = np.random.RandomState(random_state)
    indices = []
    strata = []
    for ((disaster, damage), group), quota in zip(groups, quotas):
        indices.extend(rng.choice(group.index.to_numpy(), size=int(quota), replace=False).tolist())
        strata.append(dict(disaster=disaster, damage=damage, source_train=len(group),
                           subset_train=int(quota), ideal=len(group) * target / len(train),
                           minimum_one_adjustment=len(group) * target < len(train)))
    return train.loc[sorted(indices)].copy(), strata


def _counts(frame, column):
    return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}


def _image_stats(counts):
    retained = counts[counts > 0]
    return dict(total_images=len(counts), represented_images=len(retained),
                omitted_images=int((counts == 0).sum()),
                images_with_multiple_patches=int((counts > 1).sum()),
                patches_per_represented_image=dict(
                    min=int(retained.min()), median=float(retained.median()),
                    p90=float(retained.quantile(.9)), max=int(retained.max())),
                patch_count_histogram={str(key): int(value)
                                       for key, value in counts.value_counts().sort_index().items()})


def create_subset(source, output, *, target=33507, random_state=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.parent == output.parent or output.suffix != ".csv":
        raise ValueError("Use a separate output directory and a .csv manifest")
    source_hash = file_sha256(source)
    # Preserve every cell's original string representation (coordinates, paths, IDs).
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = REQUIRED_COLUMNS | {"disaster"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(frame.columns))}")
    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unsupported source split")
    train, val = frame[frame["split"].eq("train")], frame[frame["split"].eq("val")]
    retained = pd.concat([train, val])
    if train.empty or val.empty or retained[list(required)].eq("").any().any():
        raise ValueError("Train/validation rows and required fields must be nonempty")
    if not retained["damage"].isin(LABEL_MAP).all():
        raise ValueError("Unknown train/validation damage class")
    if retained.duplicated(["split", "image_id", "uid"]).any():
        raise ValueError("Duplicate building keys")
    # Test image IDs are used only for a leakage audit, never allocation/selection.
    images = {name: set(frame.loc[frame["split"].eq(name), "image_id"])
              for name in ["train", "val", "test"]}
    overlaps = {f"{a}_{b}": len(images[a] & images[b])
                for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    if any(overlaps.values()):
        raise ValueError(f"Source image_id split leakage: {overlaps}")
    selected, strata = select_train(train, target=target, random_state=random_state)
    subset = pd.concat([selected, val], ignore_index=True)
    if len(selected) != target or selected.duplicated(["image_id", "uid"]).any():
        raise ValueError("Subset size or uniqueness check failed")
    pd.testing.assert_frame_equal(subset[subset["split"].eq("val")].reset_index(drop=True),
                                  val.reset_index(drop=True))
    image_table = train.groupby(["disaster", "image_id"], sort=True).size().rename("source_train_patches").to_frame()
    image_table["subset_train_patches"] = selected.groupby(["disaster", "image_id"]).size().reindex(
        image_table.index, fill_value=0)
    image_table["selected_fraction"] = image_table["subset_train_patches"] / image_table["source_train_patches"]
    manifest_bytes = subset.to_csv(index=False, lineterminator="\n").encode("utf-8")
    image_bytes = image_table.reset_index().to_csv(index=False, lineterminator="\n").encode("utf-8")
    classes = []
    for damage in LABEL_MAP:
        full_count, count = int(train["damage"].eq(damage).sum()), int(selected["damage"].eq(damage).sum())
        classes.append(dict(damage=damage, source_train=full_count, subset_train=count,
                            source_fraction=full_count / len(train), subset_fraction=count / target,
                            change_percentage_points=100 * (count / target - full_count / len(train))))
    summary = dict(
        schema_version=1, source_manifest=str(source), source_sha256=source_hash,
        manifest=str(output), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        seed=random_state, target_train=target, actual_train_fraction=target / len(train),
        method="patch-level stratified sampling without replacement; proportional integer quotas, minimum one per existing stratum",
        source_split_counts=_counts(frame, "split"), output_split_counts=_counts(subset, "split"),
        disaster_distribution=[dict(disaster=name, source_train=int(train["disaster"].eq(name).sum()),
                                     subset_train=int(selected["disaster"].eq(name).sum()),
                                     validation=int(val["disaster"].eq(name).sum()))
                               for name in sorted(retained["disaster"].unique())],
        class_distribution=classes, validation_class_distribution=_counts(val, "damage"),
        strata=strata, source_image_counts={name: len(ids) for name, ids in images.items()},
        output_image_counts=dict(train=selected["image_id"].nunique(), val=val["image_id"].nunique()),
        source_image_overlaps=overlaps,
        subset_train_val_image_overlap=len(set(selected["image_id"]) & images["val"]),
        source_train_image_distribution=_image_stats(image_table["source_train_patches"]),
        subset_train_image_distribution=_image_stats(image_table["subset_train_patches"]),
        validation_image_distribution=_image_stats(val.groupby("image_id").size()),
        validation_rows_unchanged=True, patch_paths_unchanged=True, test_rows_included=0,
        comparison_note="ResNet34 trained on this subset and Simple CNN trained on full data have unequal training data budgets; this is not a controlled architecture comparison. Building patches from the same train image can be partially selected; the original image-level split is unchanged.",
        versions=dict(numpy=np.__version__, pandas=pd.__version__),
    )
    artifacts = {output: manifest_bytes, output.parent / "image_distribution.csv": image_bytes}
    # Never overwrite different existing artifacts; identical reruns are no-ops.
    for path, content in artifacts.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different subset artifact: {path}")
    if file_sha256(source) != source_hash:
        raise ValueError("Source manifest changed while selecting subset")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
    # Construct the real training loaders, without reading images or training a model.
    loaders = build_loaders(TrainingConfig("resnet34_early_fusion", num_workers=0), output,
                            project_root=ROOT)
    summary["loader_compatibility"] = dict(train=len(loaders.train.dataset), val=len(loaders.val.dataset),
                                            class_counts=loaders.train_class_counts,
                                            dataset_and_build_loaders="passed; no image iteration")
    summary_path = output.parent / "summary.json"
    summary_bytes = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if summary_path.exists() and summary_path.read_bytes() != summary_bytes:
        raise FileExistsError(f"Refusing to overwrite different summary: {summary_path}")
    if not summary_path.exists():
        with summary_path.open("xb") as handle:
            handle.write(summary_bytes)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/patches_full/patches.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/patches_subset25/patches.csv")
    parser.add_argument("--target-train", type=int, default=33507)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    summary = create_subset(args.source, args.output, target=args.target_train, random_state=args.random_state)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
