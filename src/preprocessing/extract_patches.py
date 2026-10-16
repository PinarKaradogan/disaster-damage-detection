"""Extract native-size paired patches using the committed standard split lists.

Example: python -m src.preprocessing.extract_patches --disaster hurricane-matthew
Use --all-disasters explicitly for a full run. Challenge Test is final evaluation only.
An output root must not exist: no append, overwrite or recursive cleanup is performed.
Interrupted runs retain .incomplete; patches.csv is published only after every pair succeeds.
"""
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import pandas as pd
from PIL import Image

from src.preprocessing.build_index import eligible_mask, write_csv_atomic

ROOT = Path(__file__).resolve().parents[2]
SPLIT_SOURCES = {"train": "train", "val": "train", "test": "test"}
MANIFEST_COLUMNS = [
    "image_id", "uid", "disaster", "damage", "split", "pre_patch", "post_patch",
    "x_min", "y_min", "x_max", "y_max", "patch_side",
    "pad_left", "pad_top", "pad_right", "pad_bottom",
]
INPUT_COLUMNS = ["source_split", "image_id", "uid", "disaster", "damage",
                 "x_min", "y_min", "x_max", "y_max", "width", "height", "is_bad_polygon"]


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def side(self):
        return self.right - self.left

    @property
    def box(self):
        return self.left, self.top, self.right, self.bottom


def patch_geometry(x_min, y_min, x_max, y_max, width, height):
    """Compute the one integer crop shared by PRE and POST, without resizing."""
    values = (x_min, y_min, x_max, y_max, width, height)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite bbox/dimensions")
    if width <= 0 or height <= 0 or x_max <= x_min or y_max <= y_min:
        raise ValueError("Non-positive bbox/dimensions")
    # Index coordinates and dimensions are independently serialized to six decimals.
    if not (math.isclose(x_max - x_min, width, abs_tol=2e-6, rel_tol=0)
            and math.isclose(y_max - y_min, height, abs_tol=2e-6, rel_tol=0)):
        raise ValueError("BBox and width/height disagree")
    side = max(64, math.ceil(2 * max(width, height)))
    left = math.floor((x_min + x_max) / 2 - side / 2)
    top = math.floor((y_min + y_max) / 2 - side / 2)
    return Crop(left, top, left + side, top + side)


def padding_for(crop, image_size):
    """Return actual black border widths (left, top, right, bottom)."""
    width, height = image_size
    if width <= 0 or height <= 0 or crop.side <= 0 or crop.bottom - crop.top != crop.side:
        raise ValueError("Invalid image size or square crop")
    side = crop.side
    return (min(side, max(0, -crop.left)), min(side, max(0, -crop.top)),
            min(side, max(0, crop.right - width)), min(side, max(0, crop.bottom - height)))


def crop_with_padding(image, crop):
    """Paste the image intersection into a black RGB square at its original offset."""
    padding_for(crop, image.size)
    patch = Image.new("RGB", (crop.side, crop.side), (0, 0, 0))
    left, top = max(0, crop.left), max(0, crop.top)
    right, bottom = min(image.width, crop.right), min(image.height, crop.bottom)
    if left < right and top < bottom:
        patch.paste(image.crop((left, top, right, bottom)).convert("RGB"),
                    (left - crop.left, top - crop.top))
    return patch


def paired_crops(pre, post, crop):
    if pre.size != post.size:
        raise ValueError(f"PRE/POST dimensions differ: {pre.size} vs {post.size}")
    return crop_with_padding(pre, crop), crop_with_padding(post, crop)


def safe_token(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Missing or unsafe {field}: {value!r}")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                         *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"Reserved Windows filename in {field}: {value!r}")
    return value


def context(row):
    return f"image_id={row['image_id']} uid={row['uid']} split={row['split']}"


def load_selection(root, disaster=None):
    """Read both versioned building indexes and all three source-of-truth TXT files."""
    tables = {}
    for source in ["train", "test"]:
        path = root / "data" / "processed" / f"buildings_{source}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing buildings CSV: {path}")
        table = pd.read_csv(path, dtype={c: str for c in
                            ["source_split", "image_id", "uid", "disaster", "damage"]})
        missing = set(INPUT_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        if not table["source_split"].eq(source).all():
            raise ValueError(f"{path.name}: unexpected source_split")
        # Do not let astype(bool) silently interpret the string 'False' as True.
        if not pd.api.types.is_bool_dtype(table["is_bad_polygon"]):
            raise ValueError(f"{path.name}: is_bad_polygon must contain only True/False")
        tables[source] = table
    selections, seen = [], set()
    for split, source in SPLIT_SOURCES.items():
        path = root / "splits" / "standard" / f"{split}_images.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split={split}: {path}")
        ids = path.read_text(encoding="utf-8").splitlines()
        for image_id in ids:
            safe_token(image_id, "split image_id")
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"Empty, duplicate or overlapping image IDs: split={split}")
        seen.update(ids)
        table = tables[source]
        chosen = table[table["image_id"].isin(ids)]
        if disaster is not None:
            chosen = chosen[chosen["disaster"] == disaster]
        chosen = chosen[eligible_mask(chosen)].copy()
        chosen["split"] = split
        selections.append(chosen)
    selected = pd.concat(selections, ignore_index=True)
    if selected.empty:
        raise ValueError(f"No eligible buildings for disaster={disaster!r}")
    if selected.duplicated(["split", "image_id", "uid"]).any():
        row = selected[selected.duplicated(["split", "image_id", "uid"], keep=False)].iloc[0]
        raise ValueError(f"Duplicate manifest key: {context(row)}")
    selected["_order"] = selected["split"].map({s: n for n, s in enumerate(SPLIT_SOURCES)})
    return selected.sort_values(["_order", "image_id", "uid"]).drop(columns="_order").reset_index(drop=True)


def source_paths(root, row):
    split = row["split"]
    if split not in SPLIT_SOURCES:
        raise ValueError(f"Unexpected split: {context(row)}")
    directory = root / "data" / SPLIT_SOURCES[split] / "images"
    return tuple(directory / f"{row['image_id']}_{phase}_disaster.png" for phase in ["pre", "post"])


def preflight(root, selection, output_root):
    """Validate all geometry, source headers, dimensions and target names before writing."""
    sizes, targets, plan = {}, set(), []
    for row in selection.to_dict("records"):
        try:
            for field in ["image_id", "uid", "disaster"]:
                safe_token(row[field], field)
            paths = source_paths(root, row)
            crop = patch_geometry(*(row[c] for c in ["x_min", "y_min", "x_max", "y_max", "width", "height"]))
            if paths not in sizes:
                pair_sizes = []
                for path in paths:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing source image: {path}")
                    with Image.open(path) as image:
                        pair_sizes.append(image.size)
                if pair_sizes[0] != pair_sizes[1]:
                    raise ValueError(f"PRE/POST dimensions differ: {pair_sizes}")
                sizes[paths] = pair_sizes[0]
            padding = padding_for(crop, sizes[paths])
            record = {c: row[c] for c in ["image_id", "uid", "disaster", "damage", "split",
                                         "x_min", "y_min", "x_max", "y_max"]}
            record.update(patch_side=crop.side, **dict(zip(
                ["pad_left", "pad_top", "pad_right", "pad_bottom"], padding)))
            for phase in ["pre", "post"]:
                target = output_root / row["split"] / row["disaster"] / f"{row['image_id']}__{row['uid']}_{phase}.png"
                key = target.as_posix().casefold()
                if key in targets or target.exists():
                    raise FileExistsError(f"Output file collision: {target}")
                targets.add(key)
                record[f"{phase}_patch"] = target.relative_to(root).as_posix()
            plan.append((record, crop, paths))
        except (ValueError, OSError, TypeError) as exc:
            raise ValueError(f"{context(row)}: {exc}") from exc
    return plan


def extract_patches(root=ROOT, disaster=None, *, all_disasters=False, output_root=None):
    """Execute one fresh run. Existing or interrupted output roots are never reused."""
    if (disaster is None) == (not all_disasters):
        raise ValueError("Choose exactly one: disaster or all_disasters=True")
    root = Path(root).resolve()
    if disaster is not None:
        safe_token(disaster, "disaster")
    output_root = Path(output_root) if output_root is not None else Path("data/processed/patches")
    output_root = (root / output_root).resolve()
    processed = (root / "data" / "processed").resolve()
    if not output_root.is_relative_to(processed) or output_root == processed:
        raise ValueError("Output root must be a new directory strictly under data/processed")
    if output_root.exists():
        raise FileExistsError(f"Output root already exists; choose a fresh --output-root: {output_root}")
    selected = load_selection(root, disaster)
    plan = preflight(root, selected, output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    marker = output_root / ".incomplete"
    marker.write_text("Incomplete extraction; do not consume this directory.\n", encoding="utf-8")
    # Selection order groups split/image together, so each source pair is decoded once.
    cached_paths, pre, post = None, None, None
    try:
        for record, crop, paths in plan:
            try:
                if paths != cached_paths:
                    if pre is not None:
                        pre.close()
                        post.close()
                    with Image.open(paths[0]) as image:
                        pre = image.convert("RGB")
                    with Image.open(paths[1]) as image:
                        post = image.convert("RGB")
                    cached_paths = paths
                pre_patch, post_patch = paired_crops(pre, post, crop)
                for phase, patch in [("pre", pre_patch), ("post", post_patch)]:
                    target = root / record[f"{phase}_patch"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as handle:
                        patch.save(handle, format="PNG")
                    patch.close()
            except (ValueError, OSError) as exc:
                raise ValueError(f"{context(record)}: {exc}") from exc
        manifest = pd.DataFrame([record for record, _, _ in plan], columns=MANIFEST_COLUMNS)
        write_csv_atomic(manifest, output_root / "patches.csv")
        marker.unlink()
    finally:
        if pre is not None:
            pre.close()
        if post is not None:
            post.close()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--disaster")
    scope.add_argument("--all-disasters", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/patches"))
    args = parser.parse_args(argv)
    try:
        manifest = extract_patches(disaster=args.disaster, all_disasters=args.all_disasters,
                                   output_root=args.output_root)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Wrote {len(manifest)} patch pairs to {args.output_root.as_posix()}/patches.csv")
    print(manifest.groupby("split", sort=False).size().to_string())


if __name__ == "__main__":
    main()
