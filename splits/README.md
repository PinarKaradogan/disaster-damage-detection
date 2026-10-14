# Splits

Image-level splits of the xBD Challenge data. Each `*_images.txt` file lists one `image_id`
per line, sorted. All buildings of an image belong to the same split.

## `splits/standard/`

- `train_images.txt`, `val_images.txt`: from the Challenge train set, split 85% / 15% by image,
  separately for each disaster, seed 42.
- `test_images.txt`: the whole Challenge test set.

## `splits/unseen_disaster/<disaster>/`

- `train_images.txt`, `val_images.txt`: Challenge train set without the holdout disaster,
  same 85% / 15% rule.
- `holdout_train_source_images.txt`: Challenge train images of the holdout disaster.
- `holdout_challenge_test_images.txt`: Challenge test images of the holdout disaster.

## Filters

- Removed: buildings with `damage == "un-classified"` and buildings with `is_bad_polygon == True`.
- Kept: `out_of_bounds` buildings. During patch extraction the crop size is preserved and the area
  outside the image is filled with black (0) padding; crops are not shrunk and the aspect ratio is kept.

## Rule

Files that come from the Challenge test set (`standard/test_images.txt` and
`holdout_challenge_test_images.txt`) are used **only for the final evaluation**.
Model selection, augmentation, learning rate and all other hyperparameter choices are made
on train / val only.

## How to generate

From the repository root:

```bash
python -m src.preprocessing.build_index
python -m src.preprocessing.make_splits
python -m src.preprocessing.make_splits --mode unseen-disaster --holdout-disaster <disaster>
```

`build_index` writes `data/processed/buildings_{train,test}.csv`, which `make_splits` reads.
