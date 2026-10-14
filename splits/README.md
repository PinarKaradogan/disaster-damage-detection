# Splits

Split files are image-level and contain all images of the corresponding Challenge set
(including images without buildings). Each `*_images.txt` file lists one `image_id` per line,
sorted. All buildings of an image belong to the same split.

The building eligibility filter (`un-classified`, `is_bad_polygon`) is applied only at the patch
extraction stage; images without buildings or without eligible buildings produce no patches.
The same split boundaries are used by all models (patch classifiers and U-Net).

## Train / validation stratification

Inside each disaster, images are split into two strata by the **raw** number of building
annotations in the post-disaster label:

1. `n_buildings > 0`
2. `n_buildings == 0`

`damage`, `un-classified`, `is_bad_polygon` and patch eligibility do **not** affect this grouping.

For each (disaster, stratum) group: the `image_id` list is `sorted()`, shuffled deterministically
with seed 42, and the first `n_val` images go to validation, the rest to train, where

```
n_val = floor(0.15 * n + 0.5)        # n = images in the group; halves round up
```

The value is computed exactly (as a fraction), so it does not depend on floating-point rounding.
In small groups `n_val` can be 0; the fraction is not forced, the result stays deterministic and
`make_splits` marks such groups in its report (`val_0_strata`).

The rule is used for the standard train / validation split and, in unseen-disaster mode, for all
disasters except the holdout disaster. The holdout disaster goes entirely to its own split and never
appears in train / validation.

## `splits/standard/`

- `train_images.txt`, `val_images.txt`: all images of the Challenge train set, split by the
  stratification rule above.
- `test_images.txt`: all images of the Challenge test set.

### Class distribution per split

Splits are made by image, not by building, so building-level class ratios differ somewhat between
splits. Eligible buildings (without `un-classified` and `is_bad_polygon`) in the standard split:

| Split | Images | Eligible buildings | no-damage | minor-damage | major-damage | destroyed |
|---|---|---|---|---|---|---|
| train | 2380 | 134,028 | 99,573 (74.3%) | 12,060 (9.0%) | 12,064 (9.0%) | 10,331 (7.7%) |
| val | 419 | 25,615 | 17,736 (69.2%) | 2,916 (11.4%) | 2,076 (8.1%) | 2,887 (11.3%) |
| test | 933 | 53,806 | 41,391 (76.9%) | 4,793 (8.9%) | 3,847 (7.1%) | 3,775 (7.0%) |

Seed 42 was fixed before any split result was seen and is not tuned to balance these ratios.

## `splits/unseen_disaster/<disaster>/`

- `train_images.txt`, `val_images.txt`: all Challenge train images except the holdout disaster,
  same stratification rule.
- `holdout_train_source_images.txt`: all Challenge train images of the holdout disaster.
- `holdout_challenge_test_images.txt`: all Challenge test images of the holdout disaster
  (written only when the test inventory exists and contains the disaster).

When a folder is regenerated, split files that the run did not produce (for example an old
`holdout_challenge_test_images.txt` from a run with the test set) are removed. Other files in the
folder, such as this README, are not touched.

## Patch extraction filters

- Not used for patches: buildings with `damage == "un-classified"` and buildings with
  `is_bad_polygon == True`.
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

`build_index` writes `data/processed/images_{train,test}.csv` (the split universe, one row per
image) and `data/processed/buildings_{train,test}.csv` (one row per building, used for reports and
patch extraction). `make_splits` reads both.
