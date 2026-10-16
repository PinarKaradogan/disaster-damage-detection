"""Synthetic extraction tests; no real dataset patch extraction is run here."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.preprocessing import extract_patches as ep


def geometry(box):
    left, top, right, bottom = box
    return ep.patch_geometry(*box, right - left, bottom - top)


@pytest.mark.parametrize("box, expected", [
    ((10, 10, 20, 20), (-17, -17, 47, 47)),
    ((100, 100, 140, 130), (80, 75, 160, 155)),
    ((100, 100, 130, 140), (75, 80, 155, 160)),
    ((0, 0, 32.25, 10), (-17, -28, 48, 37)),
    ((0.5, 0.5, 10.5, 10.5), (-27, -27, 37, 37)),
    ((0, 0, 1010, 200), (-505, -910, 1515, 1110)),
])
def test_geometry_minimum_ceil_center_floor_and_fixed_side(box, expected):
    crop = geometry(box)
    assert crop.box == expected
    assert crop.right == crop.left + crop.side
    assert crop.bottom == crop.top + crop.side
    assert crop.side == max(64, math.ceil(2 * max(box[2] - box[0], box[3] - box[1])))


def test_real_florence_geometry_regression():
    crop = ep.patch_geometry(-0.000008, 509.903268, 33.753849, 546.722924, 33.753857, 36.819656)
    assert crop.side == 74 and crop.box == (-21, 491, 53, 565)
    assert ep.padding_for(crop, (1024, 1024)) == (21, 0, 0, 0)


@pytest.mark.parametrize("box", [(float('nan'), 0, 10, 10), (0, 0, float('inf'), 10),
                                    (0, 0, 0, 10), (10, 0, 0, 10)])
def test_invalid_geometry_rejected(box):
    with pytest.raises(ValueError):
        geometry(box)


def test_inconsistent_dimensions_rejected():
    with pytest.raises(ValueError, match="disagree"):
        ep.patch_geometry(0, 0, 10, 10, 11, 10)


def rgb_pattern(width=13, height=9):
    data = np.arange(width * height * 3, dtype=np.uint16).reshape(height, width, 3)
    return Image.fromarray((data % 254 + 1).astype(np.uint8))


@pytest.mark.parametrize("box, padding", [
    ((2, 1, 6, 5), (0, 0, 0, 0)),
    ((-2, 1, 4, 7), (2, 0, 0, 0)),
    ((2, -2, 8, 4), (0, 2, 0, 0)),
    ((10, 1, 16, 7), (0, 0, 3, 0)),
    ((2, 6, 8, 12), (0, 0, 0, 3)),
    ((-2, -3, 14, 13), (2, 3, 1, 4)),
    ((20, 20, 26, 26), (0, 0, 6, 6)),
    ((-20, -20, -14, -14), (6, 6, 0, 0)),
])
def test_padding_pixels_and_content_position(box, padding):
    image = rgb_pattern()
    crop = ep.Crop(*box)
    patch = ep.crop_with_padding(image, crop)
    assert patch.mode == "RGB" and patch.size == (crop.side, crop.side)
    assert ep.padding_for(crop, image.size) == padding
    actual, source = np.asarray(patch), np.asarray(image)
    # An independent pixel oracle tests every content pixel and every black pixel.
    for y in range(crop.side):
        for x in range(crop.side):
            sx, sy = x + crop.left, y + crop.top
            expected = source[sy, sx] if 0 <= sx < image.width and 0 <= sy < image.height else [0, 0, 0]
            np.testing.assert_array_equal(actual[y, x], expected)


def test_paired_crops_use_identical_geometry():
    pre = rgb_pattern()
    post = Image.fromarray(255 - np.asarray(pre))
    crop = ep.Crop(-2, 1, 4, 7)
    a, b = ep.paired_crops(pre, post, crop)
    assert a.size == b.size == (6, 6)
    np.testing.assert_array_equal(np.asarray(a)[:, :2], 0)
    np.testing.assert_array_equal(np.asarray(b)[:, :2], 0)
    np.testing.assert_array_equal(np.asarray(a)[:, 2:].astype(int) + np.asarray(b)[:, 2:], 255)


def row(image_id, uid, source="train", **overrides):
    result = dict(source_split=source, image_id=image_id, uid=uid, disaster="event-a",
                  damage="no-damage", x_min=10.0, y_min=10.0, x_max=20.0, y_max=20.0,
                  width=10.0, height=10.0, is_bad_polygon=False, out_of_bounds=False)
    result.update(overrides)
    return result


def make_project(root):
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    train = pd.DataFrame([
        row("train_b", "z"), row("train_a", "b", out_of_bounds=True),
        row("train_a", "bad", is_bad_polygon=True), row("train_a", "un", damage="un-classified"),
        row("val_a", "a", damage="major-damage"), row("not_selected", "a"),
        row("train_a", "other", disaster="event-other"),
    ])
    test = pd.DataFrame([row("test_a", "a", "test", damage="destroyed")])
    for source, table in [("train", train), ("test", test)]:
        table.to_csv(processed / f"buildings_{source}.csv", index=False)
    split_dir = root / "splits" / "standard"
    split_dir.mkdir(parents=True)
    for split, ids in [("train", ["train_b", "train_a"]), ("val", ["val_a"]), ("test", ["test_a"])]:
        (split_dir / f"{split}_images.txt").write_text("".join(i + "\n" for i in ids), encoding="utf-8")
        source_dir = root / "data" / ep.SPLIT_SOURCES[split] / "images"
        source_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            for phase in ["pre", "post"]:
                rgb_pattern(40, 30).save(source_dir / f"{image_id}_{phase}_disaster.png")
    return root


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "project")


def test_extraction_eligibility_names_native_sizes_manifest_and_input_preservation(project):
    inputs = {p: p.read_bytes() for directory in [project / "data" / "processed", project / "splits"]
              for p in directory.rglob('*') if p.is_file()}
    manifest = ep.extract_patches(project, "event-a")
    assert list(manifest.columns) == ep.MANIFEST_COLUMNS
    assert list(zip(manifest.split, manifest.image_id, manifest.uid)) == [
        ("train", "train_a", "b"), ("train", "train_b", "z"),
        ("val", "val_a", "a"), ("test", "test_a", "a")]
    assert not manifest.duplicated(["split", "image_id", "uid"]).any()
    assert manifest.patch_side.tolist() == [64] * 4
    for record in manifest.to_dict("records"):
        for phase in ["pre", "post"]:
            path = Path(record[f"{phase}_patch"])
            assert not path.is_absolute() and '\\' not in str(record[f"{phase}_patch"])
            assert path.name == f"{record['image_id']}__{record['uid']}_{phase}.png"
            with Image.open(project / path) as image:
                assert image.size == (record["patch_side"],) * 2
    output = project / "data" / "processed" / "patches"
    assert not (output / '.incomplete').exists()
    assert len(list(output.rglob('*.png'))) == 8
    data = (output / 'patches.csv').read_bytes()
    assert b'\r' not in data and b'10.000000' in data
    assert all(p.read_bytes() == original for p, original in inputs.items())


def test_deterministic_manifest_and_png_bytes_across_input_order(tmp_path):
    first, second = make_project(tmp_path / 'first'), make_project(tmp_path / 'second')
    for source in ['train', 'test']:
        path = second / 'data' / 'processed' / f'buildings_{source}.csv'
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    ep.extract_patches(first, 'event-a')
    ep.extract_patches(second, 'event-a')
    out = Path('data/processed/patches')
    for path in (first / out).rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()


def test_saved_pair_keeps_nonstandard_size_and_exact_content(project):
    path = project / 'data/processed/buildings_train.csv'
    table = pd.read_csv(path)
    table.loc[0, ['x_max', 'width']] = [50.25, 40.25]
    table.to_csv(path, index=False)
    manifest = ep.extract_patches(project, 'event-a')
    record = manifest.set_index('image_id').loc['train_b']
    assert record.patch_side == 81  # ceil(2 * 40.25); neither 64 nor a model input size
    assert record[['pad_left', 'pad_top', 'pad_right', 'pad_bottom']].tolist() == [11, 26, 30, 25]
    for phase in ['pre', 'post']:
        with Image.open(project / record[f'{phase}_patch']) as patch:
            assert patch.size == (81, 81)
            pixels = np.asarray(patch)
            np.testing.assert_array_equal(pixels[26:56, 11:51], np.asarray(rgb_pattern(40, 30)))
            assert not pixels[:26].any() and not pixels[56:].any()
            assert not pixels[:, :11].any() and not pixels[:, 51:].any()


@pytest.mark.parametrize('phase', ['pre', 'post'])
def test_missing_source_fails_before_output(project, phase):
    (project / 'data/train/images' / f'train_a_{phase}_disaster.png').unlink()
    with pytest.raises(ValueError, match=r'image_id=train_a uid=b split=train.*Missing source'):
        ep.extract_patches(project, 'event-a')
    assert not (project / 'data/processed/patches').exists()


def test_dimension_mismatch_fails_before_output(project):
    Image.new('RGB', (41, 30)).save(project / 'data/train/images/train_a_post_disaster.png')
    with pytest.raises(ValueError, match='dimensions differ'):
        ep.extract_patches(project, 'event-a')
    assert not (project / 'data/processed/patches').exists()


@pytest.mark.parametrize('relative', ['data/processed/buildings_train.csv',
                                      'data/processed/buildings_test.csv',
                                      'splits/standard/train_images.txt',
                                      'splits/standard/val_images.txt',
                                      'splits/standard/test_images.txt'])
def test_missing_required_inputs(project, relative):
    (project / relative).unlink()
    with pytest.raises(FileNotFoundError):
        ep.extract_patches(project, 'event-a')


@pytest.mark.parametrize('change, message', [('duplicate', 'Duplicate manifest'), ('nan', 'Non-finite'),
                                           ('uid', 'unsafe uid'), ('source', 'source_split'),
                                           ('bool', 'True/False'), ('case', 'collision')])
def test_bad_rows_fail_before_output(project, change, message):
    path = project / 'data/processed/buildings_train.csv'
    table = pd.read_csv(path)
    if change == 'duplicate':
        table = pd.concat([table, table.iloc[:1]])
    elif change == 'nan':
        table.loc[0, 'x_min'] = float('nan')
    elif change == 'uid':
        table.loc[0, 'uid'] = '../escape'
    elif change == 'source':
        table.loc[0, 'source_split'] = 'test'
    elif change == 'bool':
        table['is_bad_polygon'] = 'invalid'
    elif change == 'case':
        extra = table.iloc[[1]].copy()
        extra['uid'] = 'B'
        table = pd.concat([table, extra])
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        ep.extract_patches(project, 'event-a')
    assert not (project / 'data/processed/patches').exists()


def test_overlapping_split_ids_rejected(project):
    (project / 'splits/standard/val_images.txt').write_text('train_a\n')
    with pytest.raises(ValueError, match='overlapping'):
        ep.extract_patches(project, 'event-a')


def test_unexpected_split_rejected(project):
    with pytest.raises(ValueError, match='Unexpected split'):
        ep.source_paths(project, dict(image_id='x', uid='u', split='validation'))


def test_existing_output_is_preserved_and_rerun_refused(project):
    ep.extract_patches(project, 'event-a')
    output = project / 'data/processed/patches'
    snapshot = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError, match='already exists'):
        ep.extract_patches(project, 'event-a')
    assert all(p.read_bytes() == data for p, data in snapshot.items())


def test_partial_run_has_marker_no_manifest_and_cannot_be_reused(project, monkeypatch):
    original = Image.Image.save
    calls = 0

    def interrupted_save(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('disk full')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'save', interrupted_save)
    with pytest.raises(ValueError, match='image_id=train_a uid=b split=train.*disk full'):
        ep.extract_patches(project, 'event-a')
    output = project / 'data/processed/patches'
    assert (output / '.incomplete').is_file() and not (output / 'patches.csv').exists()
    with pytest.raises(FileExistsError):
        ep.extract_patches(project, 'event-a')


def test_explicit_scope_and_safe_output_required(project):
    with pytest.raises(ValueError, match='exactly one'):
        ep.extract_patches(project)
    with pytest.raises(ValueError, match='exactly one'):
        ep.extract_patches(project, 'event-a', all_disasters=True)
    with pytest.raises(ValueError, match='strictly under'):
        ep.extract_patches(project, 'event-a', output_root='data/processed')
    with pytest.raises(ValueError, match='strictly under'):
        ep.extract_patches(project, 'event-a', output_root='../escape')
    with pytest.raises(SystemExit) as exc:
        ep.main([])
    assert exc.value.code == 2


def test_all_disasters_requires_explicit_flag(project):
    manifest = ep.extract_patches(project, all_disasters=True)
    assert len(manifest) == 5 and set(manifest.disaster) == {'event-a', 'event-other'}
