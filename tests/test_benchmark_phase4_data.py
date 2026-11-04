"""Timing boundaries and fail-fast checks, using no real dataset patches."""
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_phase4_data as benchmark


def fake_loader(sample_count=6):
    batch = {
        "pre": torch.zeros(2, 3, 128, 128),
        "post": torch.zeros(2, 3, 128, 128),
        "label": torch.zeros(2, dtype=torch.long),
        "image_id": ["image0", "image1"], "uid": ["0", "1"],
    }

    class Iterator:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return batch

        def _shutdown_workers(self):
            self.closed = True

    iterator = Iterator()

    class Loader:
        dataset = range(sample_count)
        batch_size = 2
        num_workers = 0

        def __iter__(self):
            return iterator

    return Loader(), iterator


def test_timing_excludes_startup_and_warmup(monkeypatch):
    loader, iterator = fake_loader()
    clock = iter([0, 0.2, 0.2, 0.7, 0.7, 10, 11, 20, 23])
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    result = benchmark.measure_loader(loader, split="train", batches=2, warmup=1, seed=42)
    assert result.startup_seconds == pytest.approx(0.7)
    assert result.iterator_seconds == pytest.approx(0.2)
    assert result.first_batch_seconds == pytest.approx(0.5)
    assert result.mean_seconds == result.median_seconds == 2
    assert result.samples == 4
    assert result.samples_per_second == 1
    assert result.estimated_loading_seconds == 6
    assert iterator.closed


def test_reject_short_split_before_iteration():
    loader, _ = fake_loader(sample_count=5)
    with pytest.raises(ValueError, match="insufficient full batches"):
        benchmark.measure_loader(loader, split="val", batches=2, warmup=1, seed=42)


def test_shutdown_on_invalid_batch(monkeypatch):
    loader, iterator = fake_loader()
    monkeypatch.setattr(benchmark, "record_batch", lambda *args: (_ for _ in ()).throw(ValueError("bad batch")))
    with pytest.raises(ValueError, match="bad batch"):
        benchmark.measure_loader(loader, split="train", batches=2, warmup=1, seed=42)
    assert iterator.closed


def test_requested_workers_must_actually_exist():
    loader, iterator = fake_loader()
    loader.num_workers = 2
    with pytest.raises(RuntimeError, match="worker processes are not alive"):
        benchmark.measure_loader(loader, split="train", batches=2, warmup=1, seed=42)
    assert iterator.closed


@pytest.mark.parametrize("arguments", [
    ["--batch-size", "0"], ["--batches", "0"], ["--warmup-batches", "0"],
    ["--workers", "-1"], ["--workers", "2", "2"],
])
def test_invalid_cli(arguments):
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args(arguments)
    assert exc.value.code == 2
