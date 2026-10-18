"""CPU checks for bounded epochs, Windows worker reuse, and projection arithmetic."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import benchmark_phase4_training as benchmark


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
def test_bounded_exhaustion_and_sample_order_across_epochs(workers, persistent):
    source = DataLoader(TensorDataset(torch.arange(12)), batch_size=2, shuffle=True,
                        num_workers=workers, generator=torch.Generator().manual_seed(42))
    loader = benchmark.bounded_loader(source, batches=3, persistent=persistent, seed=42)
    pids = None
    try:
        assert len(loader) == 3
        for seed in (42, 43):
            source.generator.manual_seed(seed)
            iterator = iter(loader)
            current = [p.pid for p in getattr(iterator, "_workers", ())]
            if persistent and pids is not None:
                assert current == pids
            pids = current
            actual = torch.cat([batch[0] for batch in iterator])
            expected = torch.randperm(12, generator=torch.Generator().manual_seed(seed))[:6]
            assert torch.equal(actual, expected)
            with pytest.raises(StopIteration):
                next(iterator)
            if workers:
                assert all(p.is_alive() == persistent for p in iterator._workers)
    finally:
        benchmark.shutdown_iterator(getattr(loader, "_iterator", None))


def test_rejects_oversized_or_empty_window():
    source = DataLoader(TensorDataset(torch.arange(5)), batch_size=2)
    for batches in (0, 3):
        with pytest.raises(ValueError, match="Insufficient full batches"):
            benchmark.LimitedBatchSampler(source.batch_sampler, batches)


def test_no_persistent_workers_without_processes():
    source = DataLoader(TensorDataset(torch.arange(8)), batch_size=2)
    with pytest.raises(ValueError, match="requires num_workers"):
        benchmark.bounded_loader(source, batches=2, persistent=True, seed=42)


def test_epoch_projection_uses_e2e_and_avoids_double_counting_first_wait():
    result = SimpleNamespace(e2e_s=20, batches=10, startup_s=5, wait_s=10, end_s=0.5)
    assert benchmark.estimate_split_seconds(result, 65, 32) == pytest.approx(10.5)


def test_initial_weight_hash_includes_buffers_and_restores_exactly():
    model = benchmark.SimpleCNN()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    digest = benchmark.state_digest(original)
    model.features[1].running_mean.add_(1)
    assert benchmark.state_digest(model.state_dict()) != digest
    model.load_state_dict(original)
    assert benchmark.state_digest(model.state_dict()) == digest


def test_no_cuda_fails_without_building_data(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        benchmark.main([])


@pytest.mark.parametrize("args", [
    ["--workers", "-1"], ["--workers", "2", "2"], ["--batches", "0"],
    ["--val-batches", "0"], ["--persistent", "true", "true"],
    ["--workers", "0", "--persistent", "true"],
    ["--json-output", str(benchmark.ROOT / "results/phase4_cnn_weighted_ce_seed42/new.json")],
])
def test_invalid_or_protected_arguments(args):
    with pytest.raises(SystemExit) as exc:
        benchmark.parse_args(args)
    assert exc.value.code == 2


def test_output_refuses_overwrite(tmp_path):
    target = tmp_path / "existing.json"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--json-output", str(target)])
    assert target.read_text(encoding="utf-8") == "keep"
