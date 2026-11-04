"""CPU checks for bounded epochs, Windows worker reuse, and projection arithmetic."""
from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import benchmark_phase4_training as benchmark
from src.training.config import TrainingConfig
from src.training.engine import run_epoch
from src.training.losses import make_criterion


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


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


def test_legacy_defaults_and_head_only_cli():
    defaults = benchmark.parse_args([])
    assert (defaults.model, defaults.training_mode, defaults.loss, defaults.sampler) == (
        "simple_cnn", "full", "weighted_ce", "none")
    assert defaults.batch_size == 32 and defaults.workers == [0, 2, 4]
    args = benchmark.parse_args([
        "--model", "resnet34_early_fusion", "--training-mode", "head_only", "--loss", "focal",
        "--sampler", "none", "--manifest", "data/processed/patches_subset25/patches.csv",
        "--batch-size", "16", "--workers", "0", "--persistent", "false",
        "--warmup-batches", "1", "--batches", "2", "--val-warmup-batches", "1", "--val-batches", "2",
    ])
    assert args.model == "resnet34_early_fusion" and args.training_mode == "head_only"
    assert args.loss == "focal" and args.sampler == "none"
    assert args.batch_size == 16 and args.workers == [0] and args.persistent == ["false"]
    assert args.warmup_batches + args.batches == args.val_warmup_batches + args.val_batches == 3
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--training-mode", "head_only"])


def test_pretrained_factory_and_head_mode_without_download(monkeypatch):
    calls = []
    factory = benchmark.create_model

    def offline_factory(name, **kwargs):
        calls.append((name, kwargs.copy()))
        return factory(name, **{**kwargs, "pretrained": False})

    monkeypatch.setattr(benchmark, "create_model", offline_factory)
    args = benchmark.parse_args(["--model", "resnet34_early_fusion", "--training-mode", "head_only"])
    model = benchmark.make_benchmark_model(args)
    assert calls == [(args.model, {"training_mode": "head_only", "pretrained": True})]
    model.train()
    assert model.backbone.fc.training and not model.backbone.bn1.training
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.startswith("backbone.fc."))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    digest = benchmark.state_digest(original)
    restored = benchmark.make_benchmark_model(args, pretrained=False)
    restored.load_state_dict(original)
    assert benchmark.state_digest(restored.state_dict()) == digest
    assert calls[-1][1]["pretrained"] is False


def test_pretrained_failure_has_no_random_fallback(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("weights unavailable")

    monkeypatch.setattr(benchmark, "create_model", fail)
    args = benchmark.parse_args(["--model", "resnet34_early_fusion", "--training-mode", "head_only"])
    with pytest.raises(RuntimeError, match="Pretrained ResNet34 initialization failed"):
        benchmark.make_benchmark_model(args)
    assert len(calls) == 1 and calls[0]["pretrained"] is True


def test_optimizer_only_receives_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 4))
    model[0].requires_grad_(False)
    optimizer = benchmark.make_benchmark_optimizer(model, TrainingConfig("resnet34_early_fusion", training_mode="head_only"))
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in model[1].parameters()]
    before = model[0].weight.detach().clone()
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    assert torch.equal(before, model[0].weight)
    assert model[0].weight.grad is None


@pytest.mark.parametrize("loss", ["weighted_ce", "ce", "focal"])
@pytest.mark.parametrize("training", [False, True])
def test_step_loss_mass_matches_engine_on_cpu(monkeypatch, loss, training):
    """Exercise real loss/backward on CPU; CUDA timing/transfer calls are test doubles."""
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

    original_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())

    class PairedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, 4)

        def forward(self, pre, post):
            return self.head(torch.cat([pre, post], dim=1))

    generator = torch.Generator().manual_seed(7)
    batch = dict(pre=torch.rand(3, 2, generator=generator), post=torch.rand(3, 2, generator=generator),
                 label=torch.tensor([0, 0, 3]))
    model, reference = PairedLinear(), PairedLinear()
    reference.load_state_dict(model.state_dict())
    criterion = make_criterion(loss, "none", torch.tensor([1., 2., 3., 4.]))
    optimizer = benchmark.make_benchmark_optimizer(model, TrainingConfig("simple_cnn"))
    reference_optimizer = benchmark.make_benchmark_optimizer(reference, TrainingConfig("simple_cnn"))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model.train(training)
    _, loss_mass, mass, targets, predictions = benchmark.training_step(
        model, batch, criterion, optimizer, scaler, training=training)
    metrics = run_epoch(reference, [batch], criterion, torch.device("cpu"),
                        optimizer=reference_optimizer if training else None)
    assert mass == (6. if loss == "weighted_ce" else 3.)
    assert loss_mass / mass == pytest.approx(metrics["loss"])
    assert targets == [0, 0, 3] and len(predictions) == 3
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference.state_dict()[key])
    assert (model.head.weight.grad is not None) == training
