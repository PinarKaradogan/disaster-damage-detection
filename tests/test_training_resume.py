"""Epoch-boundary resume tests, entirely synthetic and CPU-only."""
import csv
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.training.checkpoint import atomic_save
from src.training.config import TrainingConfig
from src.training.experiment import run_experiment
from src.training.engine import _cpu_snapshot
from src.train import build_parser
from test_training_data import synthetic_manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def run(root, manifest, tmp_path, name, **kwargs):
    config = kwargs.pop("config", replace(TrainingConfig("simple_cnn"), epochs=4,
                                         batch_size=2, input_size=16, amp=False,
                                         early_stopping_patience=10))
    return run_experiment(config, manifest_path=manifest, experiment_name=name,
                          results_root=tmp_path / "results", runs_root=tmp_path / "runs",
                          project_root=root, device="cpu", **kwargs)


def checkpoint(tmp_path, name):
    return torch.load(tmp_path / "results" / name / "last.pth", weights_only=True)


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


def test_cpu_snapshot_independent_of_optimizer_updates():
    model = torch.nn.Linear(2, 1)
    model.register_buffer("counter", torch.tensor(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    def step():
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 2)).square().sum().backward()
        optimizer.step()

    step()  # Populate AdamW moments and step tensors.
    live = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                config={"nested": ([1, 2], {"tensor": torch.ones(1, requires_grad=True)})})
    rng_before = torch.get_rng_state().clone()
    snapshot = _cpu_snapshot(live)
    frozen = _cpu_snapshot(snapshot)
    equal_nested(snapshot, live)  # Snapshot creation does not modify active state.
    assert torch.equal(rng_before, torch.get_rng_state())
    assert type(snapshot["model_state_dict"]) is type(live["model_state_dict"])
    assert snapshot["model_state_dict"]._metadata == live["model_state_dict"]._metadata

    def check_storage(source, saved):
        if isinstance(source, torch.Tensor):
            assert saved.device.type == "cpu" and not saved.requires_grad
            assert saved.data_ptr() != source.data_ptr()
        elif isinstance(source, dict):
            for key in source:
                check_storage(source[key], saved[key])
        elif isinstance(source, (list, tuple)):
            assert type(source) is type(saved)
            for x, y in zip(source, saved):
                check_storage(x, y)

    check_storage(live, snapshot)
    step()
    model.counter.add_(1)
    live["config"]["nested"][0].append(3)
    equal_nested(snapshot, frozen)
    assert not torch.equal(model.weight, snapshot["model_state_dict"]["weight"])
    state = next(iter(optimizer.state.values()))
    saved_state = next(iter(snapshot["optimizer_state_dict"]["state"].values()))
    assert state["step"].item() == saved_state["step"].item() + 1
    assert not torch.equal(state["exp_avg"], saved_state["exp_avg"])


@pytest.mark.parametrize("sampler,loss", [("none", "weighted_ce"), ("weighted", "ce")])
def test_two_sessions_equal_uninterrupted(synthetic_manifest, tmp_path, sampler, loss):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "full", sampler=sampler, loss=loss)
    paused = run(root, manifest, tmp_path, "split", sampler=sampler, loss=loss, session_epochs=2)
    assert paused["status"] == "paused" and paused["epochs_completed"] == 2
    first = checkpoint(tmp_path, "split")
    assert first["epoch"] == 2 and first["scheduler_state_dict"]["T_max"] == 4
    # Perturb every global RNG before restarting; construction must not affect restoration.
    random.random(), np.random.rand(), torch.rand(3)
    result = run(root, manifest, tmp_path, "split", sampler=sampler, loss=loss,
                 resume=True, session_epochs=2)
    assert result["status"] == "completed" and result["epochs_completed"] == 4
    full, split = checkpoint(tmp_path, "full"), checkpoint(tmp_path, "split")
    for key in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "early_stopping", "rng", "history", "best_metrics"]:
        equal_nested(full[key], split[key])
    output = tmp_path / "results/split"
    with (output / "history.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [1, 2, 3, 4]
    events = EventAccumulator(str(tmp_path / "runs/split")).Reload()
    assert [event.step for event in events.Scalars("val/macro_f1")] == [1, 2, 3, 4]
    best = torch.load(output / "best.pth", weights_only=True)
    equal_nested(best, split["best_checkpoint"])
    assert json.loads((output / "status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("change", ["seed", "epochs", "batch_size", "learning_rate", "loss", "sampler", "manifest"])
def test_incompatible_resume_preserves_artifacts(synthetic_manifest, tmp_path, change):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "split", session_epochs=1)
    output = tmp_path / "results/split"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    kwargs = {}
    if change in {"loss", "sampler"}:
        kwargs = {"loss": "ce", "sampler": "weighted" if change == "sampler" else "none"}
    elif change == "manifest":
        manifest.write_text(manifest.read_text() + "\n")
    else:
        base = replace(TrainingConfig("simple_cnn"), epochs=4, batch_size=2,
                       input_size=16, amp=False, early_stopping_patience=10)
        kwargs["config"] = replace(base, **{change: getattr(base, change) + 1})
    with pytest.raises(ValueError, match="configuration mismatch"):
        run(root, manifest, tmp_path, "split", resume=True, **kwargs)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_reconcile_interrupted_publication(synthetic_manifest, tmp_path):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "split", session_epochs=2)
    output = tmp_path / "results/split"
    # Simulate interrupted writes or an uncommitted future row.
    (output / "history.csv").write_text("epoch\n1\n2\n3\n3\n")
    (output / "best.pth").write_bytes(b"interrupted")
    (output / "status.json").write_text('{"status": "failed"}')
    with pytest.warns(UserWarning, match="rebuilding"):
        run(root, manifest, tmp_path, "split", resume=True, session_epochs=1)
    state = checkpoint(tmp_path, "split")
    with (output / "history.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [1, 2, 3]
    equal_nested(torch.load(output / "best.pth", weights_only=True), state["best_checkpoint"])


def test_early_stopping_continues(synthetic_manifest, tmp_path, monkeypatch):
    import src.training.engine as engine
    original = engine.run_epoch

    def constant_score(*args, **kwargs):
        result = original(*args, **kwargs)
        result["macro_f1"] = 0.25
        return result

    monkeypatch.setattr(engine, "run_epoch", constant_score)
    root, manifest = synthetic_manifest
    config = replace(TrainingConfig("simple_cnn"), epochs=8, input_size=16,
                     batch_size=2, amp=False, early_stopping_patience=2)
    run(root, manifest, tmp_path, "split", config=config, session_epochs=2)
    assert checkpoint(tmp_path, "split")["early_stopping"] == {"best": .25, "bad_epochs": 1}
    result = run(root, manifest, tmp_path, "split", config=config, resume=True, session_epochs=2)
    assert result["early_stopped"] and result["epochs_completed"] == 3
    before = (tmp_path / "results/split/last.pth").read_bytes()
    with pytest.raises(ValueError, match="completed"):
        run(root, manifest, tmp_path, "split", config=config, resume=True)
    assert (tmp_path / "results/split/last.pth").read_bytes() == before


def test_overwrite_and_legacy_refused(synthetic_manifest, tmp_path):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "split", session_epochs=1)
    output = tmp_path / "results/split"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        run(root, manifest, tmp_path, "split")
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    atomic_save(output / "last.pth", {"model_state_dict": {}})
    with pytest.raises(ValueError, match="legacy/invalid"):
        run(root, manifest, tmp_path, "split", resume=True)


def test_atomic_save_failure_keeps_previous(tmp_path, monkeypatch):
    import src.training.artifacts as artifacts
    path = tmp_path / "last.pth"
    atomic_save(path, {"epoch": 1})
    before = path.read_bytes()

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError):
        atomic_save(path, {"epoch": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_resume_cli_arguments():
    args = build_parser().parse_args(["--model", "simple_cnn", "--experiment-name", "new",
                                     "--epochs", "30", "--session-epochs", "2", "--resume"])
    assert args.resume and args.epochs == 30 and args.session_epochs == 2
    for value in ["0", "-2"]:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--model", "simple_cnn", "--experiment-name", "new",
                                       "--session-epochs", value])


def test_failed_session_retains_committed_epoch(synthetic_manifest, tmp_path, monkeypatch):
    import src.training.engine as engine
    root, manifest = synthetic_manifest
    original = engine.run_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption during epoch 2")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(root, manifest, tmp_path, "split")
    assert checkpoint(tmp_path, "split")["epoch"] == 1
    assert json.loads((tmp_path / "results/split/status.json").read_text())["status"] == "failed"
    monkeypatch.setattr(engine, "run_epoch", original)
    result = run(root, manifest, tmp_path, "split", resume=True, session_epochs=1)
    assert result["epochs_completed"] == 2 and result["status"] == "paused"
