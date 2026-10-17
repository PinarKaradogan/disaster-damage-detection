from dataclasses import replace
import json

import pytest
import torch
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.models import create_model
from src.train import build_parser, main
from src.training.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from src.training.config import TrainingConfig
from src.training.engine import fit, run_epoch
from src.training.experiment import run_experiment
from test_training_data import synthetic_manifest  # synthetic fixture, no xBD dependency


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic_loader():
    generator = torch.Generator().manual_seed(7)
    return DataLoader([
        {"pre": torch.rand(3, 32, 32, generator=generator),
         "post": torch.rand(3, 32, 32, generator=generator), "label": index}
        for index in range(4)
    ], batch_size=2)


def test_train_and_validation_steps():
    model = create_model("simple_cnn")
    loader = synthetic_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.classifier.weight.detach().clone()
    train = run_epoch(model, loader, criterion, torch.device("cpu"), optimizer=optimizer)
    assert not torch.equal(before, model.classifier.weight)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    val = run_epoch(model, loader, criterion, torch.device("cpu"))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not model.training
    assert train["loss"] > 0 and val["loss"] > 0
    assert 0 <= val["macro_f1"] <= 1


def test_checkpoint_roundtrip(tmp_path):
    model = create_model("simple_cnn")
    path = tmp_path / "best.pth"
    save_checkpoint(path, model, epoch=3, best_val_macro_f1=0.6,
                    config=TrainingConfig("simple_cnn").to_dict())
    copy = create_model("simple_cnn")
    result = load_checkpoint(path, copy)
    assert result["epoch"] == 3 and result["best_val_macro_f1"] == 0.6
    assert all(torch.equal(value, copy.state_dict()[key]) for key, value in model.state_dict().items())


def test_early_stopping_max_f1():
    stopping = EarlyStopping(patience=2)
    assert stopping.update(0.4) == (True, False)
    assert stopping.update(0.4) == (False, False)
    assert stopping.update(0.5) == (True, False)
    assert stopping.update(0.49) == (False, False)
    assert stopping.update(0.48) == (False, True)
    with pytest.raises(ValueError):
        stopping.update(float("nan"))


def test_fit_selects_f1_not_loss_and_logs(tmp_path, monkeypatch):
    import src.training.engine as engine
    scores = iter([
        {"loss": 1., "macro_f1": .2, "class_f1": [.2] * 4, "accuracy": .2},
        {"loss": 2., "macro_f1": .6, "class_f1": [.6] * 4, "accuracy": .6},
        {"loss": .8, "macro_f1": .3, "class_f1": [.3] * 4, "accuracy": .3},
        {"loss": .1, "macro_f1": .5, "class_f1": [.5] * 4, "accuracy": .5},
    ])
    def fake_epoch(model, loader, criterion, device, **kwargs):
        if kwargs.get("optimizer") is not None:
            # Real optimizer step so the scheduler is exercised without warnings.
            optimizer = kwargs["optimizer"]
            optimizer.zero_grad()
            sum(parameter.sum() for parameter in model.parameters()).backward()
            optimizer.step()
        return next(scores)
    monkeypatch.setattr(engine, "run_epoch", fake_epoch)
    config = replace(TrainingConfig("simple_cnn"), epochs=8, early_stopping_patience=1)
    model = create_model("simple_cnn")
    metrics, history = fit(model, [], [], torch.nn.CrossEntropyLoss(), config,
                           torch.device("cpu"), output_dir=tmp_path,
                           log_dir=tmp_path / "events", metadata=config.to_dict())
    assert metrics["best_epoch"] == 1 and metrics["best_val_macro_f1"] == .6
    assert metrics["early_stopped"] and len(history) == 2
    checkpoint = load_checkpoint(tmp_path / "best.pth", model)
    assert checkpoint["epoch"] == 1
    assert history[1]["learning_rate"] < history[0]["learning_rate"]
    events = EventAccumulator(str(tmp_path / "events")).Reload()
    assert set(events.Tags()["scalars"]) == {
        "train/loss", "val/loss", "train/macro_f1", "val/macro_f1", "learning_rate",
    }
    assert len(events.Scalars("val/macro_f1")) == 2


@pytest.mark.parametrize("loss,sampler", [("weighted_ce", "none"), ("focal", "none"), ("ce", "weighted")])
def test_cpu_experiment_metadata(synthetic_manifest, tmp_path, loss, sampler):
    root, manifest = synthetic_manifest
    config = replace(TrainingConfig("simple_cnn"), epochs=1, smoke_batch_size=2, amp=False)
    result = run_experiment(config, manifest_path=manifest, experiment_name="cpu_smoke",
                            loss=loss, sampler=sampler, smoke=True, device="cpu",
                            results_root=tmp_path / "results", runs_root=tmp_path / "runs",
                            project_root=root)
    output = tmp_path / "results/cpu_smoke"
    metadata = json.loads((output / "config.json").read_text())
    stored = json.loads((output / "metrics.json").read_text())
    assert stored == result and result["smoke"] and result["evaluation_split"] == "val"
    assert len(result["class_f1"]) == 4 and result["training_time_seconds"] > 0
    assert metadata["loss"] == loss and metadata["sampler"] == sampler
    assert metadata["batch_size"] == 2 and metadata["parameter_count"] > 0
    assert metadata["train_class_counts"] == [1, 1, 1, 1]
    assert (output / "best.pth").is_file() and (output / "history.csv").is_file()
    assert list((tmp_path / "runs/cpu_smoke").glob("events.*"))
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        run_experiment(config, manifest_path=manifest, experiment_name="cpu_smoke",
                       results_root=tmp_path / "results", runs_root=tmp_path / "runs")


def test_cli_parser_and_invalid_treatments():
    args = build_parser().parse_args(["--model", "simple_cnn", "--experiment-name", "smoke", "--smoke"])
    assert args.smoke and args.num_workers == 0
    for extra in [["--split", "test"], ["--batch-size", "0"], ["--epochs", "-1"], ["--num-workers", "-1"]]:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--model", "simple_cnn", "--experiment-name", "bad", *extra])
    with pytest.raises(SystemExit):
        main(["--model", "simple_cnn", "--experiment-name", "bad", "--sampler", "weighted"])
