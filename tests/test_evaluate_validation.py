"""Validation-only legacy checkpoint evaluation on tiny CPU images."""
import json
import math

import pytest
import torch

from src import evaluate_validation as evaluation
from src.models import create_model
from src.training.artifacts import file_sha256, write_json
from src.training.checkpoint import atomic_save, save_checkpoint
from src.training.config import TrainingConfig
from src.training.data import IMAGENET_MEAN, IMAGENET_STD
from src.data.constants import LABEL_MAP
from test_training_data import synthetic_manifest


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def interrupted(synthetic_manifest, tmp_path):
    root, manifest = synthetic_manifest
    directory = tmp_path / "source_experiment"
    directory.mkdir()
    metadata = dict(
        **TrainingConfig("simple_cnn", batch_size=2, input_size=16, amp=False).to_dict(),
        evaluation_split="val", smoke=False, label_map=LABEL_MAP,
        experiment_name="interrupted", loss="ce", sampler="weighted", class_weights=[1.] * 4,
        normalization={"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        augmentation={"val": None}, manifest_path=str(manifest),
        manifest_sha256=file_sha256(manifest), split_counts={"val": 4},
        device="cpu", versions={"torch": str(torch.__version__)},
    )
    model = create_model("simple_cnn")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_checkpoint(directory / "best.pth", model, epoch=12, best_val_macro_f1=.7, config=metadata)
    write_json(directory / "config.json", metadata)
    write_json(directory / "status.json", {"status": "running"})
    (directory / "history.csv").write_text("epoch\n12\n13\n14\n")
    # No non-validation images are available; accidentally reading them must fail.
    for split in ["train", "test"]:
        for index in range(4):
            (root / f"{split}_{index}.png").unlink()
    return root, directory, tmp_path / "reports/validation.json"


def test_validation_report_and_source_unchanged(interrupted, monkeypatch):
    root, directory, output = interrupted
    original = {p.name: p.read_bytes() for p in directory.iterdir()}
    factory, torch_load = evaluation.create_model, torch.load
    seen = []

    def checked_factory(*args, **kwargs):
        assert kwargs["pretrained"] is False
        model = factory(*args, **kwargs)
        forward = model.forward

        def checked_forward(pre, post):
            assert not model.training and not torch.is_grad_enabled()
            assert pre.shape[0] == 2 and pre.shape[2:] == (16, 16)
            if not seen:
                expected = (torch.tensor([0., 70., 120.]) / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
                assert torch.allclose(pre[0, :, 0, 0], expected)
            seen.append(True)
            return forward(pre, post)

        model.forward = checked_forward
        return model

    def checked_load(*args, **kwargs):
        assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"
        return torch_load(*args, **kwargs)

    def forbid_training(*args, **kwargs):
        pytest.fail("Optimizer construction is forbidden")

    monkeypatch.setattr(evaluation, "create_model", checked_factory)
    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(torch.optim, "AdamW", forbid_training)
    report = evaluation.evaluate_validation(directory, output=output, project_root=root)
    assert len(seen) == 2
    assert json.loads(output.read_text()) == report
    assert report["interrupted_experiment"] and report["source_status"] == "running"
    assert report["checkpoint_epoch"] == 12 and report["last_history_epoch"] == 14
    assert report["evaluation_split"] == "val" and not report["test_used"]
    assert not report["training_performed"] and not report["amp"]
    metrics = report["validation"]
    assert metrics["macro_f1"] == pytest.approx(.1)
    assert metrics["accuracy"] == .25 and metrics["loss"] == pytest.approx(math.log(4))
    assert metrics["class_precision"] == [.25, 0., 0., 0.]
    assert metrics["class_recall"] == [1., 0., 0., 0.]
    assert metrics["class_f1"] == pytest.approx([.4, 0., 0., 0.])
    assert metrics["confusion_matrix"] == [[1, 0, 0, 0]] * 4
    assert report["macro_f1_comparison"]["signed_difference"] == pytest.approx(-.6)
    assert not report["macro_f1_comparison"]["matches_within_tolerance"]
    assert original == {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        evaluation.evaluate_validation(directory, output=output, project_root=root)


@pytest.mark.parametrize("change,message", [
    ("sha", "SHA256"), ("checkpoint_config", "configuration differs"),
    ("architecture", "architecture mismatch"), ("unsafe", "safely load"),
    ("normalization", "preprocessing"), ("test", "full validation"),
    ("completed", "unfinished"), ("cuda", "requires CUDA"),
    ("count", "sample count"),
])
def test_invalid_inputs_produce_no_report(interrupted, monkeypatch, change, message):
    root, directory, output = interrupted
    metadata = json.loads((directory / "config.json").read_text())
    if change in {"checkpoint_config", "architecture", "unsafe"}:
        path = directory / "best.pth"
        checkpoint = torch.load(path, weights_only=True)
        if change == "checkpoint_config":
            checkpoint["config"]["batch_size"] = 99
        elif change == "architecture":
            checkpoint["model_state_dict"]["classifier.weight"] = torch.zeros(3, 128)
        else:
            path.write_bytes(b"invalid checkpoint")
        if change != "unsafe":
            atomic_save(path, checkpoint)
    elif change == "completed":
        write_json(directory / "status.json", {"status": "completed"})
    else:
        if change == "sha":
            metadata["manifest_sha256"] = "0" * 64
        elif change == "normalization":
            metadata["normalization"]["mean"][0] = 0.
        elif change == "test":
            metadata["evaluation_split"] = "test"
        elif change == "cuda":
            metadata.update(device="cuda", amp=True)
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        elif change == "count":
            metadata["split_counts"]["val"] = 5
            path = directory / "best.pth"
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["config"] = metadata
            atomic_save(path, checkpoint)
        write_json(directory / "config.json", metadata)
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate_validation(directory, output=output, project_root=root)
    assert not output.exists()


@pytest.mark.parametrize("destination", ["source", "metrics", "existing"])
def test_report_output_protection(interrupted, destination):
    root, directory, output = interrupted
    if destination == "source":
        output = directory / "new_report.json"
    elif destination == "metrics":
        output = output.parent / "metrics.json"
    else:
        output.parent.mkdir()
        output.write_text("keep this")
    with pytest.raises((ValueError, FileExistsError)):
        evaluation.evaluate_validation(directory, output=output, project_root=root)
    if destination == "existing":
        assert output.read_text() == "keep this"


def test_cli(interrupted, monkeypatch, capsys):
    root, directory, output = interrupted
    original = evaluation.evaluate_validation
    monkeypatch.setattr(evaluation, "evaluate_validation",
                        lambda directory, **kwargs: original(directory, project_root=root, **kwargs))
    evaluation.main(["--experiment-dir", str(directory), "--output", str(output)])
    captured = capsys.readouterr()
    assert json.loads(captured.out)["checkpoint_epoch"] == 12
    assert "[val] epoch_start completed_batches=0" in captured.err
    assert "[val] first_batch_received completed_batches=0" in captured.err


def test_matching_checkpoint_score(interrupted):
    root, directory, output = interrupted
    path = directory / "best.pth"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["best_val_macro_f1"] = .1
    atomic_save(path, checkpoint)
    report = evaluation.evaluate_validation(directory, output=output, project_root=root)
    assert report["macro_f1_comparison"]["matches_within_tolerance"]
    assert report["macro_f1_comparison"]["absolute_difference"] == pytest.approx(0.)
