"""CPU freezing checks and lightweight orchestration/resume checks; no downloads."""
import json
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import create_model
from src.models.resnet_early_fusion import ResNet34EarlyFusion
from src.train import build_parser, main
from src.training.checkpoint import atomic_save
from src.training.config import TrainingConfig
from src.training.engine import run_epoch
from src.training.experiment import run_experiment
from test_training_data import synthetic_manifest
from test_training_resume import equal_nested


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", ["full", "head_only"])
def test_real_resnet_freezing_and_batchnorm(mode):
    torch.manual_seed(5)
    model = create_model("resnet34_early_fusion", pretrained=False, training_mode=mode)
    # Same initialization: freezing must not change six-channel weights or BN buffers.
    torch.manual_seed(5)
    reference = create_model("resnet34_early_fusion", pretrained=False)
    equal_nested(model.state_dict(), reference.state_dict())
    del reference
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loader = DataLoader([dict(pre=torch.randn(3, 32, 32), post=torch.randn(3, 32, 32), label=i)
                         for i in range(2)], batch_size=2)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=.1)
    model.eval()
    run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer=optimizer)
    assert model.training and model.backbone.fc.training
    assert not torch.equal(before["backbone.fc.weight"], model.backbone.fc.weight)
    assert sum(p.numel() for p in model.parameters()) == 21296132
    if mode == "head_only":
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2052
        for name, value in model.state_dict().items():
            if not name.startswith("backbone.fc."):
                assert torch.equal(value, before[name]), name
        for name, parameter in model.named_parameters():
            assert parameter.requires_grad == name.startswith("backbone.fc.")
            if not parameter.requires_grad:
                assert parameter.grad is None
        assert all(not layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm2d))
    else:
        assert all(p.requires_grad for p in model.parameters())
        assert not torch.equal(before["backbone.conv1.weight"], model.backbone.conv1.weight)
        assert not torch.equal(before["backbone.bn1.running_mean"], model.backbone.bn1.running_mean)
        assert all(layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm2d))
    model.eval()
    assert not model.backbone.fc.training


class TinyResNet(nn.Module):
    """Same classifier/BN contract as timm; keeps repeated checkpoint tests small."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)
        self.pretrained_cfg = {"test_fixture": True}

    def get_classifier(self):
        return self.fc

    def forward(self, x):
        return self.fc(self.pool(self.bn1(self.conv1(x))).flatten(1))


@pytest.fixture
def tiny_backbone(monkeypatch):
    import src.models.resnet_early_fusion as resnet
    created = []

    def factory(name, *, pretrained, in_chans, num_classes):
        assert name == "resnet34" and in_chans == 6 and num_classes == 4
        model = TinyResNet()
        created.append((pretrained, model))
        return model

    monkeypatch.setattr(resnet.timm, "create_model", factory)
    return created


def run(root, manifest, tmp_path, name, *, mode="head_only", **kwargs):
    config = TrainingConfig("resnet34_early_fusion", training_mode=mode, batch_size=2,
                            input_size=16, epochs=3, amp=False, early_stopping_patience=10)
    return run_experiment(config, manifest_path=manifest, experiment_name=name,
                          results_root=tmp_path / "results", runs_root=tmp_path / "runs",
                          project_root=root, device="cpu", loss="ce", sampler="weighted", **kwargs)


def test_head_only_resume_continuity(synthetic_manifest, tmp_path, tiny_backbone):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "full_session")
    run(root, manifest, tmp_path, "split", session_epochs=1)
    output = tmp_path / "results/split"
    first = torch.load(output / "last.pth", weights_only=True)
    assert len(first["optimizer_state_dict"]["param_groups"][0]["params"]) == 2
    run(root, manifest, tmp_path, "split", resume=True)
    uninterrupted = torch.load(tmp_path / "results/full_session/last.pth", weights_only=True)
    resumed = torch.load(output / "last.pth", weights_only=True)
    for key in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                "history", "rng", "early_stopping", "best_metrics"]:
        equal_nested(uninterrupted[key], resumed[key])
    for name, value in resumed["model_state_dict"].items():
        if not name.startswith("backbone.fc."):
            assert torch.equal(first["model_state_dict"][name], value)
    assert [pretrained for pretrained, _ in tiny_backbone] == [True, True, False]
    assert all(not model.bn1.training and model.bn1.weight.grad is None for _, model in tiny_backbone)
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["training_mode"] == "head_only"
    assert metadata["trainable_parameter_count"] == 36
    assert metadata["parameter_count"] > metadata["trainable_parameter_count"]


@pytest.mark.parametrize("original,requested", [("head_only", "full"), ("full", "head_only")])
def test_resume_rejects_mode_change(synthetic_manifest, tmp_path, tiny_backbone, original, requested):
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "split", mode=original, session_epochs=1)
    output = tmp_path / "results/split"
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="configuration mismatch.*training_mode"):
        run(root, manifest, tmp_path, "split", mode=requested, resume=True)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_pre_mode_complete_resume_checkpoint_is_full(synthetic_manifest, tmp_path):
    from test_training_resume import run as run_cnn
    root, manifest = synthetic_manifest
    run_cnn(root, manifest, tmp_path, "legacy_full", session_epochs=1)
    output = tmp_path / "results/legacy_full"
    config = json.loads((output / "config.json").read_text())
    for key in ["training_mode", "trainable_parameter_count"]:
        del config[key]
    (output / "config.json").write_text(json.dumps(config))
    state = torch.load(output / "last.pth", weights_only=True)
    state["config"] = config
    state["best_checkpoint"]["config"] = config
    atomic_save(output / "last.pth", state)
    original_config_bytes = (output / "config.json").read_bytes()
    result = run_cnn(root, manifest, tmp_path, "legacy_full", resume=True, session_epochs=1)
    assert result["epochs_completed"] == 2
    assert (output / "config.json").read_bytes() == original_config_bytes
    result = run_cnn(root, manifest, tmp_path, "legacy_full", resume=True)
    assert result["epochs_completed"] == 4


def test_cli_and_simple_cnn_default():
    model = create_model("simple_cnn", pretrained=False)
    assert all(parameter.requires_grad for parameter in model.parameters())
    args = build_parser().parse_args(["--model", "resnet34_early_fusion", "--experiment-name", "new",
                                     "--training-mode", "head_only"])
    assert args.training_mode == "head_only"
    assert TrainingConfig("simple_cnn").training_mode == "full"
    with pytest.raises(ValueError, match="requires resnet"):
        create_model("simple_cnn", training_mode="head_only")
    with pytest.raises(ValueError, match="requires resnet"):
        replace(TrainingConfig("simple_cnn"), training_mode="head_only")
    with pytest.raises(ValueError, match="full or head_only"):
        ResNet34EarlyFusion(pretrained=False, training_mode="invalid")
    with pytest.raises(SystemExit):
        main(["--model", "simple_cnn", "--experiment-name", "bad", "--training-mode", "head_only"])
