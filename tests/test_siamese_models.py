"""Phase 5 Siamese classifier checks on CPU; no downloads."""
import json

import pytest
import timm
import torch
from torch import nn

from src.models import MODEL_NAMES, PRETRAINED_MODELS, create_model
from src.models.siamese import SiameseClassifier
from src.train import build_parser
from src.training.config import TrainingConfig
from src.training.experiment import run_experiment
from test_training_data import synthetic_manifest  # noqa: F401  (pytest fixture)
from test_training_resume import equal_nested


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def resnet_siamese():
    torch.manual_seed(0)
    return create_model("siamese_resnet34", pretrained=False)


def test_registered_and_cli_choice():
    assert "siamese_resnet34" in MODEL_NAMES and "siamese_resnet34" in PRETRAINED_MODELS
    args = build_parser().parse_args(["--model", "siamese_resnet34", "--experiment-name", "x", "--loss", "focal"])
    assert args.model == "siamese_resnet34"


def test_forward_shape_and_finite(resnet_siamese):
    resnet_siamese.eval()
    logits = resnet_siamese(torch.randn(3, 3, 64, 64), torch.randn(3, 3, 64, 64))
    assert logits.shape == (3, 4) and torch.isfinite(logits).all()


def test_single_shared_encoder(resnet_siamese):
    encoders = [m for m in resnet_siamese.modules() if isinstance(m, timm.models.resnet.ResNet)]
    assert len(encoders) == 1 and encoders[0] is resnet_siamese.backbone
    expected = (sum(p.numel() for p in resnet_siamese.backbone.parameters())
                + sum(p.numel() for p in resnet_siamese.head.parameters()))
    assert sum(p.numel() for p in resnet_siamese.parameters()) == expected
    # Identical images must give identical PRE/POST features (same weights, eval BN).
    resnet_siamese.eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        f_pre, f_post = resnet_siamese.encode_pair(image, image.clone())
    assert torch.equal(f_pre, f_post)


def test_head_input_is_pre_post_absdiff(resnet_siamese):
    resnet_siamese.eval()
    pre, post = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        f_pre, f_post = resnet_siamese.encode_pair(pre, post)
        expected = resnet_siamese.head(torch.cat([f_pre, f_post, (f_pre - f_post).abs()], 1))
        assert torch.allclose(resnet_siamese(pre, post), expected)
    first = resnet_siamese.head[0]
    assert first.in_features == 3 * resnet_siamese.backbone.num_features == 3 * 512


def test_backward_reaches_shared_encoder_from_both_inputs(resnet_siamese):
    resnet_siamese.train()
    resnet_siamese.zero_grad()
    pre = torch.randn(2, 3, 64, 64, requires_grad=True)
    post = torch.randn(2, 3, 64, 64, requires_grad=True)
    nn.functional.cross_entropy(resnet_siamese(pre, post), torch.tensor([0, 3])).backward()
    assert pre.grad is not None and post.grad is not None
    assert pre.grad.abs().sum() > 0 and post.grad.abs().sum() > 0
    conv1 = resnet_siamese.backbone.conv1.weight.grad
    assert conv1 is not None and torch.isfinite(conv1).all() and conv1.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in resnet_siamese.head.parameters())


def test_mismatched_inputs_rejected(resnet_siamese):
    with pytest.raises(ValueError, match="matching"):
        resnet_siamese(torch.zeros(2, 3, 64, 64), torch.zeros(2, 3, 32, 32))
    with pytest.raises(ValueError, match="matching"):
        resnet_siamese(torch.zeros(2, 6, 64, 64), torch.zeros(2, 6, 64, 64))


def test_head_only_rejected_for_siamese():
    with pytest.raises(ValueError, match="head_only"):
        create_model("siamese_resnet34", pretrained=False, training_mode="head_only")
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("siamese_unknown", pretrained=False)


class TinyEncoder(nn.Module):
    """timm-like encoder contract: num_features, pretrained_cfg, pooled features."""

    num_features = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_cfg = {"hf_hub_id": "test/fixture"}

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x)))).flatten(1)


@pytest.fixture
def tiny_encoder(monkeypatch):
    import src.models.siamese as siamese
    created = []

    def factory(name, *, pretrained, num_classes, **kwargs):
        assert name == "resnet34" and num_classes == 0 and not kwargs
        created.append(pretrained)
        return TinyEncoder()

    monkeypatch.setattr(siamese.timm, "create_model", factory)
    return created


def run(root, manifest, tmp_path, name, **kwargs):
    config = TrainingConfig("siamese_resnet34", batch_size=2, input_size=16, epochs=3, amp=False,
                            early_stopping_patience=10)
    return run_experiment(config, manifest_path=manifest, experiment_name=name,
                          results_root=tmp_path / "results", runs_root=tmp_path / "runs",
                          project_root=root, device="cpu", loss="focal", **kwargs)


def test_experiment_metadata_checkpoints_and_resume(synthetic_manifest, tmp_path, tiny_encoder):  # noqa: F811
    root, manifest = synthetic_manifest
    run(root, manifest, tmp_path, "full_session")
    metadata = json.loads((tmp_path / "results/full_session/config.json").read_text())
    assert metadata["model_name"] == "siamese_resnet34" and metadata["loss"] == "focal"
    assert metadata["pretrained"] is True
    assert metadata["pretrained_source"] == {"hf_hub_id": "test/fixture"}
    assert metadata["pretrained_adaptation"].startswith("none: shared")
    for file in ("best.pth", "last.pth", "history.csv", "metrics.json", "status.json"):
        assert (tmp_path / "results/full_session" / file).is_file()

    run(root, manifest, tmp_path, "split", session_epochs=1)
    run(root, manifest, tmp_path, "split", resume=True)
    uninterrupted = torch.load(tmp_path / "results/full_session/last.pth", weights_only=True)
    resumed = torch.load(tmp_path / "results/split/last.pth", weights_only=True)
    for key in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "history",
                "early_stopping", "best_metrics"]:
        equal_nested(uninterrupted[key], resumed[key])
    # Fresh runs request pretrained weights; resume rebuilds without downloading.
    assert tiny_encoder == [True, True, False]

    with pytest.raises(FileExistsError):
        run(root, manifest, tmp_path, "full_session")


def test_real_encoder_factory_receives_pretrained_flag(monkeypatch):
    seen = {}
    real = timm.create_model

    def spy(name, *, pretrained, num_classes, **kwargs):
        seen.update(name=name, pretrained=pretrained, num_classes=num_classes)
        return real(name, pretrained=False, num_classes=num_classes, **kwargs)

    import src.models.siamese as siamese
    monkeypatch.setattr(siamese.timm, "create_model", spy)
    model = SiameseClassifier("resnet34", pretrained=True)
    assert seen == {"name": "resnet34", "pretrained": True, "num_classes": 0}
    assert model.backbone.num_features == 512
