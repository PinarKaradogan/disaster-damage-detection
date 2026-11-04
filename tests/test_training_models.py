import pytest
import torch
from timm.models._manipulate import adapt_input_conv

from src.models import create_model
from src.models.fusion import early_fusion


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", ["simple_cnn", "resnet34_early_fusion"])
def test_model_forward_backward(name):
    model = create_model(name, pretrained=False)
    pre, post = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    output = model(pre, post)
    assert output.shape == (2, 4)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 3])).backward()
    first_conv = next(layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d))
    assert first_conv.in_channels == 6
    assert first_conv.weight.grad is not None and torch.isfinite(first_conv.weight.grad).all()


@pytest.mark.parametrize("name", ["simple_cnn", "resnet34_early_fusion"])
def test_mismatched_inputs(name):
    model = create_model(name, pretrained=False)
    with pytest.raises(ValueError, match="matching"):
        model(torch.zeros(2, 3, 128, 128), torch.zeros(2, 3, 64, 64))
    with pytest.raises(ValueError, match="matching"):
        model(torch.zeros(2, 1, 128, 128), torch.zeros(2, 1, 128, 128))


def test_fusion_order_and_pretrained_conversion():
    fused = early_fusion(torch.ones(2, 3, 8, 8), torch.zeros(2, 3, 8, 8))
    assert fused.shape == (2, 6, 8, 8)
    assert fused[:, :3].eq(1).all() and fused[:, 3:].eq(0).all()
    rgb = torch.randn(64, 3, 7, 7)
    adapted = adapt_input_conv(6, rgb)
    assert torch.equal(adapted, rgb.repeat(1, 2, 1, 1) * 0.5)
