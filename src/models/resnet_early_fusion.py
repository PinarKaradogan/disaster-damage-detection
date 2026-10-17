import timm
from torch import nn

from src.models.fusion import early_fusion


class ResNet34EarlyFusion(nn.Module):
    """timm ResNet34, six-channel PRE/POST input, four output logits.

    With pretrained=True timm adapts the pretrained RGB first-conv weights
    by repeating the RGB channels twice and scaling each by 3/6 (0.5).
    The four-class head is freshly initialized. No fallback to random weights
    is made on download/cache errors. CPU/CI tests explicitly use False.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet34", pretrained=pretrained, in_chans=6, num_classes=num_classes,
        )

    def forward(self, pre, post):
        return self.backbone(early_fusion(pre, post))
