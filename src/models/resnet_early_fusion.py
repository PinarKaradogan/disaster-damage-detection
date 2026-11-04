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

    def __init__(self, num_classes=4, pretrained=True, training_mode="full"):
        super().__init__()
        if training_mode not in {"full", "head_only"}:
            raise ValueError("training_mode must be full or head_only")
        self.training_mode = training_mode
        self.backbone = timm.create_model(
            "resnet34", pretrained=pretrained, in_chans=6, num_classes=num_classes,
        )
        if training_mode == "head_only":
            self.backbone.requires_grad_(False)
            self.backbone.get_classifier().requires_grad_(True)
            self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.training_mode == "head_only":
            # run_epoch calls model.train() each epoch: keep frozen BN buffers fixed.
            self.backbone.eval()
            self.backbone.get_classifier().train(mode)
        return self

    def forward(self, pre, post):
        return self.backbone(early_fusion(pre, post))
