from torch import nn

from src.models.fusion import early_fusion


class SimpleCNN(nn.Module):
    """Small baseline with PRE/POST concatenated in that order."""

    def __init__(self, num_classes=4):
        super().__init__()
        blocks = []
        channels = [6, 32, 64, 128]
        for inputs, outputs in zip(channels, channels[1:]):
            blocks.extend([
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ])
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, pre, post):
        features = self.features(early_fusion(pre, post))
        return self.classifier(self.pool(features).flatten(1))
