"""Phase 5 Siamese classifiers: one shared timm encoder for PRE and POST."""
import timm
import torch
from torch import nn

from src.models.fusion import check_pair

# Registered model name -> timm encoder name and extra create_model kwargs.
SIAMESE_ENCODERS = {
    "siamese_resnet34": ("resnet34", {}),
}


class SiameseClassifier(nn.Module):
    """PRE and POST go through the SAME encoder (one module, one set of weights).

    Features are fused as [f_pre, f_post, |f_pre - f_post|] and classified by a small
    MLP head into four damage logits. Standard 3-channel ImageNet weights are used
    as-is; no first-conv adaptation is needed (unlike early fusion).
    Both images are encoded in one batched call, so BatchNorm statistics are computed
    over PRE and POST together.
    """

    training_mode = "full"

    def __init__(self, encoder_name, *, num_classes=4, pretrained=True, hidden_dim=256, dropout=0.2,
                 encoder_kwargs=None):
        super().__init__()
        # Attribute name matches ResNet34EarlyFusion so metadata code can read pretrained_cfg.
        self.backbone = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0,
                                          **(encoder_kwargs or {}))
        features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(3 * features, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def encode_pair(self, pre, post):
        check_pair(pre, post)
        features = self.backbone(torch.cat([pre, post], dim=0))
        return features[:len(pre)], features[len(pre):]

    def forward(self, pre, post):
        f_pre, f_post = self.encode_pair(pre, post)
        return self.head(torch.cat([f_pre, f_post, (f_pre - f_post).abs()], dim=1))


def create_siamese(name, *, num_classes=4, pretrained=True):
    encoder_name, kwargs = SIAMESE_ENCODERS[name]
    return SiameseClassifier(encoder_name, num_classes=num_classes, pretrained=pretrained,
                             encoder_kwargs=kwargs)
