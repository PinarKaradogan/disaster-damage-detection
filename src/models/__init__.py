"""Phase 4 paired-image classifiers."""


def create_model(name, *, num_classes=4, pretrained=True):
    if name == "simple_cnn":
        from src.models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet34_early_fusion":
        from src.models.resnet_early_fusion import ResNet34EarlyFusion
        return ResNet34EarlyFusion(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
