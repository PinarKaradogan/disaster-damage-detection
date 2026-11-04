"""Phase 4 paired-image classifiers."""


def create_model(name, *, num_classes=4, pretrained=True, training_mode="full"):
    if training_mode not in {"full", "head_only"}:
        raise ValueError("training_mode must be full or head_only")
    if name == "simple_cnn":
        if training_mode != "full":
            raise ValueError("head_only training requires resnet34_early_fusion")
        from src.models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet34_early_fusion":
        from src.models.resnet_early_fusion import ResNet34EarlyFusion
        return ResNet34EarlyFusion(num_classes=num_classes, pretrained=pretrained,
                                   training_mode=training_mode)
    raise ValueError(f"Unknown model: {name}")
