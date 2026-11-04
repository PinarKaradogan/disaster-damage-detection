"""Paired-image classifiers: Phase 4 early fusion and Phase 5 Siamese models."""

# Models whose backbone loads ImageNet weights (recorded in experiment metadata).
PRETRAINED_MODELS = {"resnet34_early_fusion", "siamese_resnet34", "siamese_vit_small"}
MODEL_NAMES = ("simple_cnn", "resnet34_early_fusion", "siamese_resnet34", "siamese_vit_small")


def create_model(name, *, num_classes=4, pretrained=True, training_mode="full"):
    if training_mode not in {"full", "head_only"}:
        raise ValueError("training_mode must be full or head_only")
    if name.startswith("siamese_"):
        if training_mode != "full":
            raise ValueError("head_only training requires resnet34_early_fusion")
        from src.models.siamese import SIAMESE_ENCODERS, create_siamese
        if name not in SIAMESE_ENCODERS:
            raise ValueError(f"Unknown model: {name}")
        return create_siamese(name, num_classes=num_classes, pretrained=pretrained)
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
