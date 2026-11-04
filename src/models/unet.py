"""Phase 5 building localization model: U-Net with an ImageNet ResNet encoder."""

UNET_ENCODERS = {"unet_resnet34": "resnet34"}


def create_unet(name="unet_resnet34", *, pretrained=True):
    """Binary building segmentation: 3-channel PRE image -> 1 logit per pixel.

    pretrained=True loads segmentation-models-pytorch ImageNet encoder weights
    (smp-hub/resnet34.imagenet); tests and CI use pretrained=False.
    """
    if name not in UNET_ENCODERS:
        raise ValueError(f"Unknown segmentation model: {name}")
    import segmentation_models_pytorch as smp
    return smp.Unet(UNET_ENCODERS[name], encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def pretrained_source(name="unet_resnet34"):
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import encoders
    return {"library": "segmentation-models-pytorch", "version": smp.__version__,
            "encoder": UNET_ENCODERS[name], **encoders[UNET_ENCODERS[name]]["pretrained_settings"]["imagenet"]}
