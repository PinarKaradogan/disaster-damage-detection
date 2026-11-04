import torch


def check_pair(pre, post):
    if pre.ndim != 4 or post.ndim != 4 or pre.shape != post.shape or pre.shape[1] != 3:
        raise ValueError("PRE/POST must have matching [B, 3, H, W] shapes")
    if pre.dtype != post.dtype or pre.device != post.device:
        raise ValueError("PRE/POST must have matching dtype and device")


def early_fusion(pre, post):
    check_pair(pre, post)
    return torch.cat([pre, post], dim=1)
