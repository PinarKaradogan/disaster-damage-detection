"""Independent imbalance treatments: weighted CE, focal, or sampler + CE."""
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.shape != (4,) or not torch.isfinite(alpha).all() or (alpha <= 0).any():
                raise ValueError("alpha must contain four finite positive weights")
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        # log_softmax avoids taking log of underflowed probabilities under AMP.
        log_pt = F.log_softmax(logits.float(), dim=1).gather(1, targets[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def make_criterion(loss, sampler="none", weights=None):
    if sampler not in {"none", "weighted"}:
        raise ValueError("Unknown sampler")
    if sampler == "weighted" and loss != "ce":
        raise ValueError("Weighted sampler must be used with standard CE only")
    if loss == "ce":
        return nn.CrossEntropyLoss()
    if loss == "weighted_ce":
        if weights is None:
            raise ValueError("weighted_ce requires train class weights")
        return nn.CrossEntropyLoss(weight=weights)
    if loss == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"Unknown loss: {loss}")
