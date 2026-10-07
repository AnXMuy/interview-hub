import torch
from torch import nn


class LoRALinear(nn.Module):
    """LoRALinear(base: nn.Linear, rank: int, alpha: float); freeze base.

    Expose trainable A [rank,in_features] and B [out_features,rank]; B starts zero.
    forward(x) = base(x) + (x @ A.T @ B.T) * alpha/rank.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
