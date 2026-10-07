import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm(dim, eps=1e-6), trainable weight initialized to ones; normalize last dim."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
