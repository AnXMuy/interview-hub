import torch
from torch import nn


class FeedForward(nn.Module):
    """FeedForward(dim, hidden_dim): Linear(dim,hidden_dim) -> GELU -> Linear(hidden_dim,dim).

    Expose layers as fc1 and fc2 for parameter-copy reference tests.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
