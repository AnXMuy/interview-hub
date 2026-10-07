import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B,T,D], mask: bool [B,T]; return [B,D] per-sample valid-token mean.

    For an all-masked sample return zeros. Preserve autograd connectivity.
    """
    raise NotImplementedError
