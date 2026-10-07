import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                  ignore_index: int = -100) -> torch.Tensor:
    """[B,C] logits, [B] long targets; stable mean over non-ignored rows."""
    raise NotImplementedError
