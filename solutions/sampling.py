import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """[B,V] logits; keep smallest sorted prefix with cumulative softmax >= p; mask rest -inf."""
    raise NotImplementedError
