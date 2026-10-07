import torch


def apply_rope(x: torch.Tensor, offset: int = 0, base: float = 10000.0) -> torch.Tensor:
    """x: [B,H,T,D], D even; rotate adjacent pairs (0,1),(2,3),... at offset+t."""
    raise NotImplementedError
