import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """q/k/v: [B,H,T,D]; mask: broadcastable bool [B,H,Tq,Tk], True=visible."""
    raise NotImplementedError
