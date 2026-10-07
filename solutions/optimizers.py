import torch


def adamw_step(param: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
               step: int, lr: float = 1e-3, beta1: float = 0.9,
               beta2: float = 0.999, eps: float = 1e-8,
               weight_decay: float = 0.01) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return new (param,m,v) with bias correction and decoupled weight decay; no mutation."""
    raise NotImplementedError
