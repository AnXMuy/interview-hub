import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
    """[T] rewards/dones, [T+1] values; return [T] advantages; done=1 ends episode."""
    raise NotImplementedError
