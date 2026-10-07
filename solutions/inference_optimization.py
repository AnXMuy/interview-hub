import torch


class KVCache:
    """KVCache(); append(k,v) with [B,H,T,D], return concatenated (k,v); reset() clears."""

    def __init__(self):
        raise NotImplementedError

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
