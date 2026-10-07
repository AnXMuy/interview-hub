import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from grader.__main__ import (check, test_attention, test_efficient_training,
                             test_inference_optimization, test_loss_functions,
                             test_normalization, test_optimizers, test_position_encoding,
                             test_reinforcement_learning, test_sampling, test_torch_basics,
                             test_transformer_architecture)


class GraderTests(unittest.TestCase):
    def test_shape_diagnostic(self):
        with self.assertRaisesRegex(AssertionError, '形状'):
            check(torch.zeros(2), torch.zeros(3), 'shape')

    def test_value_diagnostic(self):
        with self.assertRaisesRegex(AssertionError, '最大绝对误差'):
            check(torch.zeros(2), torch.ones(2), 'value')

    def test_attention_reference(self):
        def reference(q, k, v, mask=None):
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            if mask is not None:
                scores = scores.masked_fill(~mask, -torch.inf)
            return scores.softmax(-1) @ v
        test_attention(SimpleNamespace(scaled_dot_product_attention=reference))

    def test_torch_basics_reference(self):
        def reference(x, mask):
            weights = mask.unsqueeze(-1)
            return (x * weights).sum(1) / mask.sum(1).clamp(min=1).unsqueeze(-1)
        test_torch_basics(SimpleNamespace(masked_mean=reference))

    def test_sampling_reference(self):
        def reference(logits, p):
            sorted_logits, ids = logits.sort(-1, descending=True)
            probs = sorted_logits.softmax(-1)
            drop = probs.cumsum(-1) - probs >= p
            drop[:, 0] = False
            return logits.masked_fill(torch.zeros_like(drop).scatter(-1, ids, drop), -torch.inf)
        test_sampling(SimpleNamespace(top_p_filter=reference))

    def test_cross_entropy_reference(self):
        test_loss_functions(SimpleNamespace(cross_entropy=lambda z, y: F.cross_entropy(z, y)))

    def test_ffn_reference(self):
        class FFN(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, dim)

            def forward(self, x):
                return self.fc2(F.gelu(self.fc1(x)))
        test_transformer_architecture(SimpleNamespace(FeedForward=FFN))

    def test_rmsnorm_reference(self):
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x):
                return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight
        test_normalization(SimpleNamespace(RMSNorm=RMSNorm))

    def test_rope_reference(self):
        def rope(x, offset=0, base=10000.0):
            dim = x.size(-1)
            pos = torch.arange(offset, offset + x.size(-2), device=x.device, dtype=x.dtype)
            theta = pos[:, None] * base ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
            pairs = x.reshape(*x.shape[:-1], dim // 2, 2)
            return torch.stack((pairs[..., 0] * theta.cos() - pairs[..., 1] * theta.sin(),
                                pairs[..., 0] * theta.sin() + pairs[..., 1] * theta.cos()), -1).reshape_as(x)
        test_position_encoding(SimpleNamespace(apply_rope=rope))

    def test_adamw_reference(self):
        def step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad.square()
            param = param * (1 - lr * weight_decay) - lr * m / (1 - beta1 ** t) / ((v / (1 - beta2 ** t)).sqrt() + eps)
            return param, m, v
        test_optimizers(SimpleNamespace(adamw_step=step))

    def test_gae_reference(self):
        def gae(rewards, values, dones, gamma=0.99, lam=0.95):
            out = torch.zeros_like(rewards)
            running = 0.
            for t in reversed(range(len(rewards))):
                running = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t] + gamma * lam * (1 - dones[t]) * running
                out[t] = running
            return out
        test_reinforcement_learning(SimpleNamespace(compute_gae=gae))

    def test_lora_reference(self):
        class LoRA(nn.Module):
            def __init__(self, base, rank, alpha):
                super().__init__()
                self.base = base
                self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
                self.B = nn.Parameter(torch.zeros(base.out_features, rank))
                self.scale = alpha / rank
                base.requires_grad_(False)

            def forward(self, x):
                return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
        test_efficient_training(SimpleNamespace(LoRALinear=LoRA))

    def test_cache_reference(self):
        class Cache:
            def __init__(self):
                self.reset()

            def append(self, k, v):
                self.k = k if self.k is None else torch.cat((self.k, k), dim=2)
                self.v = v if self.v is None else torch.cat((self.v, v), dim=2)
                return self.k, self.v

            def reset(self):
                self.k, self.v = None, None
        test_inference_optimization(SimpleNamespace(KVCache=Cache))


if __name__ == '__main__':
    unittest.main()
