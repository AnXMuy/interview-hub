## Transformer 概览

Encoder 的 self-attention 可双向读取输入；Decoder 的 self-attention 必须因果遮罩，Encoder-Decoder 还使用 cross-attention 从输入序列读取信息。下面的代码展示这些部件的张量接口，不代替完整模型实现。

```
原始 Transformer (Encoder-Decoder):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    Encoder                         Decoder              │
│    ┌─────────┐                    ┌─────────┐          │
│    │  MHA    │                    │ Masked  │          │
│    │ (Self)  │                    │  MHA    │          │
│    └────┬────┘                    └────┬────┘          │
│         │                              │               │
│    ┌────┴────┐                    ┌────┴────┐          │
│    │   FFN   │                    │  Cross  │          │
│    └────┬────┘                    │   MHA   │          │
│         │                         └────┬────┘          │
│    x N layers                          │               │
│                                   ┌────┴────┐          │
│                                   │   FFN   │          │
│                                   └────┬────┘          │
│                                        │               │
│                                   x N layers           │
└─────────────────────────────────────────────────────────┘

现代 LLM (Decoder-Only):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    Decoder-Only (GPT/LLaMA/...)                        │
│    ┌─────────┐                                          │
│    │ Masked  │  ← Causal Mask 保证自回归                │
│    │  MHA    │                                          │
│    └────┬────┘                                          │
│         │                                               │
│    ┌────┴────┐                                          │
│    │   FFN   │  ← 通常 4x 或 8/3x hidden_size          │
│    └────┬────┘                                          │
│         │                                               │
│    x N layers                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 最小代码：三种注意力数据流

```python
import torch
from torch import nn

d = 16
src = torch.randn(2, 5, d)  # [batch, source_length, hidden]
tgt = torch.randn(2, 3, d)  # [batch, target_length, hidden]
attention = nn.MultiheadAttention(d, num_heads=4, batch_first=True)

# Encoder：每个源 token 可以看见所有源 token。
encoder_out, _ = attention(src, src, src, need_weights=False)

# Decoder 自注意力：上三角为 True 的位置被阻止看到未来 token。
causal_mask = torch.ones(3, 3, dtype=torch.bool).triu(diagonal=1)
decoder_self, _ = attention(tgt, tgt, tgt, attn_mask=causal_mask, need_weights=False)

# Cross-attention：Query 来自 decoder，Key/Value 来自 encoder。
decoder_cross, _ = attention(decoder_self, encoder_out, encoder_out, need_weights=False)
assert decoder_cross.shape == (2, 3, d)
```

注意这里 `nn.MultiheadAttention` 的布尔 `attn_mask=True` 表示禁止访问；练习中的 `scaled_dot_product_attention` 则约定 `mask=True` 表示可见。接口不同，不可直接复用而不取反。

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [ffn](ffn.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
