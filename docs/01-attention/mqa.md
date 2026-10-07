## Multi-Query Attention (MQA)

### 🎯 核心思想

MQA 是 GQA 的极端情况：**所有 Query 头共享同一个 Key-Value**。

> 🤔 **Q: repeat_interleave 和 expand 有什么区别？能不能用 expand？**
>
> 可以！`expand` 不复制内存，`repeat_interleave` 会复制。
>
> 实际上用 `expand` 更高效：`K = K.expand(-1, self.num_heads, -1, -1)`
>
> 但 `repeat_interleave` 语义更清晰，面试写这个没问题。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA)
    
    所有 Query 头共享同一组 K, V
    极大减少 KV Cache，但可能损失一些表达能力
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q: 多头
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K, V: 只有一个头！
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Q: [batch, seq, num_heads * head_dim]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        
        # K, V: [batch, seq, head_dim] -> [batch, 1, seq, head_dim]
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)
        
        # 广播：K, V 自动扩展到 [batch, num_heads, seq, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.W_o(output)
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [scaled-dot-product-attention](scaled-dot-product-attention.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
