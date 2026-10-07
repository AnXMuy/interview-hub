## Transformer Encoder Layer

### 📝 实现代码

```python
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer (BERT 风格)
    
    Pre-Norm 架构:
    x -> LayerNorm -> Self-Attn -> Add -> LayerNorm -> FFN -> Add
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        norm_type: str = "layernorm"
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norms
        if norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        mask: [seq_len, seq_len] attention mask
        key_padding_mask: [batch, seq_len] padding mask
        """
        # Pre-Norm Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_out)
        
        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x


class RMSNorm(nn.Module):
    """RMSNorm 实现"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [ffn](ffn.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
