## Transformer Decoder Layer

### 📝 实现代码

```python
class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer (原始 Transformer 风格)
    
    包含:
    1. Masked Self-Attention (因果)
    2. Cross-Attention (对 Encoder 输出)
    3. FFN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # Masked Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Norms (Pre-Norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: [batch, tgt_len, d_model] decoder 输入
        encoder_output: [batch, src_len, d_model] encoder 输出
        """
        # Masked Self-Attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=self_attn_mask
        )
        x = residual + self.dropout(attn_out)
        
        # Cross-Attention
        residual = x
        x = self.norm2(x)
        cross_out, _ = self.cross_attn(
            x,                # Query: 来自 decoder
            encoder_output,   # Key: 来自 encoder
            encoder_output,   # Value: 来自 encoder
            attn_mask=cross_attn_mask
        )
        x = residual + self.dropout(cross_out)
        
        # FFN
        residual = x
        x = self.norm3(x)
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)
        
        return x
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
