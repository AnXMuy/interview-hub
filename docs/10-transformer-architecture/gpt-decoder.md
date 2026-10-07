## GPT-style Decoder-Only

### 🎯 核心思想

现代 LLM（GPT、LLaMA、Mistral 等）都使用 Decoder-Only 架构：
- 没有 Encoder，没有 Cross-Attention
- 只有 Causal Self-Attention
- 更简单，更容易扩展

### 📝 实现代码

```python
class GPTDecoderLayer(nn.Module):
    """
    GPT/LLaMA 风格的 Decoder Layer
    
    只有 Masked Self-Attention + FFN
    没有 Cross-Attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = None,  # GQA
        d_ff: int = None,
        dropout: float = 0.0,
        use_swiglu: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 使用 GQA 或 MHA
        if num_kv_heads:
            self.self_attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        
        # 使用 SwiGLU 或标准 FFN
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout, activation="gelu")
        
        # RMSNorm (现代 LLM 标配)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        kv_cache = None
    ) -> tuple[torch.Tensor, any]:
        """
        x: [batch, seq_len, d_model]
        mask: causal mask
        kv_cache: KV Cache 对象
        """
        # Pre-Norm Masked Self-Attention
        residual = x
        x = self.norm1(x)
        
        if hasattr(self.self_attn, 'forward_with_cache'):
            # 自定义 Attention（支持 KV Cache）
            attn_out, kv_cache = self.self_attn.forward_with_cache(x, mask, kv_cache)
        else:
            # PyTorch 标准 MHA
            attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        
        x = residual + attn_out
        
        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x, kv_cache


class GroupedQueryAttention(nn.Module):
    """GQA 实现（简化版）"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 扩展 KV
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=2)
        V = V.repeat_interleave(repeat_factor, dim=2)
        
        # Transpose for attention
        Q = Q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out), None
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
