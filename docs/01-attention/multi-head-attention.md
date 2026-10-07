## Multi-Head Attention

### 🎯 核心思想

多头注意力让模型在**不同的表示子空间**中学习不同的注意力模式，然后合并结果。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 完整实现
    
    MHA(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
    其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
    """
    
    def __init__(
        self, 
        d_model: int,           # 模型维度（如 768, 1024）
        num_heads: int,         # 注意力头数（如 12, 16）
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        # 检查维度能否整除
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 线性投影层
        # 实际实现中通常合并 Q, K, V 的投影以提高效率
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier 初始化"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            query: [batch, seq_len_q, d_model]
            key:   [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model]
            mask:  [batch, seq_len_q, seq_len_k] 或可广播
            need_weights: 是否返回注意力权重
        
        Returns:
            output: [batch, seq_len_q, d_model]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_k] (如果 need_weights=True)
        """
        batch_size = query.size(0)
        
        # ========== Step 1: 线性投影 ==========
        # [batch, seq, d_model] -> [batch, seq, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # ========== Step 2: 拆分多头 ==========
        # [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
        # -> [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ========== Step 3: 计算注意力 ==========
        # 调整 mask 维度 [batch, 1, 1, seq_k] 用于广播
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
        # -> [batch, heads, seq_q, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # ========== Step 4: 合并多头 ==========
        # [batch, heads, seq_q, head_dim] -> [batch, seq_q, heads, head_dim]
        # -> [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # ========== Step 5: 输出投影 ==========
        output = self.W_o(attn_output)
        
        if need_weights:
            return output, attn_weights
        return output, None


# ==================== 验证代码 ====================
def test_mha():
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 自注意力
    output, weights = mha(x, x, x, need_weights=True)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")
    
    assert output.shape == x.shape, "输出维度错误!"
    print("✅ MHA 测试通过!")

if __name__ == "__main__":
    test_mha()
```

### 💡 面试追问

**Q1: 为什么要用多头而不是一个大的 Attention？**

> 多头让模型在不同子空间学习不同的注意力模式（如语法、语义、位置关系），类似 CNN 中的多个 filter。

**Q2: head_dim 和 num_heads 怎么选？**

> 通常 d_model / num_heads = 64 或 128。头数太多单头容量不足，太少多样性不够。

**Q3: 为什么要除以 √d_k？**

> 假设 q, k 是独立的均值 0 方差 1 的随机变量，点积的方差 = d_k。不缩放的话，d_k 大时 softmax 输入值大，梯度趋近于 0。

> 🤔 **Q: transpose(1, 2) 是在干什么？为什么需要 contiguous()？**
>
> `transpose(1, 2)` 把 `[batch, seq, heads, dim]` 变成 `[batch, heads, seq, dim]`，让每个 head 可以独立做矩阵乘法。
>
> `contiguous()` 是因为 transpose 只改变 stride 不改变内存布局，view() 需要连续内存。

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
