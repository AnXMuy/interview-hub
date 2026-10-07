## Grouped-Query Attention (GQA)

### 🎯 核心思想

GQA 是 MHA 和 MQA 的折中：**多个 Query 头共享同一组 Key-Value 头**。

- MHA: num_kv_heads = num_heads（每个 Q 头有独立的 KV）
- MQA: num_kv_heads = 1（所有 Q 头共享一个 KV）
- GQA: 1 < num_kv_heads < num_heads（分组共享）

**优势**：减少 KV Cache 内存，同时保持接近 MHA 的效果。

### 📝 实现代码

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    
    LLaMA-2 70B, Mistral 等使用
    通过减少 KV 头数来降低 KV Cache 内存
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,          # Query 头数
        num_kv_heads: int = None,  # KV 头数（默认等于 num_heads，即 MHA）
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        
        # 检查 num_heads 能否被 num_kv_heads 整除
        assert num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        
        # 每个 KV 头对应的 Q 头数
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Q 投影: d_model -> num_heads * head_dim
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        # K, V 投影: d_model -> num_kv_heads * head_dim（减少了！）
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 投影
        Q = self.W_q(x)  # [batch, seq, num_heads * head_dim]
        K = self.W_k(x)  # [batch, seq, num_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq, num_kv_heads * head_dim]
        
        # 重塑 Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 重塑 K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # ========== 关键：扩展 KV 以匹配 Q 头数 ==========
        # 每个 KV 头重复 num_queries_per_kv 次
        # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # 标准 Attention 计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        
        return output


# ==================== 内存分析 ====================
def analyze_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_kv_heads: int,
    dtype_bytes: int = 2  # FP16 = 2 bytes
):
    """分析 KV Cache 内存占用"""
    head_dim = d_model // num_heads
    
    # MHA: 2 * batch * num_heads * seq * head_dim
    mha_memory = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    
    # GQA: 2 * batch * num_kv_heads * seq * head_dim
    gqa_memory = 2 * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
    
    print(f"=== KV Cache 内存分析 ===")
    print(f"配置: batch={batch_size}, seq={seq_len}, d_model={d_model}")
    print(f"       num_heads={num_heads}, num_kv_heads={num_kv_heads}")
    print(f"MHA 内存: {mha_memory / 1024**2:.2f} MB")
    print(f"GQA 内存: {gqa_memory / 1024**2:.2f} MB")
    print(f"节省: {(1 - gqa_memory/mha_memory) * 100:.1f}%")


if __name__ == "__main__":
    # 模拟 LLaMA-2 70B 配置
    analyze_kv_cache_memory(
        batch_size=1,
        seq_len=4096,
        d_model=8192,
        num_heads=64,
        num_kv_heads=8  # GQA 配置
    )
```

### 💡 面试追问

**Q: GQA 和 MQA 的区别？**

| 方法 | num_kv_heads | KV Cache 内存 | 质量 |
|:---|:---:|:---:|:---:|
| MHA | = num_heads | 100% | 最好 |
| GQA | 介于两者之间 | 12.5%-50% | 接近 MHA |
| MQA | 1 | 最小 | 略有下降 |

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
