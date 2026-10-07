## Scaled Dot-Product Attention

### 🎯 核心思想

Attention 的本质是**加权求和**：根据 Query 和 Key 的相似度，对 Value 进行加权聚合。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention
    
    Args:
        query: [batch, num_heads, seq_len_q, head_dim]
        key:   [batch, num_heads, seq_len_k, head_dim]
        value: [batch, num_heads, seq_len_v, head_dim]  (seq_len_v == seq_len_k)
        mask:  [batch, 1, seq_len_q, seq_len_k] 或可广播的形状
        dropout_p: Dropout 概率
        training: 是否训练模式
    
    Returns:
        output: [batch, num_heads, seq_len_q, head_dim]
        attn_weights: [batch, num_heads, seq_len_q, seq_len_k]
    """
    d_k = query.size(-1)
    
    # Step 1: 计算注意力分数 Q @ K^T
    # [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
    # -> [batch, heads, seq_q, seq_k]
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: 缩放（防止点积过大导致 softmax 梯度消失）
    # 为什么除以 sqrt(d_k)？
    # 假设 q, k 是均值0方差1的独立随机变量，q·k 的方差 = d_k
    # 除以 sqrt(d_k) 使方差归一化为 1
    scores = scores / math.sqrt(d_k)
    
    # Step 3: 应用 mask（如果有）
    # mask=True/1 的位置会被置为 -inf，softmax 后变成 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Dropout（训练时）
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Step 6: 加权求和
    # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
    # -> [batch, heads, seq_q, head_dim]
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights
```

### 🔍 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|:---|:---:|:---:|
| Q @ K^T | O(n² · d) | O(n²) |
| Softmax | O(n²) | O(n²) |
| Attn @ V | O(n² · d) | O(n · d) |
| **总计** | **O(n² · d)** | **O(n²)** |

> ⚠️ **瓶颈**：n² 的复杂度是长序列的主要瓶颈！

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
