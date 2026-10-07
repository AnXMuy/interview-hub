## KV Cache

### 🎯 核心思想

自回归生成时，缓存历史 token 的 Key 和 Value，避免重复计算。

**没有 KV Cache**：生成第 n 个 token 需要计算 n 次 K, V
**有 KV Cache**：只计算新 token 的 K, V，复用历史缓存

> 🤔 **Q: KV Cache 占多少显存？怎么估算？**
>
> 公式：`2 × num_layers × batch × seq_len × num_kv_heads × head_dim × dtype_bytes`
>
> 例如 LLaMA-2 7B（FP16）：
> - 32 layers, 32 heads, head_dim=128, seq=4096, batch=1
> - = 2 × 32 × 1 × 4096 × 32 × 128 × 2 bytes = **2 GB**
>
> 用 GQA 的话（num_kv_heads=8），只需 **0.5 GB**！

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCache:
    """
    KV Cache 🔥🔥🔥 面试必考
    
    缓存 Attention 的 Key 和 Value，避免重复计算
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # 预分配最大空间
        # 形状: [batch, num_heads, max_seq_len, head_dim]
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        # 当前缓存长度
        self.current_len = 0
    
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple:
        """
        更新缓存并返回完整的 K, V
        
        Args:
            k_new: [batch, num_heads, new_seq_len, head_dim] 新的 Key
            v_new: [batch, num_heads, new_seq_len, head_dim] 新的 Value
        
        Returns:
            k_full: [batch, num_heads, current_len + new_len, head_dim]
            v_full: 同上
        """
        new_len = k_new.size(2)
        
        # 写入新位置
        self.k_cache[:, :, self.current_len:self.current_len + new_len, :] = k_new
        self.v_cache[:, :, self.current_len:self.current_len + new_len, :] = v_new
        
        self.current_len += new_len
        
        # 返回有效部分
        k_full = self.k_cache[:, :, :self.current_len, :]
        v_full = self.v_cache[:, :, :self.current_len, :]
        
        return k_full, v_full
    
    def get_seq_len(self) -> int:
        """获取当前缓存长度"""
        return self.current_len
    
    def clear(self):
        """清空缓存"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.current_len = 0


class AttentionWithKVCache(nn.Module):
    """
    带 KV Cache 的 Attention
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache = None,
        use_cache: bool = True
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Args:
            x: [batch, seq_len, d_model]
               - Prefill 阶段: seq_len 是整个 prompt 长度
               - Decode 阶段: seq_len = 1（每次只处理新 token）
            kv_cache: KV 缓存对象
            use_cache: 是否使用缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用 KV Cache
        if use_cache and kv_cache is not None:
            K, V = kv_cache.update(K, V)
        
        # Attention 计算
        # Q: [batch, heads, seq_q, head_dim] （seq_q 可能只有 1）
        # K, V: [batch, heads, seq_k, head_dim] （seq_k 包含历史）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask（只在 prefill 时需要，decode 时 Q 只有一个 token）
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, K.size(2), device=x.device),
                diagonal=K.size(2) - seq_len + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, kv_cache


# ==================== 内存分析 ====================
def analyze_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16
):
    """分析 KV Cache 内存占用"""
    
    # 每层的 KV Cache: 2 * batch * heads * seq * head_dim
    per_layer = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    total = num_layers * per_layer
    
    print(f"=== KV Cache 内存分析 ===")
    print(f"配置: batch={batch_size}, seq={seq_len}, layers={num_layers}")
    print(f"      heads={num_heads}, head_dim={head_dim}")
    print(f"每层 KV Cache: {per_layer / 1024**2:.2f} MB")
    print(f"总 KV Cache: {total / 1024**3:.2f} GB")
    
    # 对比模型参数量
    model_params = num_layers * (4 * num_heads * head_dim * num_heads * head_dim)  # 简化估计
    print(f"对比：模型参数约 {model_params * dtype_bytes / 1024**3:.2f} GB")


if __name__ == "__main__":
    # LLaMA-7B 配置
    analyze_kv_cache_memory(
        batch_size=1,
        seq_len=4096,
        num_layers=32,
        num_heads=32,
        head_dim=128
    )
```

### 💡 面试追问

**Q: 为什么 KV Cache 能加速？**

> 自回归生成时，历史 token 的 K, V 不变。没有缓存每次要重算 O(n²)，有缓存只需 O(n)。

**Q: KV Cache 的内存瓶颈是什么？**

> 内存随 batch_size × seq_len × num_layers 线性增长。长序列或大 batch 时成为瓶颈。

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：KV Cache 缓存什么？**

答：缓存每层先前 token 的投影 K/V，不缓存 Q。

**问：Prefill 和 decode 瓶颈？**

答：前者偏计算密集，后者常受访存和缓存读取制约。

[返回本章目录](README.md)
