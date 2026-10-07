## Rotary Position Embedding (RoPE)

### 🎯 核心思想

RoPE 将位置信息编码为**旋转**，通过复数乘法实现。

**核心公式**：
$$q_m \cdot k_n = \text{Re}[(R_m q)(R_n k)^*] = f(q, k, m-n)$$

即：经过 RoPE 变换后，Q 和 K 的点积只依赖于**相对位置** $m-n$。

### 📝 实现代码

```python
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 标配
    
    核心思想：
    1. 将 Q, K 向量分成两两一组
    2. 对每组应用 2D 旋转矩阵
    3. 不同维度使用不同的旋转频率
    """
    
    def __init__(
        self,
        dim: int,           # head_dim
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算 cos/sin 缓存"""
        # 位置索引 [seq_len]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 计算角度 [seq_len, dim/2]
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 注册缓存 [seq_len, dim/2]
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 到 Q 和 K
        
        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_heads, seq_len, head_dim]
            seq_len: 序列长度
        
        Returns:
            旋转后的 q, k
        """
        if seq_len is None:
            seq_len = q.size(2)
        
        # 获取 cos/sin [seq_len, dim/2]
        cos = self.cos_cached[:seq_len]  # [seq_len, dim/2]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim/2]
        
        # 应用旋转
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        应用旋转变换
        
        旋转公式 (对于每对相邻维度):
        [x1, x2] @ [[cos, -sin], [sin, cos]] = [x1*cos - x2*sin, x1*sin + x2*cos]
        """
        # 将 x 分成两半: [x0, x1, x2, x3, ...] -> 前半和后半
        # 分成 pairs: (x0, x1), (x2, x3), ... 其中 x1=x[..., :d//2], x2=x[..., d//2:]
        d = x.shape[-1]
        x1 = x[..., :d//2]   # 前一半 [batch, heads, seq, dim/2]
        x2 = x[..., d//2:]   # 后一半 [batch, heads, seq, dim/2]
        
        # 调整 cos/sin 维度以广播 [1, 1, seq, dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 应用旋转矩阵
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # 拼接回去 [batch, heads, seq, dim]
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


class RoPESimple(nn.Module):
    """
    RoPE 简化版（面试手写推荐）
    
    使用复数形式，代码更简洁
    """
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Args:
            q, k: [batch, heads, seq_len, head_dim]
        """
        seq_len = q.size(2)
        
        # 计算角度 [seq_len, dim/2]
        t = torch.arange(seq_len, device=q.device, dtype=q.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # 构造复数形式的旋转因子 e^(i*θ)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # cos + i*sin
        
        # 将 q, k 转为复数 (每两个相邻维度组成一个复数)
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        
        # 复数乘法 = 旋转
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
        q_rotated = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
        k_rotated = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
        
        return q_rotated.type_as(q), k_rotated.type_as(k)


# ==================== 验证代码 ====================
def test_rope():
    batch, heads, seq, dim = 2, 8, 64, 64
    
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    
    rope = RotaryPositionEmbedding(dim)
    q_rot, k_rot = rope(q, k)
    
    print(f"Q 形状: {q.shape} -> {q_rot.shape}")
    print(f"K 形状: {k.shape} -> {k_rot.shape}")
    
    # 验证相对位置性质
    # Q_m · K_n 应该只依赖于 m-n
    # 这里简单验证形状正确
    assert q_rot.shape == q.shape
    print("✅ RoPE 测试通过!")


if __name__ == "__main__":
    test_rope()
```

### 🔍 RoPE 的数学原理

> 🤔 **Q: 为什么用旋转而不是加法编码位置？**
>
> 旋转的妙处在于：旋转矩阵的乘法等于角度相加！
>
> $R_m^T \cdot R_n = R_{n-m}$
>
> 所以 $q_m \cdot k_n = (R_m q)^T (R_n k) = q^T R_{n-m} k$，只依赖相对位置！

```
假设二维情况（一对相邻维度）：
原始: q = [q1, q2], k = [k1, k2]

位置 m 的旋转矩阵:
R_m = [[cos(mθ), -sin(mθ)],
       [sin(mθ),  cos(mθ)]]

应用旋转:
q_m = R_m @ q = [q1*cos(mθ) - q2*sin(mθ), q1*sin(mθ) + q2*cos(mθ)]

关键性质（面试必考！）:
q_m · k_n = (R_m @ q)^T @ (R_n @ k)
         = q^T @ R_m^T @ R_n @ k
         = q^T @ R_{n-m} @ k    (因为旋转矩阵相乘 = 角度相加)
         
结论: 点积只依赖于相对位置 (n-m)，而非绝对位置！
```

### 💡 面试追问

**Q: RoPE 如何实现长度外推？**

> 基础 RoPE 外推能力有限。改进方案：
> - **NTK-aware**: 调整 base 参数
> - **YaRN**: 结合 NTK 和插值
> - **Dynamic NTK**: 动态调整频率

**Q: 为什么 RoPE 要在每个 head 内部应用，而不是整个 hidden？**

> 每个 head 独立处理位置信息，让不同 head 可以学习不同的位置关系模式。

> 🤔 **Q: 为什么用 `einsum('i,j->ij', t, inv_freq)` 而不是直接乘？**
>
> `einsum('i,j->ij', t, inv_freq)` 等价于 `torch.outer(t, inv_freq)`，
> 是计算外积：`t[i] * inv_freq[j]` 得到 `[seq_len, dim/2]`。
>
> 直接乘法 `t * inv_freq` 要求维度相同，而这里维度不同。

---

## 动手练习

本题的单接口练习见 [练习契约](../../grader/README.md)；仅编辑 `solutions/position_encoding.py`，运行 `python -m grader 03-position-encoding`。原文代码为学习示例，测评以练习契约为准。

## 常见面试问答

**问：RoPE 施加到哪些张量？**

答：旋转 Q、K 而非 V；点积编码相对位置。

**问：位置偏移如何处理？**

答：增量解码时用全局位置，不可每步重新从零计数。

[返回本章目录](README.md)
