## ALiBi

### 🎯 核心思想

ALiBi (Attention with Linear Biases) 完全不修改 Q/K，而是直接在 attention score 上加一个与距离成正比的偏置。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{m} \cdot \text{dist}\right)V$$

其中 `dist[i,j] = -|i-j|`（距离越远，偏置越负）

> 🤔 **Q: ALiBi 的斜率 m 是怎么设计的？为什么不同 head 斜率不同？**
>
> 斜率 $m = 2^{-8/n}, 2^{-8 \cdot 2/n}, ...$，呈指数递减。
>
> 斜率大的 head 更关注局部（远距离惩罚大），斜率小的 head 能看到更远的信息。
>
> 这让不同 head “分工”关注不同范围。

### 📝 实现代码

```python
class ALiBi(nn.Module):
    """
    Attention with Linear Biases
    
    BLOOM 模型使用
    优点：天然支持长度外推
    """
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        # 每个 head 有不同的斜率 m
        # m = 2^(-8/n), 2^(-8*2/n), ..., 2^(-8*n/n)
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, n: int) -> torch.Tensor:
        """
        计算每个 head 的斜率
        
        使用指数递减的斜率，让不同 head 关注不同范围
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            # 非 2 的幂次，插值
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            # 添加额外的斜率
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            slopes = slopes + extra_slopes[:n - closest_power_of_2]
            return torch.tensor(slopes)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成 ALiBi 偏置矩阵
        
        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # 距离矩阵: dist[i,j] = j - i（右边为正，左边为负）
        positions = torch.arange(seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
        
        # 因果mask: 只保留下三角（未来位置为负无穷）
        # 这里 distance 保持原样，由调用者决定如何处理因果性
        
        # 乘以斜率: [num_heads, 1, 1] * [seq, seq] -> [num_heads, seq, seq]
        bias = self.slopes.view(-1, 1, 1) * distance.unsqueeze(0)
        
        return bias


def apply_alibi_to_attention(scores: torch.Tensor, alibi_bias: torch.Tensor) -> torch.Tensor:
    """
    将 ALiBi 偏置应用到 attention scores
    
    Args:
        scores: [batch, num_heads, seq_q, seq_k]
        alibi_bias: [num_heads, seq_q, seq_k]
    """
    return scores + alibi_bias.unsqueeze(0)


# ==================== 使用示例 ====================
def test_alibi():
    num_heads = 8
    seq_len = 16
    
    alibi = ALiBi(num_heads)
    bias = alibi(seq_len)
    
    print(f"ALiBi 偏置形状: {bias.shape}")
    print(f"斜率: {alibi.slopes}")
    
    # 可视化第一个 head 的偏置
    print(f"\nHead 0 的偏置矩阵 (前 8x8):")
    print(bias[0, :8, :8].numpy().round(2))


if __name__ == "__main__":
    test_alibi()
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rope](rope.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：RoPE 施加到哪些张量？**

答：旋转 Q、K 而非 V；点积编码相对位置。

**问：位置偏移如何处理？**

答：增量解码时用全局位置，不可每步重新从零计数。

[返回本章目录](README.md)
