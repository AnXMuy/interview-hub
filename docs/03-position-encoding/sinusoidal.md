## Sinusoidal Position Encoding

### 🎯 核心思想

原始 Transformer 的位置编码：使用不同频率的 sin/cos 函数，让模型学习相对位置关系。

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**直觉**：
- 不同维度使用不同频率
- 低维度频率高（变化快，捕捉局部位置）
- 高维度频率低（变化慢，捕捉全局位置）

### 📝 实现代码

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal Position Encoding
    
    原始 Transformer 论文 "Attention Is All You Need" 的位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 频率分母: 10000^(2i/d_model)
        # 使用 log 空间计算，避免大数溢出
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度: cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不参与训练，但会保存到 state_dict）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + position_encoding
        """
        seq_len = x.size(1)
        # 加上位置编码
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


def visualize_sinusoidal_pe(d_model: int = 128, max_len: int = 100):
    """可视化位置编码"""
    import matplotlib.pyplot as plt
    
    pe = SinusoidalPositionEncoding(d_model, max_len)
    encoding = pe.pe[0].numpy()  # [max_len, d_model]
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(encoding, cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title('Sinusoidal Position Encoding')
    plt.savefig('sinusoidal_pe.png')
    print("已保存可视化图片: sinusoidal_pe.png")


if __name__ == "__main__":
    # 测试
    pe = SinusoidalPositionEncoding(d_model=512)
    x = torch.randn(2, 100, 512)
    output = pe(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
```

### 💡 面试追问

**Q: 为什么用 sin/cos 而不是其他函数？**

> sin/cos 有一个重要性质：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。这让模型更容易学习相对位置关系。

**Q: 为什么是 10000？**

> 这是一个经验值。它决定了频率范围：最高频率 $1$，最低频率 $1/10000$。

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rope](rope.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：RoPE 施加到哪些张量？**

答：旋转 Q、K 而非 V；点积编码相对位置。

**问：位置偏移如何处理？**

答：增量解码时用全局位置，不可每步重新从零计数。

[返回本章目录](README.md)
