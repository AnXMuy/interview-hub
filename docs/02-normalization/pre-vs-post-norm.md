## Pre-Norm vs Post-Norm

### 🎯 核心区别

> 💡 **记忆技巧**：Pre-Norm “先洗澡再进门”，Post-Norm “进门后再洗澡”

```
Post-Norm (原始 Transformer):
x = LayerNorm(x + Sublayer(x))   # 注意: Norm 在最外层

Pre-Norm (现代 LLM):
x = x + Sublayer(LayerNorm(x))   # 注意: Norm 在最里层，残差在外面
```

### 📝 实现对比

```python
import torch
from torch import nn

class PostNormBlock(nn.Module):
    """Post-Norm: 原始 Transformer 架构"""
    
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先计算残差，再归一化
        return self.norm(x + self.sublayer(x))


class PreNormBlock(nn.Module):
    """Pre-Norm: 现代 LLM 架构（GPT-2+, LLaMA 等）"""
    
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先归一化，再计算残差
        return x + self.sublayer(self.norm(x))


# Pre-Norm 的梯度流分析
def analyze_gradient_flow():
    """
    Pre-Norm 的优势：梯度可以直接通过残差连接回传
    
    Post-Norm: grad 需要经过 LayerNorm
    ∂L/∂x = ∂L/∂y · ∂LayerNorm/∂(x + sublayer(x))
    
    Pre-Norm: grad 有直接的恒等路径
    ∂L/∂x = ∂L/∂y · (1 + ∂sublayer/∂LayerNorm · ∂LayerNorm/∂x)
    
    第一项的 1 保证了梯度能直接回传，避免梯度消失
    """
    print("""
    Pre-Norm 优势:
    1. 梯度直通路径，训练更稳定
    2. 不需要 warmup 或特殊学习率调度
    3. 可以训练更深的模型
    
    Pre-Norm 的劣势:
    1. 最终输出前需要额外的 LayerNorm
    2. 理论上表达能力略弱（但实践中可忽略）
    """)
```

> 🤔 **Q: 为什么 Pre-Norm 梯度更稳定？看这个图：**
>
> ```
> Post-Norm: x ──▶ Sublayer ──▶ + ──▶ LayerNorm ──▶ output
>            └─────────────┘
>            梯度必须经过 LayerNorm，可能被压缩
>
> Pre-Norm:  x ──▶ LayerNorm ──▶ Sublayer ──▶ + ──▶ output
>            └────────────────────┘
>            梯度可以通过残差连接直接回传，永远不会消失！
> ```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rms-norm](rms-norm.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：LN 与 RMSNorm 有何差别？**

答：RMSNorm 不减均值，只除均方根；均沿特征维归一化。

**问：为什么小 batch 不利于 BN？**

答：统计量由 batch 估计，样本少时方差较大。

[返回本章目录](README.md)
