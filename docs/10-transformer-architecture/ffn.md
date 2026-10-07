## Feed-Forward Network

### 🎯 核心思想

FFN 是 Transformer 的"记忆库"，存储知识。通常是两层 MLP：

$$\text{FFN}(x) = \text{Act}(xW_1 + b_1)W_2 + b_2$$

**维度变化**：d_model → d_ff → d_model（通常 d_ff = 4 × d_model）

> 🤔 **Q: 为什么 FFN 要先升维再降维？不是浪费计算吗？**
>
> 升维是为了表达能力！线性变换 + 非线性激活在高维空间更容易分离特征。
>
> 这也是为什么研究表明 FFN 是"记忆库"——它存储了大量知识。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    标准 FFN (GPT-2 风格)
    
    d_model -> d_ff -> d_model
    激活函数: GELU
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        x = self.w1(x)           # [batch, seq, d_ff]
        x = self.activation(x)    # 非线性
        x = self.dropout(x)
        x = self.w2(x)           # [batch, seq, d_model]
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU FFN (LLaMA 风格)
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 使用
    
    SwiGLU(x) = (xW1 * SiLU(xW_gate)) @ W2
    
    参数量: 3 * d_model * d_ff（比标准 FFN 多 50%）
    但效果更好，通常用更小的 d_ff 来补偿
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # LLaMA 的 d_ff 计算: 2/3 * 4 * d_model，然后取 256 的倍数
        d_ff = d_ff or int(2 * 4 * d_model / 3)
        d_ff = ((d_ff + 255) // 256) * 256  # 对齐到 256
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # Gate 投影
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ W_gate) * (x @ W1) @ W2
        gate = F.silu(self.w_gate(x))  # SiLU = x * sigmoid(x)
        x = self.w1(x) * gate          # 门控
        x = self.dropout(x)
        x = self.w2(x)
        return x


class GeGLU(nn.Module):
    """
    GeGLU FFN (另一种门控变体)
    
    GeGLU(x) = GELU(xW_gate) * (xW1) @ W2
    """
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 合并 W1 和 W_gate 为一个大矩阵
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.gelu(gate) * up
        x = self.w_down(x)
        return x
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
