## LoRA

### 🎯 核心思想

LoRA (Low-Rank Adaptation)：冻结预训练权重，只训练低秩分解的增量矩阵。

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

**参数量**：原始 $d \times k$ → LoRA $(d + k) \times r$

> 🤔 **Q: 为什么 B 初始化为 0，A 用 Kaiming？**
>
> 关键洞察：初始时 BA = 0，模型输出 = W + BA = W！
>
> 这样微调从**原模型出发**，而不是随机点。
>
> A 用 Kaiming 是保证梯度能流动（如果 A 也是 0，梯度为 0）。

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA Layer 🔥 面试必考
    
    核心思想：
    - 冻结原始权重 W
    - 学习低秩增量 ΔW = BA
    - 输出 = Wx + BAx = Wx + B(Ax)
    
    参数量从 d*k 降低到 (d+k)*r
    当 r=8, d=k=4096 时，参数量降低 99.6%
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        
        # 缩放因子：控制 LoRA 更新的幅度
        # alpha/rank 类似于学习率的作用
        self.scaling = lora_alpha / rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight.requires_grad = False  # 冻结
        
        # LoRA 可训练参数
        # A: [rank, in_features] - 降维
        # B: [out_features, rank] - 升维
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        初始化策略：
        - A: Kaiming 均匀初始化
        - B: 零初始化
        
        这样初始时 BA = 0，从原模型开始微调
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, in_features]
        """
        # 原始输出（冻结权重）
        original_output = F.linear(x, self.weight)
        
        # LoRA 输出
        # x @ A^T @ B^T * scaling
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # [batch, seq, rank]
        lora_output = F.linear(lora_output, self.lora_B)  # [batch, seq, out_features]
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        合并 LoRA 权重到原始权重（推理时使用）
        
        W' = W + scaling * BA
        """
        with torch.no_grad():
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
    
    def unmerge_weights(self):
        """取消合并（继续训练时使用）"""
        with torch.no_grad():
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)


class LoRALinear(nn.Module):
    """
    完整的 LoRA Linear 层（包含 bias）
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        
        self.original = original_linear
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # 冻结原始参数
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRA 参数
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_out = self.original(x)
        
        # LoRA 输出
        lora_out = self.dropout(x)
        lora_out = F.linear(F.linear(lora_out, self.lora_A), self.lora_B)
        
        return original_out + lora_out * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    lora_alpha: float = 16.0,
    target_modules: list = None
):
    """
    将 LoRA 应用到模型的指定层
    
    target_modules: 要替换的模块名称（如 ["q_proj", "v_proj"]）
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # 默认只对 Q, V 投影应用
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 创建 LoRA 层
                lora_layer = LoRALinear(module, rank=rank, lora_alpha=lora_alpha)
                
                # 替换原始层
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, lora_layer)
    
    return model


# ==================== 参数量分析 ====================
def analyze_lora_params(d: int, k: int, rank: int):
    """分析 LoRA 参数量"""
    original_params = d * k
    lora_params = (d + k) * rank
    ratio = lora_params / original_params * 100
    
    print(f"=== LoRA 参数分析 ===")
    print(f"原始参数: {d} x {k} = {original_params:,}")
    print(f"LoRA 参数 (rank={rank}): ({d} + {k}) x {rank} = {lora_params:,}")
    print(f"参数比例: {ratio:.2f}%")
    print(f"参数减少: {100 - ratio:.2f}%")


if __name__ == "__main__":
    # LLaMA-7B 的典型配置
    analyze_lora_params(d=4096, k=4096, rank=8)
```

### 💡 面试追问

**Q: 为什么 B 初始化为 0？**

> 保证训练开始时 ΔW = BA = 0，即从原模型开始。如果不是零初始化，初始就会偏离预训练模型。

**Q: 为什么只对 Q, V 应用 LoRA，不对 K 应用？**

> 实验发现对 Q, V 效果最好，对 K 效果较差。原因可能是 K 的变化会影响所有 Q 的 attention pattern。

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：LoRA 为什么能省参数？**

答：冻结原权重，只训练低秩矩阵 A、B。

**问：梯度累积要注意什么？**

答：每个 microbatch 的 loss 按累积步数缩放，最后才更新和清梯度。

[返回本章目录](README.md)
