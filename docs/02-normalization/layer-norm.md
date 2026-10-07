## Layer Normalization

### 🎯 核心思想

对**每个样本的特征维度**进行归一化，使其均值为 0，方差为 1，然后通过可学习参数进行缩放和平移。

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$（特征维度的均值）
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$（特征维度的方差）
- $\gamma, \beta$ 是可学习参数

### 📝 实现代码

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Layer Normalization 手动实现
    
    对最后 normalized_shape 维度进行归一化
    例如: 输入 [batch, seq, hidden]，normalized_shape=hidden
          则对每个 [hidden] 向量归一化
    """
    
    def __init__(
        self, 
        normalized_shape: int | tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True  # 是否使用可学习参数
    ):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # 可学习的缩放和平移参数
            self.weight = nn.Parameter(torch.ones(normalized_shape))   # gamma
            self.bias = nn.Parameter(torch.zeros(normalized_shape))    # beta
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., *normalized_shape]
        
        Returns:
            归一化后的张量，形状不变
        """
        # 确定归一化的维度
        # 如果 normalized_shape = (hidden,)，则 dims = (-1,)
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Step 1: 计算均值
        mean = x.mean(dim=dims, keepdim=True)
        
        # Step 2: 计算方差（使用无偏估计 unbiased=False 与 PyTorch 保持一致）
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Step 3: 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Step 4: 缩放和平移
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        return x_normalized


# ==================== 验证代码 ====================
def test_layer_norm():
    batch, seq, hidden = 2, 10, 768
    
    # 我们的实现
    my_ln = LayerNorm(hidden)
    # PyTorch 官方实现
    torch_ln = nn.LayerNorm(hidden)
    
    # 同步参数
    torch_ln.weight.data = my_ln.weight.data.clone()
    torch_ln.bias.data = my_ln.bias.data.clone()
    
    x = torch.randn(batch, seq, hidden)
    
    my_output = my_ln(x)
    torch_output = torch_ln(x)
    
    diff = (my_output - torch_output).abs().max()
    print(f"最大误差: {diff:.8f}")
    assert diff < 1e-5, "实现有误!"
    print("✅ LayerNorm 测试通过!")


if __name__ == "__main__":
    test_layer_norm()
```

### 🔍 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|:---|:---:|:---:|
| 计算均值 | O(d) | O(1) |
| 计算方差 | O(d) | O(1) |
| 归一化 | O(d) | O(d) |
| **总计** | **O(d)** | **O(d)** |

> 🤔 **Q: 为什么 LayerNorm 用 `unbiased=False`？**
>
> PyTorch 的 `var()` 默认用无偏估计（除以 n-1），但 LayerNorm 的原论文用的是有偏估计（除以 n）。
>
> 要与 PyTorch 官方 `nn.LayerNorm` 结果一致，必须用 `unbiased=False`！

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rms-norm](rms-norm.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：LN 与 RMSNorm 有何差别？**

答：RMSNorm 不减均值，只除均方根；均沿特征维归一化。

**问：为什么小 batch 不利于 BN？**

答：统计量由 batch 估计，样本少时方差较大。

[返回本章目录](README.md)
