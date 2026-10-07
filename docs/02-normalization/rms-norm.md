## RMS Normalization

### 🎯 核心思想

RMSNorm 是 LayerNorm 的简化版本：**去掉均值中心化**，只做方差归一化。

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x) + \epsilon}$$

其中：
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

**优势**：
- 计算更快（少一次 mean 计算）
- 实验表明在 LLM 中效果与 LayerNorm 相当
- LLaMA、Mistral、DeepSeek 等都使用 RMSNorm

### 📝 实现代码

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    LLaMA, Mistral, DeepSeek 等现代 LLM 使用
    相比 LayerNorm 去掉了 mean centering，更高效
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数，没有偏移参数
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim]
        
        Returns:
            归一化后的张量
        """
        # 计算 RMS: sqrt(mean(x^2))
        # 等价于 x 的 L2 范数除以 sqrt(dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        return x / rms * self.weight
    
    def forward_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        优化版本：使用 rsqrt 避免两次开方
        """
        # rsqrt = 1 / sqrt(x)，比 sqrt 后再除法更快
        rms_inv = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * rms_inv * self.weight


class RMSNormWithCast(nn.Module):
    """
    带类型转换的 RMSNorm（处理混合精度训练）
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        
        # 在 float32 下计算，数值更稳定
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # 转回原始类型
        return (self.weight * x).to(input_dtype)


# ==================== 对比测试 ====================
def compare_norms():
    """对比 LayerNorm 和 RMSNorm"""
    import time
    
    dim = 4096
    batch, seq = 32, 512
    x = torch.randn(batch, seq, dim).cuda()
    
    ln = LayerNorm(dim).cuda()
    rms = RMSNorm(dim).cuda()
    
    # 预热
    for _ in range(10):
        _ = ln(x)
        _ = rms(x)
    
    torch.cuda.synchronize()
    
    # LayerNorm 计时
    start = time.perf_counter()
    for _ in range(100):
        _ = ln(x)
    torch.cuda.synchronize()
    ln_time = time.perf_counter() - start
    
    # RMSNorm 计时
    start = time.perf_counter()
    for _ in range(100):
        _ = rms(x)
    torch.cuda.synchronize()
    rms_time = time.perf_counter() - start
    
    print(f"LayerNorm: {ln_time*1000:.2f} ms")
    print(f"RMSNorm:   {rms_time*1000:.2f} ms")
    print(f"加速比:    {ln_time/rms_time:.2f}x")


if __name__ == "__main__":
    test_layer_norm()
```

### 💡 面试追问

**Q: 为什么 LLM 用 RMSNorm 而不是 LayerNorm？**

> 1. **更快**：省掉 mean 计算，约快 10-15%
> 2. **效果相当**：实验表明在 LLM 场景下，mean centering 贡献不大
> 3. **参数更少**：没有 bias 参数

**Q: RMSNorm 的数学直觉是什么？**

> RMSNorm 可以理解为将向量投影到单位球面上再缩放。它保持向量方向，只调整模长。

---

## 动手练习

本题的单接口练习见 [练习契约](../../grader/README.md)；仅编辑 `solutions/normalization.py`，运行 `python -m grader 02-normalization`。原文代码为学习示例，测评以练习契约为准。

## 常见面试问答

**问：LN 与 RMSNorm 有何差别？**

答：RMSNorm 不减均值，只除均方根；均沿特征维归一化。

**问：为什么小 batch 不利于 BN？**

答：统计量由 batch 估计，样本少时方差较大。

[返回本章目录](README.md)
