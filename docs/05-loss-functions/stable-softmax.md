## Softmax 数值稳定性

### 🎯 核心问题

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**问题**：当 $x_i$ 很大（如 100）时，$e^{100}$ 会溢出（float32 最大约 $3.4 \times 10^{38}$）

> 🤔 **Q: 为什么减去 max 就能解决溢出问题？**
>
> 数学上：`softmax(x) = softmax(x - c)`，任意常数 c 不改变结果。
>
> 选 `c = max(x)` 后，最大的 `x_i - c = 0`，`exp(0) = 1`，不会溢出！
>
> 其他的 `x_j - c < 0`，`exp(负数) < 1`，也不会溢出。

### 📝 实现代码

```python
def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """
    朴素 Softmax（不稳定，仅作对比）
    """
    exp_x = torch.exp(x)  # 可能溢出！
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def softmax_stable(x: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 Softmax
    
    技巧: softmax(x) = softmax(x - c)，任意常数 c
    选择 c = max(x)，使得最大的 exp 参数为 0
    """
    # Step 1: 减去最大值
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Step 2: 安全的 exp
    exp_x = torch.exp(x_shifted)  # 最大值的 exp 是 1，不会溢出
    
    # Step 3: 归一化
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def log_softmax_stable(x: torch.Tensor) -> torch.Tensor:
    """
    数值稳定的 Log-Softmax
    
    log_softmax(x) = x - log(sum(exp(x)))
                   = x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = x.max(dim=-1, keepdim=True)[0]
    x_shifted = x - x_max
    
    log_sum_exp = torch.log(torch.exp(x_shifted).sum(dim=-1, keepdim=True))
    
    return x_shifted - log_sum_exp


# ==================== 测试溢出情况 ====================
def test_overflow():
    # 大数值测试
    x_large = torch.tensor([100.0, 200.0, 300.0])
    
    print("=== 溢出测试 ===")
    print(f"输入: {x_large}")
    
    try:
        naive_result = softmax_naive(x_large)
        print(f"朴素 Softmax: {naive_result}")  # 会有 inf/nan
    except Exception as e:
        print(f"朴素 Softmax 失败: {e}")
    
    stable_result = softmax_stable(x_large)
    print(f"稳定 Softmax: {stable_result}")  # 正确结果
    
    pytorch_result = F.softmax(x_large, dim=-1)
    print(f"PyTorch:      {pytorch_result}")


if __name__ == "__main__":
    test_overflow()
```

输出：
```
=== 溢出测试 ===
输入: tensor([100., 200., 300.])
朴素 Softmax: tensor([0., 0., nan])
稳定 Softmax: tensor([0.0000e+00, 0.0000e+00, 1.0000e+00])
PyTorch:      tensor([0.0000e+00, 0.0000e+00, 1.0000e+00])
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为什么直接算 log(softmax) 不稳定？**

答：exp 可能溢出或下溢；用 logsumexp/log_softmax。

**问：padding token 如何处理？**

答：用 ignore_index 排除且平均值只除有效 token 数。

[返回本章目录](README.md)
