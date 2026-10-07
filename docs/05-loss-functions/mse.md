## MSE Loss

### 🎯 核心思想

均方误差，用于回归任务：

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### 📝 实现代码

```python
def mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Mean Squared Error Loss
    
    Args:
        predictions: 预测值
        targets: 目标值
        reduction: 'mean', 'sum', 'none'
    """
    squared_diff = (predictions - targets) ** 2
    
    if reduction == 'none':
        return squared_diff
    elif reduction == 'sum':
        return squared_diff.sum()
    elif reduction == 'mean':
        return squared_diff.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def rmse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error"""
    return torch.sqrt(mse_loss(predictions, targets, reduction='mean'))


def mae_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (L1 Loss)"""
    return torch.abs(predictions - targets).mean()


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0
) -> torch.Tensor:
    """
    Huber Loss: MSE 和 MAE 的平滑组合
    
    |error| < delta: MSE（平滑）
    |error| >= delta: MAE（抗离群点）
    """
    error = predictions - targets
    abs_error = torch.abs(error)
    
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    
    return (0.5 * quadratic ** 2 + delta * linear).mean()
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
