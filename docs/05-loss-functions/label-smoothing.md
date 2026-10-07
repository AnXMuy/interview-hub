## Label Smoothing

### 🎯 核心思想

将 hard label（one-hot）软化，防止模型过于自信。

$$y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{hard}} + \frac{\epsilon}{K}$$

### 📝 实现代码

```python
def label_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Label Smoothing Cross Entropy
    
    Args:
        logits: [batch, num_classes]
        targets: [batch] hard labels
        smoothing: 平滑系数（通常 0.1）
    """
    num_classes = logits.size(-1)
    
    # 创建 soft labels
    # confidence = 1 - smoothing (真实类别)
    # smoothing / num_classes (其他类别)
    confidence = 1.0 - smoothing
    smooth_value = smoothing / num_classes
    
    # 创建 soft target
    soft_targets = torch.full_like(logits, smooth_value)
    soft_targets.scatter_(
        dim=-1,
        index=targets.unsqueeze(-1),
        value=confidence + smooth_value
    )
    
    # KL divergence (等价于 cross entropy with soft targets)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    
    return loss.mean()
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
