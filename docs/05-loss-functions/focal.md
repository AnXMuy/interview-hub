## Focal Loss

### 🎯 核心思想

解决类别不平衡问题：降低易分类样本的权重，关注难分类样本。

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- $(1 - p_t)^\gamma$：调制因子，$p_t$ 接近 1 时权重降低
- $\gamma$：聚焦参数，通常取 2

### 📝 实现代码

```python
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Focal Loss for Dense Object Detection
    
    Args:
        logits: [batch, num_classes] 模型输出
        targets: [batch] 真实标签
        alpha: 平衡正负样本的权重
        gamma: 聚焦参数（gamma=0 时退化为 CE）
        reduction: 'mean', 'sum', 'none'
    """
    # Step 1: 计算 CE
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Step 2: 计算 pt（正确类别的预测概率）
    probs = F.softmax(logits, dim=-1)
    pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Step 3: 计算 focal weight
    # (1 - pt)^gamma: 易分类样本 pt->1，权重->0
    focal_weight = (1 - pt) ** gamma
    
    # Step 4: 应用权重
    focal_loss = focal_weight * ce_loss
    
    # Step 5: 类别平衡权重（可选）
    if alpha is not None:
        # 简化处理：正类用 alpha，负类用 1-alpha
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(alpha, device=logits.device),
            torch.tensor(1 - alpha, device=logits.device)
        )
        focal_loss = alpha_t * focal_loss
    
    # Step 6: Reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    return focal_loss


def focal_loss_multiclass(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = None,  # [num_classes]
    gamma: float = 2.0
) -> torch.Tensor:
    """
    多分类 Focal Loss
    
    alpha: 每个类别的权重，可用于处理类别不平衡
    """
    num_classes = logits.size(-1)
    
    # CE loss
    ce = F.cross_entropy(logits, targets, reduction='none')
    
    # pt
    probs = F.softmax(logits, dim=-1)
    pt = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Focal weight
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weight
    if alpha is not None:
        alpha_t = alpha[targets]
        return (alpha_t * focal_weight * ce).mean()
    
    return (focal_weight * ce).mean()
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
