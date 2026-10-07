## Cross Entropy Loss

### 🎯 核心思想

交叉熵衡量两个概率分布的差异。对于分类任务：

$$\text{CE}(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)$$

当 $y$ 是 one-hot 编码时（真实类别为 $c$）：

$$\text{CE} = -\log(\hat{y}_c) = -\log\left(\frac{e^{z_c}}{\sum_j e^{z_j}}\right)$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean',
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Cross Entropy Loss 手动实现（数值稳定版）
    
    Args:
        logits: [batch, num_classes] 模型原始输出（未经 softmax）
        targets: [batch] 真实类别索引
        reduction: 'mean', 'sum', 'none'
        ignore_index: 忽略的标签（如 padding）
    
    Returns:
        loss: 标量或 [batch]（取决于 reduction）
    """
    batch_size, num_classes = logits.shape
    
    # ========== 数值稳定性处理 ==========
    # 直接计算 softmax 会导致 exp(logits) 溢出
    # 技巧: softmax(x) = softmax(x - max(x))
    # 这样最大的 logit 变成 0，exp(0) = 1，不会溢出
    
    # Step 1: 减去最大值（数值稳定）
    logits_max = logits.max(dim=-1, keepdim=True)[0]
    logits_stable = logits - logits_max  # 最大值变成 0
    
    # Step 2: 计算 log_softmax
    # log_softmax(x) = x - log(sum(exp(x)))
    # 但 log(sum(exp(x))) 可以用 logsumexp 更稳定地计算
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
    log_softmax = logits_stable - log_sum_exp  # [batch, num_classes]
    
    # Step 3: 取出真实类别的 log 概率（负对数似然）
    # gather: 从 log_softmax 中按 targets 索引取值
    nll = -log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # nll: [batch]
    
    # Step 4: 处理 ignore_index
    if ignore_index is not None:
        mask = targets != ignore_index
        nll = nll * mask.float()
    
    # Step 5: Reduction
    if reduction == 'none':
        return nll
    elif reduction == 'sum':
        return nll.sum()
    elif reduction == 'mean':
        if ignore_index is not None:
            return nll.sum() / mask.sum().clamp(min=1)
        return nll.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def cross_entropy_simple(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy 极简版（面试快速手写）
    """
    # 方法1: 分解为 log_softmax + nll_loss
    log_probs = F.log_softmax(logits, dim=-1)  # 内部已做数值稳定
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()
    
    # 方法2: 更简洁
    # return F.cross_entropy(logits, targets)


# ==================== 验证代码 ====================
def test_cross_entropy():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    
    my_loss = cross_entropy_loss(logits, targets)
    torch_loss = F.cross_entropy(logits, targets)
    
    print(f"我的实现: {my_loss:.6f}")
    print(f"PyTorch:  {torch_loss:.6f}")
    print(f"误差:     {abs(my_loss - torch_loss):.8f}")
    
    assert abs(my_loss - torch_loss) < 1e-5
    print("✅ Cross Entropy 测试通过!")


if __name__ == "__main__":
    test_cross_entropy()
```

> 🤔 **Q: 为什么用 `gather` 而不是直接索引 `log_softmax[targets]`？**
>
> 因为需要批量处理！`log_softmax[targets]` 对 2D tensor 不行。
>
> `gather(dim=-1, index=targets.unsqueeze(-1))` 正确地从每个样本的分布中取出目标类别的概率。

---

## 动手练习

本题的单接口练习见 [练习契约](../../grader/README.md)；仅编辑 `solutions/loss_functions.py`，运行 `python -m grader 05-loss-functions`。原文代码为学习示例，测评以练习契约为准。

## 常见面试问答

**问：为什么直接算 log(softmax) 不稳定？**

答：exp 可能溢出或下溢；用 logsumexp/log_softmax。

**问：padding token 如何处理？**

答：用 ignore_index 排除且平均值只除有效 token 数。

[返回本章目录](README.md)
