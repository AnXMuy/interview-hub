## KL Divergence

### 🎯 核心思想

KL 散度衡量两个概率分布的差异：

$$D_{KL}(P \| Q) = \sum_i P(i) \log\frac{P(i)}{Q(i)}$$

**注意**：KL 散度不对称！$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

### 📝 实现代码

```python
def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    KL Divergence: D_KL(P || Q)
    
    Args:
        p: [batch, num_classes] 目标分布（概率）
        q: [batch, num_classes] 预测分布（概率）
        reduction: 'batchmean', 'sum', 'none'
    """
    eps = 1e-10
    
    # 确保概率有效
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    
    # KL = sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
    kl = p * (torch.log(p) - torch.log(q))
    kl = kl.sum(dim=-1)  # 对类别求和
    
    if reduction == 'batchmean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl


def kl_divergence_from_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    从 logits 计算 KL 散度（更稳定）
    
    用 log_softmax 避免 softmax 后再 log 的精度损失
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    
    p = torch.exp(log_p)
    
    # KL = sum(p * (log_p - log_q))
    kl = (p * (log_p - log_q)).sum(dim=-1)
    
    if reduction == 'batchmean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    return kl


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon Divergence（对称版 KL）
    
    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    其中 M = (P + Q) / 2
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, 'batchmean') + 0.5 * kl_divergence(q, m, 'batchmean')


# ==================== RL 中的 KL 惩罚 ====================
def kl_penalty_for_rl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor
) -> torch.Tensor:
    """
    RLHF/PPO 中的 KL 惩罚项
    
    用于防止 policy 偏离 reference model 太远
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    
    policy_probs = torch.exp(policy_log_probs)
    
    # D_KL(policy || ref)
    kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
    
    return kl.mean()
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
