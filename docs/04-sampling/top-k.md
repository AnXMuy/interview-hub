## Top-k Sampling

### 🎯 核心思想

只从概率最高的 k 个 token 中采样，过滤掉长尾低概率词。

### 📝 实现代码

```python
def top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k Sampling
    
    只保留概率最高的 k 个 token，其余置为 -inf
    
    Args:
        logits: [batch, vocab_size] 模型输出
        k: 保留的 token 数量
        temperature: 温度参数
        min_tokens_to_keep: 最少保留的 token 数
    
    Returns:
        sampled_token: 采样结果
        probs: 过滤后的概率分布
    """
    # 确保 k 至少为 min_tokens_to_keep
    k = max(k, min_tokens_to_keep)
    
    # Step 1: 温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # Step 2: 找到 top-k 的阈值
    # topk 返回 (values, indices)，取 values 的最后一个（第 k 大）
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1, None]  # 第 k 大的值
    
    # Step 3: 将小于阈值的 logits 置为 -inf
    # 这样 softmax 后这些位置的概率趋近于 0
    filtered_logits = torch.where(
        logits >= threshold,
        logits,
        torch.full_like(logits, float('-inf'))
    )
    
    # Step 4: Softmax + 采样
    probs = F.softmax(filtered_logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


def top_k_sampling_simple(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-k 简化版（面试快速手写）
    """
    # 找到第 k 大的值作为阈值
    threshold = torch.topk(logits, k).values[..., -1:]
    
    # 低于阈值的置为 -inf
    logits[logits < threshold] = float('-inf')
    
    # Softmax + 采样
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 💡 Top-k 的问题

```
场景1: 分布很尖（高置信度）
  词: [the, a, an, ...]
  概率: [0.9, 0.05, 0.02, ...]
  → k=50 会引入很多不相关的词

场景2: 分布很平（低置信度）
  词: [red, blue, green, yellow, ...]
  概率: [0.1, 0.09, 0.08, 0.08, ...]
  → k=5 可能漏掉合理的选项

结论: 固定的 k 无法适应不同的分布形状
解决: 使用 Top-p（动态调整候选数量）
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [top-p](top-p.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

[返回本章目录](README.md)
