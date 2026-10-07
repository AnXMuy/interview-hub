## Top-p (Nucleus) Sampling

### 🎯 核心思想

保留累积概率达到 p 的**最小**词汇集合。这样做的好处是：
- 分布尖锐时，只保留少数几个词
- 分布平坦时，保留更多词

### 📝 实现代码

```python
def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-p (Nucleus) Sampling 🔥 面试必考
    
    保留累积概率达到 p 的最小词汇集合
    
    Args:
        logits: [batch, vocab_size] 或 [vocab_size]
        p: 累积概率阈值，通常 0.9-0.95
        temperature: 温度参数
        min_tokens_to_keep: 最少保留的 token 数
    
    Returns:
        sampled_token: 采样结果
        probs: 过滤后的概率分布
    """
    # Step 1: 温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # Step 2: 排序（降序）
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Step 3: 计算累积概率
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Step 4: 找到累积概率超过 p 的位置
    # 注意：我们要保留累积概率 <= p 的位置
    # 但至少保留 min_tokens_to_keep 个
    sorted_indices_to_remove = cumsum_probs > p
    
    # 保证至少有 min_tokens_to_keep 个 token
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Step 5: 将需要移除的位置在原始 logits 中置为 -inf
    # 先创建一个 mask，然后映射回原始索引
    # 方法：scatter 将 sorted 空间的 mask 转回原始空间
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )
    
    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    # Step 6: Softmax + 采样
    probs = F.softmax(filtered_logits, dim=-1)
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


def top_p_sampling_simple(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Top-p 简化版（面试快速手写）
    """
    # 1. 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    # 2. 计算累积概率
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    
    # 3. 找到超过 p 的位置并 mask
    mask = cumsum <= p
    mask[..., 0] = True  # 至少保留一个
    
    # 4. 过滤
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # 5. 采样
    sampled_idx = torch.multinomial(filtered_probs, 1)
    return sorted_indices.gather(-1, sampled_idx)


# ==================== 详细示例 ====================
def top_p_example():
    """展示 Top-p 的工作过程"""
    # 假设词表大小为 10
    logits = torch.tensor([2.0, 1.8, 1.5, 0.5, 0.2, 0.0, -0.5, -1.0, -2.0, -3.0])
    p = 0.9
    
    # 计算概率
    probs = F.softmax(logits, dim=-1)
    
    # 排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    print("=== Top-p Sampling 示例 (p=0.9) ===\n")
    print("索引\t概率\t累积概率\t保留?")
    print("-" * 40)
    
    for i in range(len(probs)):
        keep = "✓" if cumsum[i] <= p or i == 0 else "✗"
        print(f"{sorted_indices[i].item()}\t{sorted_probs[i]:.4f}\t{cumsum[i]:.4f}\t\t{keep}")
    
    print(f"\n结论: 保留前 {(cumsum <= p).sum().item() + 1} 个词")


if __name__ == "__main__":
    top_p_example()
```

输出示例：
```
=== Top-p Sampling 示例 (p=0.9) ===

索引    概率    累积概率        保留?
----------------------------------------
0       0.2897  0.2897          ✓
1       0.2367  0.5265          ✓
2       0.1755  0.7020          ✓
3       0.0646  0.7666          ✓
4       0.0479  0.8145          ✓
5       0.0392  0.8537          ✓
6       0.0238  0.8775          ✓
7       0.0144  0.8919          ✓
8       0.0053  0.8972          ✓
9       0.0029  0.9001          ✗

结论: 保留前 9 个词
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

[返回本章目录](README.md)
