## SFT Loss

### 🎯 核心思想

Supervised Fine-Tuning Loss：LLM 的标准训练损失，本质是 **next token prediction**。

### 📝 实现代码

```python
def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    SFT Loss: Language Model 的标准训练损失
    
    Args:
        logits: [batch, seq_len, vocab_size] 模型输出
        labels: [batch, seq_len] 目标 token（通常是 input_ids 左移一位）
        ignore_index: 忽略的标签（如 padding、prompt 部分）
    
    Returns:
        loss: 平均 loss
    """
    # 重塑为 2D
    # logits: [batch * seq_len, vocab_size]
    # labels: [batch * seq_len]
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Cross Entropy（内部处理 ignore_index）
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


def sft_loss_with_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor
) -> torch.Tensor:
    """
    带 mask 的 SFT Loss
    
    loss_mask: [batch, seq_len] 1 表示计算 loss，0 表示忽略
    
    常见用法：
    - 只对 response 部分计算 loss（忽略 prompt）
    - 忽略 padding
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Shift: labels 应该是 logits 预测的下一个 token
    # logits[:, :-1] 预测 labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()
    
    # 计算每个 token 的 loss
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='none'
    ).view(batch_size, -1)
    
    # 应用 mask
    masked_loss = loss_per_token * shift_mask
    
    # 平均（只除以有效 token 数）
    return masked_loss.sum() / shift_mask.sum().clamp(min=1)


# ==================== 使用示例 ====================
def sft_example():
    """SFT 训练示例"""
    batch_size, seq_len, vocab_size = 2, 10, 1000
    
    # 模拟模型输出
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 模拟标签（通常是 input_ids 本身，模型预测下一个 token）
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 模拟 mask（前 3 个是 prompt，不计算 loss）
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[:, :3] = 0  # prompt 部分
    
    loss = sft_loss_with_mask(logits, labels, loss_mask)
    print(f"SFT Loss: {loss:.4f}")
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [cross-entropy](cross-entropy.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为什么直接算 log(softmax) 不稳定？**

答：exp 可能溢出或下溢；用 logsumexp/log_softmax。

**问：padding token 如何处理？**

答：用 ignore_index 排除且平均值只除有效 token 数。

[返回本章目录](README.md)
