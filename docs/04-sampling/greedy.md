## Greedy Decoding

### 🎯 核心思想

每一步选择概率最高的 token。简单但容易陷入重复。

### 📝 实现代码

```python
def greedy_decoding(logits: torch.Tensor) -> torch.Tensor:
    """
    贪婪解码：选择概率最高的 token
    
    Args:
        logits: [batch, vocab_size]
    
    Returns:
        token_ids: [batch]
    """
    return logits.argmax(dim=-1)


def greedy_decode_sequence(
    model,
    input_ids: torch.Tensor,
    max_length: int = 50,
    eos_token_id: int = 2
) -> torch.Tensor:
    """
    贪婪解码完整序列
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
        
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if (next_token == eos_token_id).all():
            break
    
    return generated
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

[返回本章目录](README.md)
