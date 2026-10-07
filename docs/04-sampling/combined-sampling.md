## 组合采样策略

### 📝 工业级实现

```python
class Sampler:
    """
    组合采样器：支持 Temperature + Top-k + Top-p + 重复惩罚
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int = 0,         # 0 表示不使用
        top_p: float = 1.0,     # 1.0 表示不使用
        repetition_penalty: float = 1.0,  # 1.0 表示不惩罚
        min_tokens_to_keep: int = 1
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep
    
    def __call__(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, vocab_size] 或 [vocab_size]
            generated_tokens: [batch, seq_len] 已生成的 token（用于重复惩罚）
        """
        # Step 1: 重复惩罚
        if self.repetition_penalty != 1.0 and generated_tokens is not None:
            logits = self._apply_repetition_penalty(logits, generated_tokens)
        
        # Step 2: Temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Step 3: Top-k
        if self.top_k > 0:
            logits = self._top_k_filtering(logits)
        
        # Step 4: Top-p
        if self.top_p < 1.0:
            logits = self._top_p_filtering(logits)
        
        # Step 5: 采样
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        重复惩罚：降低已生成 token 的概率
        """
        for batch_idx in range(logits.size(0) if logits.dim() > 1 else 1):
            for token_id in generated_tokens[batch_idx].unique():
                if logits.dim() > 1:
                    score = logits[batch_idx, token_id]
                else:
                    score = logits[token_id]
                
                # 正分数除以惩罚，负分数乘以惩罚
                if score > 0:
                    logits[batch_idx, token_id] = score / self.repetition_penalty
                else:
                    logits[batch_idx, token_id] = score * self.repetition_penalty
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        k = max(self.top_k, self.min_tokens_to_keep)
        threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < threshold, float('-inf'))
    
    def _top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumsum_probs > self.top_p
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float('-inf'))


# ==================== 常用配置 ====================
SAMPLING_CONFIGS = {
    "greedy": {"temperature": 0.0},
    "creative": {"temperature": 1.0, "top_k": 50, "top_p": 0.95},
    "balanced": {"temperature": 0.7, "top_p": 0.9},
    "code": {"temperature": 0.2, "top_p": 0.95, "repetition_penalty": 1.1},
    "chat": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.05},
}
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
