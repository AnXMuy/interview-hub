## Learnable Position Encoding

### 🎯 核心思想

直接学习每个位置的 embedding，简单粗暴但有效。BERT、GPT-2 等使用。

### 📝 实现代码

```python
class LearnablePositionEncoding(nn.Module):
    """
    可学习位置编码
    
    BERT, GPT-2 等使用
    简单直接，但无法外推到训练时未见过的位置
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 位置 embedding 表
        self.pe = nn.Embedding(max_len, d_model)
        
        # 初始化
        nn.init.normal_(self.pe.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        # [seq_len] -> [seq_len, d_model] -> [1, seq_len, d_model]
        position_embeddings = self.pe(positions).unsqueeze(0)
        
        return self.dropout(x + position_embeddings)
```

**局限性**：无法处理超过 `max_len` 的序列。

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rope](rope.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：RoPE 施加到哪些张量？**

答：旋转 Q、K 而非 V；点积编码相对位置。

**问：位置偏移如何处理？**

答：增量解码时用全局位置，不可每步重新从零计数。

[返回本章目录](README.md)
