## 完整 Transformer 实现

### 📝 GPT-style LLM

```python
class GPTModel(nn.Module):
    """
    完整的 GPT-style 语言模型
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = None,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_rope: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Position Encoding (如果不用 RoPE)
        if not use_rope:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.use_rope = use_rope
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            GPTDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = RMSNorm(d_model)
        
        # LM Head (共享 embedding 权重)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list = None
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: [batch, seq_len]
            kv_caches: list of KV cache for each layer
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_caches: updated caches
        """
        batch_size, seq_len = input_ids.shape
        
        # Token Embedding
        x = self.token_emb(input_ids)  # [batch, seq, d_model]
        
        # Position Embedding (如果不用 RoPE)
        if not self.use_rope:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_emb(positions)
        
        # Causal Mask
        # 只在 prefill 时需要，decode 时每次只有 1 个 token
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
        else:
            causal_mask = None
        
        # 初始化 KV Cache（如果需要）
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        
        new_kv_caches = []
        
        # Transformer Layers
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, mask=causal_mask, kv_cache=kv_caches[i])
            new_kv_caches.append(new_cache)
        
        # Final Norm
        x = self.final_norm(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        return logits, new_kv_caches
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        自回归生成
        """
        generated = input_ids.clone()
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # 只输入最后一个 token（使用 KV Cache）
            if kv_caches is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated
            
            # 前向
            logits, kv_caches = self.forward(curr_input, kv_caches)
            next_token_logits = logits[:, -1, :]
            
            # 采样
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated


# ==================== 模型配置示例 ====================
MODEL_CONFIGS = {
    "gpt2-small": {
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    "llama-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,  # MHA
        "d_ff": 11008,
    },
    "llama2-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "d_ff": 11008,
    },
    "mistral-7b": {
        "vocab_size": 32000,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,  # GQA
        "d_ff": 14336,
    },
}
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
