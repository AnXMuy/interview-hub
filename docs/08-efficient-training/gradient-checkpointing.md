## Gradient Checkpointing

### 🎯 核心思想

时间换空间：不保存中间激活值，反向传播时重新计算。

- **标准训练**: 保存所有层的激活 → 内存 O(n)
- **Checkpointing**: 只保存检查点，其他重算 → 内存 O(√n)，时间 +30%

### 📝 实现代码

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    """
    使用 Gradient Checkpointing 的 Transformer 层
    
    不保存中间激活值，反向传播时重新计算
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = True):
        if use_checkpoint and self.training:
            # 使用 checkpoint：不保存中间激活
            attn_out = checkpoint(
                self._attn_block,
                x,
                use_reentrant=False  # 推荐设为 False
            )
            ffn_out = checkpoint(
                self._ffn_block,
                x + attn_out,
                use_reentrant=False
            )
            return x + attn_out + ffn_out
        else:
            # 正常前向
            attn_out = self._attn_block(x)
            x = x + attn_out
            ffn_out = self._ffn_block(x)
            return x + ffn_out
    
    def _attn_block(self, x):
        """Attention 子块"""
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        return attn_out
    
    def _ffn_block(self, x):
        """FFN 子块"""
        return self.ffn(self.norm2(x))


def manual_checkpoint_forward(func, *args):
    """
    手动实现 Gradient Checkpointing
    
    原理：
    1. 前向时不保存中间结果（使用 no_grad）
    2. 反向时重新计算前向
    """
    class CheckpointFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # 保存输入（用于反向时重算）
            ctx.save_for_backward(*inputs)
            ctx.func = func
            
            # 前向计算（不追踪梯度）
            with torch.no_grad():
                outputs = func(*inputs)
            
            return outputs
        
        @staticmethod
        def backward(ctx, *grad_outputs):
            inputs = ctx.saved_tensors
            
            # 重新启用梯度计算
            with torch.enable_grad():
                # 创建需要梯度的输入副本
                inputs_with_grad = [inp.detach().requires_grad_(True) for inp in inputs]
                
                # 重新计算前向
                outputs = ctx.func(*inputs_with_grad)
                
                # 计算梯度
                torch.autograd.backward(outputs, grad_outputs)
            
            return tuple(inp.grad for inp in inputs_with_grad)
    
    return CheckpointFunction.apply(*args)


# ==================== 内存分析 ====================
def analyze_memory_savings(num_layers: int, hidden_size: int, seq_len: int, batch_size: int):
    """分析 Checkpointing 的内存节省"""
    
    # 估算每层的激活内存（简化）
    # 主要是 attention scores [batch, heads, seq, seq] 和 hidden states [batch, seq, hidden]
    activation_per_layer = (
        batch_size * seq_len * hidden_size * 2 +  # FFN 中间激活
        batch_size * seq_len * seq_len * 2         # Attention scores (假设 2 bytes per value)
    )
    
    standard_memory = num_layers * activation_per_layer
    checkpoint_memory = math.sqrt(num_layers) * activation_per_layer  # 只保存 √n 个检查点
    
    print(f"=== Gradient Checkpointing 内存分析 ===")
    print(f"配置: {num_layers} layers, hidden={hidden_size}, seq={seq_len}, batch={batch_size}")
    print(f"标准训练内存: {standard_memory / 1024**3:.2f} GB")
    print(f"Checkpointing 内存: {checkpoint_memory / 1024**3:.2f} GB")
    print(f"节省: {(1 - checkpoint_memory/standard_memory) * 100:.1f}%")


if __name__ == "__main__":
    analyze_memory_savings(
        num_layers=32,
        hidden_size=4096,
        seq_len=2048,
        batch_size=4
    )
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：LoRA 为什么能省参数？**

答：冻结原权重，只训练低秩矩阵 A、B。

**问：梯度累积要注意什么？**

答：每个 microbatch 的 loss 按累积步数缩放，最后才更新和清梯度。

[返回本章目录](README.md)
