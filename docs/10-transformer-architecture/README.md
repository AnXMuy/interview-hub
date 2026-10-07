# 🏗️ Transformer 架构

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐

完整的 Transformer 架构是所有 LLM 的基础。本章整合所有组件！

## 本章速览

## 架构一览对比

> 💡 **一句话区分**：不同架构的核心区别在于 **Attention 是否有掩码** 和 **是否有 Cross Attention**

| 架构 | Self-Attn | Cross-Attn | Mask | 代表模型 |
|:---|:---:|:---:|:---|:---|
| **Encoder-Only** | ✓ | ✗ | 无 | BERT |
| **Decoder-Only** | ✓ | ✗ | Causal | GPT, LLaMA |
| **Encoder-Decoder** | ✓ | ✓ | Causal(Dec) | T5, BART |

```python
# 三种架构的核心区别

# Encoder-Only (BERT): 看得到所有 token，适合理解
output = SelfAttn(x, x, x, mask=None)       # 全部可见

# Decoder-Only (GPT): 只能看到左边，适合生成
output = SelfAttn(x, x, x, mask=causal)     # 只能看左边

# Encoder-Decoder (T5): Encoder 全可见，Decoder 用 Encoder 输出
enc_out = EncoderSelfAttn(x, x, x, mask=None)  # Encoder 全可见
dec_out = DecoderSelfAttn(y, y, y, mask=causal) # Decoder 只看左边
final = CrossAttn(dec_out, enc_out, enc_out)    # 看 Encoder
```

> 🤔 **Q: 为什么现代 LLM 都用 Decoder-Only？**
>
> 1. **简单**：一个统一架构，无需分 Encoder/Decoder
> 2. **扩展性好**：只需堆堆层，没有复杂的 cross-attention
> 3. **训练高效**：每个 position 都是一个训练样本（next token prediction）
> 4. **推理简单**：KV Cache 实现简单，不用处理 Encoder

---

## 单题目录

- [Transformer 概览](overview.md)
- [Feed-Forward Network](ffn.md)（可测评）
- [Transformer Encoder Layer](encoder.md)
- [Transformer Decoder Layer](decoder.md)
- [GPT-style Decoder-Only](gpt-decoder.md)
- [残差连接与层归一化](residual-norm.md)
- [完整 Transformer 实现](full-transformer.md)

## 高频面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

## 测评

本章练习：[ffn](ffn.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
