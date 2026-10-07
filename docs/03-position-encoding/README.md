# 📍 位置编码

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

位置编码让 Transformer 感知序列中 token 的位置信息。RoPE 是现代 LLM 的标配，必须掌握！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：位置编码的核心区别在于**作用方式**和**作用位置**

| 方法 | 作用方式 | 作用位置 | 核心代码差异 | 使用模型 |
|:---|:---|:---|:---|:---|
| **Sinusoidal** | 加法 | Embedding | `x = x + pe` | Transformer |
| **Learnable** | 加法 | Embedding | `x = x + embed(pos)` | BERT, GPT-2 |
| **RoPE** | 乘法(旋转) | Q, K | `q = rotate(q, pos)` | LLaMA, Mistral |
| **ALiBi** | 加法偏置 | Attention Score | `score = score + bias` | BLOOM |

```python
# 四种位置编码的核心区别

# Sinusoidal/Learnable: 加到 embedding 上，影响 Q/K/V
x = x + position_encoding
q, k, v = W_q(x), W_k(x), W_v(x)  # 都含有位置信息

# RoPE: 只旋转 Q 和 K，不影响 V
q, k, v = W_q(x), W_k(x), W_v(x)
q, k = apply_rope(q, pos), apply_rope(k, pos)  # 只影响 Q/K

# ALiBi: 直接加到 attention score 上
score = q @ k.T / sqrt(d) + alibi_bias[distance]  # 不修改 Q/K/V
```

> 🤔 **Q: 为什么 RoPE 只作用于 Q 和 K，不作用于 V？**
>
> Attention 的位置关联性由 Q·K 决定。V 只是被加权的内容，不需要位置信息。
>
> 而且 RoPE 的数学性质（点积只依赖相对位置）只在 Q·K 时成立，应用到 V 没有这个效果。

---

## 单题目录

- [Sinusoidal Position Encoding](sinusoidal.md)
- [Learnable Position Encoding](learnable.md)
- [Rotary Position Embedding (RoPE)](rope.md)（可测评）
- [ALiBi](alibi.md)

## 高频面试问答

**问：RoPE 施加到哪些张量？**

答：旋转 Q、K 而非 V；点积编码相对位置。

**问：位置偏移如何处理？**

答：增量解码时用全局位置，不可每步重新从零计数。

## 测评

本章练习：[rope](rope.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
