# 🧠 Attention 机制

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

Attention 是 Transformer 的核心，也是 LLM 面试的重中之重。本章涵盖从基础的 Scaled Dot-Product Attention 到工业级优化的 Flash Attention。

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：MHA/GQA/MQA 的区别就在于 **KV 头数不同**，代码上只差一个参数！

| 方法 | KV 头数 | KV Cache 占用 | 代码差异 | 使用模型 |
|:---|:---:|:---:|:---|:---|
| **MHA** | = Q 头数 | 100% | `num_kv_heads = num_heads` | GPT-3, BERT |
| **GQA** | 介于两者之间 | 12.5%-50% | `num_kv_heads = num_heads // 8` | LLaMA-2 70B, Mistral |
| **MQA** | 1 | 最小(1/h) | `num_kv_heads = 1` | PaLM, Falcon |

```python
# 三种 Attention 的核心区别，就这几行！
class MHA:   W_k = Linear(d, num_heads * head_dim)     # 每个Q头有独立KV
class GQA:   W_k = Linear(d, num_kv_heads * head_dim)  # 多个Q头共享一组KV
class MQA:   W_k = Linear(d, head_dim)                 # 所有Q头共享一个KV
```

> 🤔 **Q: 为什么减少 KV 头数能省内存？**
>
> KV Cache 大小 = `2 × batch × num_kv_heads × seq × head_dim × dtype_bytes`
>
> 减少 `num_kv_heads` 直接减少内存！LLaMA-2 70B 用 GQA (8 KV heads vs 64 Q heads)，省了 87.5%！

---

## 单题目录

- [Scaled Dot-Product Attention](scaled-dot-product-attention.md)（可测评）
- [Multi-Head Attention](multi-head-attention.md)
- [Causal Mask](causal-mask.md)
- [Grouped-Query Attention (GQA)](gqa.md)
- [Multi-Query Attention (MQA)](mqa.md)
- [Flash Attention 原理](flash-attention.md)

## 高频面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

## 测评

本章练习：[scaled-dot-product-attention](scaled-dot-product-attention.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
