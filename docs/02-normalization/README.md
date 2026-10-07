# 📏 归一化层

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐

归一化是深度学习训练稳定的关键。LLM 领域主要使用 Layer Normalization 和 RMSNorm。

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：归一化的核心区别在于**在哪个维度计算统计量**

| 方法 | 统计维度 | 公式差异 | 使用场景 |
|:---|:---|:---|:---|
| **LayerNorm** | 每个样本的特征维度 | `(x - mean) / std * γ + β` | Transformer |
| **RMSNorm** | 每个样本的特征维度 | `x / RMS(x) * γ`（无 mean） | LLaMA, Mistral |
| **BatchNorm** | 整个 batch 的同一特征 | 同 LayerNorm，但维度不同 | CNN |

```python
# 三种 Norm 的核心区别，就这几行！
# 输入 x: [batch, seq, hidden]

# LayerNorm: 对每个 token 的 hidden 维度归一化
mean = x.mean(dim=-1)          # [batch, seq]

# RMSNorm: 同样对 hidden 维度，但只用 RMS，不用 mean
rms = sqrt(mean(x**2, dim=-1)) # [batch, seq]

# BatchNorm: 对每个特征在 batch 维度归一化
mean = x.mean(dim=0)           # [seq, hidden]
```

> 🤔 **Q: 为什么 LLM 不用 BatchNorm？**
>
> 1. 序列长度变化：不同样本长度不同，BatchNorm 难以处理
> 2. 小 batch 不稳定：LLM 训练通常用小 batch，统计量方差大
> 3. 推理时不一致：训练用 batch 统计量，推理用 running stats，有差异

---

## 单题目录

- [Layer Normalization](layer-norm.md)
- [RMS Normalization](rms-norm.md)（可测评）
- [Batch Normalization](batch-norm.md)
- [Pre-Norm vs Post-Norm](pre-vs-post-norm.md)

## 高频面试问答

**问：LN 与 RMSNorm 有何差别？**

答：RMSNorm 不减均值，只除均方根；均沿特征维归一化。

**问：为什么小 batch 不利于 BN？**

答：统计量由 batch 估计，样本少时方差较大。

## 测评

本章练习：[rms-norm](rms-norm.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
