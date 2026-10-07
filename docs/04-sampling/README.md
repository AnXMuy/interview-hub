# 🎲 采样策略

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐

采样策略决定 LLM 如何从概率分布中选择下一个 token。Top-p 是面试必考题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：采样策略的核心区别在于**如何过滤候选词**

| 方法 | 过滤方式 | 核心代码 | 特点 |
|:---|:---|:---|:---|
| **Greedy** | 只保留最大 | `argmax(logits)` | 确定但易重复 |
| **Temperature** | 缩放 logits | `softmax(logits / T)` | 控制多样性 |
| **Top-k** | 保留前 k 个 | `topk(logits, k)` | 固定候选数 |
| **Top-p** | 保留累积概率 ≤ p | `cumsum(probs) <= p` | 动态候选数 |
| **Beam Search** | 保留 k 个最优序列 | 追踪多路径 | 质量高但无聊 |

```python
# 采样策略的核心区别：过滤方式不同

# Greedy: 只要最大的
token = logits.argmax()                      # 只留 1 个

# Top-k: 只保留前 k 个（数量固定）
mask = logits < topk(logits, k).values[-1]   # 只留 k 个

# Top-p: 保留累积概率达到 p 的（数量动态）
mask = cumsum(sorted_probs) > p              # 数量随分布变化
```

> 🤔 **Q: 为什么 Top-p 比 Top-k 更常用？**
>
> Top-k 的问题：当模型很确定时（比如下一个词 90% 是 "the"），k=50 会引入 49 个几乎不可能的词。
>
> Top-p 自动适应：确定时只留几个，不确定时留很多。这就是 "nucleus" （核心）的含义。

---

## 单题目录

- [Temperature Sampling](temperature.md)
- [Top-k Sampling](top-k.md)
- [Top-p (Nucleus) Sampling](top-p.md)（可测评）
- [Beam Search](beam-search.md)
- [Greedy Decoding](greedy.md)
- [组合采样策略](combined-sampling.md)

## 高频面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

## 测评

本章练习：[top-p](top-p.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
