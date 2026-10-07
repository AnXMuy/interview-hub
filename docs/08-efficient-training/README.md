# 🚀 高效训练

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

高效训练技术让我们能够训练更大的模型。LoRA 是面试必考题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：高效训练技术解决的是**显存不够**或**计算太慢**的问题

| 方法 | 解决问题 | 核心思想 | 代价 |
|:---|:---|:---|:---|
| **LoRA** | 参数太多 | 只训练低秩增量 | 效果略差 |
| **Gradient Checkpointing** | 激活值占显存 | 不存中间层，重算 | 计算多 30% |
| **Mixed Precision** | 模型+梯度占显存 | FP16 计算，FP32 主副本 | 需 loss scaling |
| **Gradient Accumulation** | batch 太小 | 累积多次梯度再更新 | 训练变慢 |

```python
# 显存占用分析（了解这个才能选对方法！）
# 假设 7B 模型，FP32，batch_size=1，seq_len=2048

# 1. 模型参数: 7B * 4 bytes = 28 GB
# 2. 梯度: 7B * 4 bytes = 28 GB
# 3. 优化器状态 (Adam): 7B * 2 * 4 bytes = 56 GB (m + v)
# 4. 激活值: 取决于 batch/seq，通常 10-50 GB

# 解决方案：
# - 参数太多 → LoRA（只训 0.1% 参数）
# - 激活值太大 → Gradient Checkpointing（不存，重算）
# - 模型+梯度大 → Mixed Precision（FP16，减半）
# - batch 太小 → Gradient Accumulation（累积多次）
```

> 🤔 **Q: 我显存不够，应该用哪个技术？**
>
> 1. 先试 Mixed Precision：最简单，一行代码减半显存
> 2. 再试 Gradient Checkpointing：再省 30-50%，但变慢
> 3. batch 小时用 Accumulation：等效大 batch
> 4. 参数量大时用 LoRA：只训练 0.1% 参数

---

## 单题目录

- [LoRA](lora.md)（可测评）
- [Gradient Checkpointing](gradient-checkpointing.md)
- [Mixed Precision Training](mixed-precision.md)
- [Gradient Accumulation](gradient-accumulation.md)

## 高频面试问答

**问：LoRA 为什么能省参数？**

答：冻结原权重，只训练低秩矩阵 A、B。

**问：梯度累积要注意什么？**

答：每个 microbatch 的 loss 按累积步数缩放，最后才更新和清梯度。

## 测评

本章练习：[lora](lora.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
