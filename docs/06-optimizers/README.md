# ⚡ 优化器

> 面试频率：🔥🔥🔥🔥 | 难度：⭐⭐⭐

优化器决定模型如何更新参数。Adam 是面试必考题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：优化器的核心区别在于**如何修正梯度**

| 方法 | 核心思想 | 更新公式 | 特点 |
|:---|:---|:---|:---|
| **SGD** | 原始梯度下降 | `θ -= lr * g` | 简单但慢 |
| **Momentum** | 加速 + 减震 | `v = βv + g; θ -= lr * v` | 累积历史方向 |
| **Adam** | 自适应学习率 | `θ -= lr * m/√v` | 每个参数不同步长 |
| **AdamW** | Adam + 解耦权重衰减 | `θ *= (1-wd); θ -= lr * m/√v` | LLM 标配 |

```python
# 优化器的进化，每一步都在解决一个问题

# SGD: 最基础
θ -= lr * grad                         # 问题: 震荡、慢

# Momentum: 累积历史梯度
v = 0.9 * v + grad                       # 累积动量
θ -= lr * v                             # 解决: 加速 + 减震

# Adam: 每个参数自适应学习率
m = 0.9 * m + 0.1 * grad                 # 一阶矩（方向）
v = 0.999 * v + 0.001 * grad²            # 二阶矩（幅度）
θ -= lr * m / √v                       # 解决: 不同参数不同步长

# AdamW: 解耦 weight decay
θ *= (1 - lr * wd)                      # 先衰减权重
θ -= lr * m / √v                       # 再 Adam 更新
```

> 🤔 **Q: 为什么 LLM 用 AdamW 而不用 Adam？**
>
> Adam 的问题：weight decay 被加到 grad 上，会被自适应学习率缩放。
>
> 梯度大的参数 → adaptive lr 小 → weight decay 也被缩小，正则化不均匀。
>
> AdamW 把 weight decay 从 Adam 中抽出来，每个参数被一致地 decay。

---

## 单题目录

- [SGD](sgd.md)
- [SGD with Momentum](momentum.md)
- [Adam](adam.md)
- [AdamW](adamw.md)（可测评）
- [Learning Rate Scheduler](lr-scheduler.md)

## 高频面试问答

**问：AdamW 的衰减为什么要解耦？**

答：使权重衰减不经过自适应梯度缩放。

**问：偏差修正的目的？**

答：修正从零初始化的动量在初始步骤偏向零的问题。

## 测评

本章练习：[adamw](adamw.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
