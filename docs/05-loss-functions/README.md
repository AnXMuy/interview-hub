# 📉 损失函数

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐

损失函数是模型训练的核心。Cross Entropy 的数值稳定性实现是必考题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：损失函数的核心区别在于**衡量什么**

| 方法 | 衡量内容 | 核心公式 | 使用场景 |
|:---|:---|:---|:---|
| **Cross Entropy** | 预测与真实的差异 | `-log(p_target)` | 分类、LLM |
| **Focal Loss** | CE + 难样本加权 | `(1-p)² * CE` | 类别不平衡 |
| **KL Divergence** | 两个分布的差异 | `p * log(p/q)` | 知识蒸馏、RLHF |
| **MSE** | 预测与真实的差的平方 | `(y - ŷ)²` | 回归 |
| **Reward Model Loss** | 偏好对的正确排序 | `-logσ(r_w - r_l)` | RLHF |

```python
# LLM 训练中的损失函数

# SFT: 简单的 Cross Entropy
loss = -log(p(next_token | context))      # 每个 token 的 CE

# RM: 学习偏好排序
loss = -logσ(r_chosen - r_rejected)        # Bradley-Terry 模型

# DPO: 直接从偏好中学习
loss = -logσ(β * (logπ - logπ_ref))      # 无需单独的 RM

# PPO: 策略梯度 + KL 惩罚
loss = -advantage * ratio + β * KL(π || π_ref)  # 限制更新幅度
```

> 🤔 **Q: 为什么 LLM 用 Cross Entropy 而不是 MSE？**
>
> 1. LLM 是分类任务（预测下一个 token），不是回归
> 2. CE 的梯度比 MSE 更合理：当预测错误时 CE 梯度更大、更新更快
> 3. CE 跟 softmax 配合天然，softmax 的导数刚好是预测-真实

---

## 单题目录

- [Cross Entropy Loss](cross-entropy.md)（可测评）
- [Softmax 数值稳定性](stable-softmax.md)
- [Focal Loss](focal.md)
- [KL Divergence](kl-divergence.md)
- [MSE Loss](mse.md)
- [SFT Loss](sft.md)
- [Reward Model Loss](reward-model.md)
- [Label Smoothing](label-smoothing.md)

## 高频面试问答

**问：为什么直接算 log(softmax) 不稳定？**

答：exp 可能溢出或下溢；用 logsumexp/log_softmax。

**问：padding token 如何处理？**

答：用 ignore_index 排除且平均值只除有效 token 数。

## 测评

本章练习：[cross-entropy](cross-entropy.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
