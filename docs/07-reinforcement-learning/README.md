# 🎮 强化学习

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐⭐

RLHF 是 ChatGPT 成功的关键。PPO、DPO、GRPO 是面试热门话题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：RL 算法的核心区别在于**如何估计优势**和**如何更新策略**

| 方法 | 优势估计 | 策略更新 | 核心特点 | 使用场景 |
|:---|:---|:---|:---|:---|
| **REINFORCE** | R - baseline | 无约束 | 最基础 | 教学 |
| **PPO** | GAE | Clipped | 稳定 + 多轮更新 | RLHF |
| **DPO** | 内置 | 偏好直接转梯度 | 无需 RM/Critic | 离线偏好 |
| **GRPO** | 组内相对 | Clipped + KL | 无需 Critic | DeepSeek |

```python
# RL 算法的演进，理解从哪里来、解决什么问题

# REINFORCE: 最基础，梯度 = reward * logπ
loss = -R * logπ(a|s)                       # 问题: 方差大、不稳定

# PPO: 加了 GAE + Clipping
advantage = GAE(r, V)                       # 用 Critic 估计优势
loss = -min(ratio * A, clip(ratio) * A)     # 限制更新幅度

# DPO: 跳过 RM，直接从偏好学习
loss = -logσ(β * (logπ_w - logπ_l))      # 偏好对 -> 梯度

# GRPO: 组内比较，不需要 Critic
advantage = (r - mean(r)) / std(r)          # 组内相对优势
loss = PPO_loss + β * KL                    # 还是用 PPO 更新
```

> 🤔 **Q: PPO 和 DPO 有什么区别？什么时候用哪个？**
>
> | | PPO | DPO |
> |:---|:---|:---|
> | 需要 | RM + Critic + 采样 | 只需要偏好数据 |
> | 训练 | 复杂，多阶段 | 简单，一阶段 |
> | 探索 | 在线探索 | 无探索（离线） |
> | 适合 | 持续训练、需要探索 | 静态偏好数据 |
>
> 简单场景用 DPO，需要持续迭代/探索用 PPO。

---

## 单题目录

- [REINFORCE](reinforce.md)
- [GAE](gae.md)（可测评）
- [PPO](ppo.md)
- [DPO](dpo.md)
- [GRPO](grpo.md)
- [RLHF 全流程](rlhf-pipeline.md)

## 高频面试问答

**问：GAE 中 lambda 的作用？**

答：控制多步回报与 bootstrap 的偏差/方差权衡。

**问：DPO 与 PPO 最大区别？**

答：DPO 用偏好对做直接目标，无需在线 PPO 更新和单独训练的奖励模型。

## 测评

本章练习：[gae](gae.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
