## GAE

### 🎯 核心思想

GAE (Generalized Advantage Estimation) 在 TD(0) 和 Monte Carlo 之间做权衡：
- **λ = 0**: TD(0)，高偏差低方差
- **λ = 1**: Monte Carlo，低偏差高方差

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \quad \text{(TD error)}$$

> 🤔 **Q: λ 是什么意思？为什么通常取 0.95？**
>
> λ 控制"看多远"：
> - λ=0: 只用当前 step 的 TD error，偏差大但方差小
> - λ=1: 用整个 trajectory，方差大但无偏
>
> 0.95 是经验值，偏向于使用更多未来信息，同时保持稳定。

### 📝 实现代码

```python
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation 🔥 PPO 核心组件
    
    Args:
        rewards: [T] 每步奖励
        values: [T+1] 价值估计（包含 next_value）
        dones: [T] 是否终止
        gamma: 折扣因子（通常 0.99）
        lam: GAE lambda（通常 0.95，平衡偏差-方差）
    
    Returns:
        advantages: [T] 优势估计
        returns: [T] 回报（用于更新 value function）
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0
    
    # 从后往前计算（递归展开）
    for t in reversed(range(T)):
        # 终止状态：下一个状态的值为 0
        next_value = values[t + 1] * (1 - dones[t])
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE 递归: A_t = δ_t + γλ * A_{t+1}
        # (1 - dones[t]) 处理 episode 边界
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    # 回报 = 优势 + 价值（用于训练 value network）
    returns = advantages + values[:-1]
    
    # 标准化优势（稳定训练）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def compute_gae_vectorized(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    向量化 GAE（更快，适合大批量）
    
    rewards, values, dones: [batch, T]
    """
    batch_size, T = rewards.shape
    
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, device=rewards.device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[:, -1]  # 使用最后的 value
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * lam * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
    
    returns = advantages + values
    
    # 标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：GAE 中 lambda 的作用？**

答：控制多步回报与 bootstrap 的偏差/方差权衡。

**问：DPO 与 PPO 最大区别？**

答：DPO 用偏好对做直接目标，无需在线 PPO 更新和单独训练的奖励模型。

[返回本章目录](README.md)
