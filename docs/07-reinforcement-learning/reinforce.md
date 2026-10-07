## REINFORCE

### 🎯 核心思想

最基础的策略梯度算法：用奖励加权来估计梯度。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

### 📝 实现代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    baseline: torch.Tensor = None
) -> torch.Tensor:
    """
    REINFORCE 策略梯度损失
    
    Args:
        log_probs: [batch, seq_len] 每个 action 的 log 概率
        rewards: [batch] 或 [batch, seq_len] 奖励
        baseline: [batch] 基线（减少方差）
    
    Returns:
        loss: 策略梯度损失（负号因为优化器是最小化）
    """
    # 如果 rewards 是整个序列的奖励，扩展到每个 step
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1).expand_as(log_probs)
    
    # 减去 baseline 减少方差
    if baseline is not None:
        advantages = rewards - baseline.unsqueeze(1)
    else:
        advantages = rewards
    
    # REINFORCE: -E[log π(a|s) * A]
    # 负号是因为我们要最大化 reward，但 optimizer 是最小化 loss
    policy_loss = -(log_probs * advantages.detach()).sum(dim=-1).mean()
    
    return policy_loss


class REINFORCEWithBaseline:
    """
    带基线的 REINFORCE
    
    基线通常用 Value Network 估计，减少梯度方差
    """
    
    def __init__(self, policy_net, value_net, lr=1e-3):
        self.policy = policy_net
        self.value = value_net
        
        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    def compute_returns(self, rewards, gamma=0.99):
        """计算折扣回报"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def update(self, states, actions, rewards):
        """
        states: [batch, state_dim]
        actions: [batch]
        rewards: [batch] 每个 step 的奖励
        """
        # 计算回报
        returns = self.compute_returns(rewards)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算 value baseline
        values = self.value(states).squeeze()
        
        # 计算优势
        advantages = returns - values.detach()
        
        # Policy loss
        log_probs = self.policy.log_prob(states, actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # 更新 policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 更新 value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return policy_loss.item(), value_loss.item()
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：GAE 中 lambda 的作用？**

答：控制多步回报与 bootstrap 的偏差/方差权衡。

**问：DPO 与 PPO 最大区别？**

答：DPO 用偏好对做直接目标，无需在线 PPO 更新和单独训练的奖励模型。

[返回本章目录](README.md)
