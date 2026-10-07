## PPO

### 🎯 核心思想

PPO 是 RLHF 的主力算法。核心创新是 **Clipped Surrogate Objective**：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率。

> 🤔 **Q: 为什么用 min 而不是直接用 clipped？**
>
> 关键洞察：只有当 ratio “走过头” 时才需要 clip！
>
> - A > 0（好 action）: 我们想增大 ratio，但 ratio 已经 > 1+ε 时，clipped 更小，取 min 限制上升
> - A < 0（坏 action）: 我们想减小 ratio，但 ratio 已经 < 1-ε 时，clipped 更大，取 min 限制下降
>
> 总之：不管哪个方向，都防止更新过大。

### 📝 实现代码

```python
class PPO:
    """
    Proximal Policy Optimization 🔥🔥🔥 RLHF 核心算法
    """
    
    def __init__(
        self,
        policy_net,
        value_net,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy_net
        self.value = value_net
        
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 可以用一个 optimizer（如果 policy 和 value 共享参数）
        self.optimizer = torch.optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=lr
        )
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        计算 PPO 损失
        
        Returns:
            total_loss: 总损失
            info: 包含各部分损失的字典
        """
        # ========== 1. Policy Loss (Clipped Surrogate) ==========
        # 计算新的 log prob
        new_log_probs, entropy = self.policy.evaluate(states, actions)
        
        # 重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        
        # 取 min 是因为：
        # - 如果 advantage > 0，我们想增加这个 action 的概率
        #   但不希望 ratio 增加太多（被 clip 限制）
        # - 如果 advantage < 0，我们想减少概率
        #   但不希望 ratio 减少太多
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # ========== 2. Value Loss ==========
        values = self.value(states).squeeze()
        value_loss = F.mse_loss(values, returns)
        
        # ========== 3. Entropy Bonus ==========
        # 鼓励探索，防止过早收敛
        entropy_loss = -entropy.mean()
        
        # ========== 总损失 ==========
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'approx_kl': ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
        }
        
        return total_loss, info
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        num_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        PPO 更新一轮
        """
        # 计算 value
        with torch.no_grad():
            values = self.value(states).squeeze()
            # 添加 next_value
            next_value = self.value(states[-1:]).squeeze()
            values_with_next = torch.cat([values, next_value])
        
        # 计算 GAE
        advantages, returns = compute_gae(rewards, values_with_next, dones, self.gamma, self.lam)
        
        # 多轮更新（PPO 可以重复使用采样数据）
        dataset_size = len(states)
        
        for epoch in range(num_epochs):
            # Mini-batch 更新
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                loss, info = self.compute_loss(
                    states[batch_indices],
                    actions[batch_indices],
                    old_log_probs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices]
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
        
        return info
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
