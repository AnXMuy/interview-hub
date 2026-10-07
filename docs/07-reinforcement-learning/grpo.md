## GRPO

### 🎯 核心思想

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的算法：
- **去掉 Critic**: 用组内相对奖励代替 value baseline
- **组内比较**: 同一 prompt 采样多个 response，相互比较

$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)} \quad \text{(组内标准化优势)}$$

> 🤔 **Q: GRPO 为什么不需要 Critic？这不会不稳定吗？**
>
> GRPO 的精妙在于：同一个 prompt 的多个 response 形成"对照组"。
>
> 组内标准化 `(r - mean) / std` 相当于自动的 baseline，不需要学一个 Critic。
>
> 为什么稳定？因为比较的是同一个 prompt 的 response，变量少很多。

### 📝 实现代码

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization 🔥🔥🔥 DeepSeek 使用
    
    核心思想：
    1. 对同一个 prompt 采样 G 个不同的 response
    2. 用组内相对奖励作为优势，不需要 Critic
    3. 用 PPO 的 clipping 更新 policy
    
    优势：
    - 比 PPO 简单（不需要 value network）
    - 比 DPO 更在线（可以探索）
    - 训练更稳定
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,
        group_size: int = 4,  # 每个 prompt 采样几个 response
        beta: float = 0.04,   # KL 惩罚系数
        clip_eps: float = 0.2,
        lr: float = 1e-6
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.group_size = group_size
        self.beta = beta
        self.clip_eps = clip_eps
        
        # 冻结 reference
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    
    def compute_rewards(self, prompts, responses):
        """
        计算奖励（可以是规则或 Reward Model）
        
        例如：数学题检查答案是否正确
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            # 这里用简化的规则奖励
            # 实际可以用 Reward Model 或执行代码验证
            reward = self.rule_based_reward(prompt, response)
            rewards.append(reward)
        return torch.tensor(rewards)
    
    def rule_based_reward(self, prompt, response):
        """示例：基于规则的奖励"""
        # 实际中可以：执行代码检查输出、数学验证、格式检查等
        return 0.0  # placeholder
    
    def compute_loss(
        self,
        prompts,           # [batch_size] 提示
        responses_list,    # [[batch_size] * G] G 组回答
        rewards_tensor,    # [batch_size, G] 奖励
        old_log_probs_list  # [[batch_size] * G] 采样时的 log prob
    ):
        """
        GRPO 核心损失计算
        """
        batch_size = len(prompts)
        G = self.group_size
        
        # ========== 1. 计算组内相对优势 ==========
        # rewards: [batch, G]
        mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)  # [batch, 1]
        std_rewards = rewards_tensor.std(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        
        # 标准化优势：不需要 Critic！
        advantages = (rewards_tensor - mean_rewards) / std_rewards  # [batch, G]
        
        # ========== 2. 计算 PPO-style clipped loss ==========
        total_loss = 0
        
        for g in range(G):
            # 当前 policy 的 log prob
            new_log_probs = self.get_log_probs(
                self.policy,
                prompts,
                responses_list[g]
            )
            
            # Reference 的 log prob
            with torch.no_grad():
                ref_log_probs = self.get_log_probs(
                    self.ref,
                    prompts,
                    responses_list[g]
                )
            
            # 重要性采样比率
            old_log_probs = old_log_probs_list[g]
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate
            adv = advantages[:, g]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL 惩罚
            kl = (new_log_probs - ref_log_probs).mean()
            kl_penalty = self.beta * kl
            
            total_loss += policy_loss + kl_penalty
        
        return total_loss / G
    
    def get_log_probs(self, model, prompts, responses):
        """计算 response 的 log prob"""
        # 简化实现
        # 实际需要正确处理 tokenization 和 padding
        pass
    
    def train_step(self, prompts):
        """
        GRPO 训练一步
        
        1. 对每个 prompt 采样 G 个 response
        2. 计算奖励
        3. 计算组内优势
        4. PPO 更新
        """
        # 采样
        responses_list = []
        old_log_probs_list = []
        
        with torch.no_grad():
            for _ in range(self.group_size):
                responses, log_probs = self.policy.sample(prompts)
                responses_list.append(responses)
                old_log_probs_list.append(log_probs)
        
        # 计算奖励
        all_rewards = []
        for responses in responses_list:
            rewards = self.compute_rewards(prompts, responses)
            all_rewards.append(rewards)
        rewards_tensor = torch.stack(all_rewards, dim=1)  # [batch, G]
        
        # 计算并优化
        loss = self.compute_loss(
            prompts,
            responses_list,
            rewards_tensor,
            old_log_probs_list
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
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
