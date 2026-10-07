## RLHF 全流程

### 📝 完整流程代码

```python
class RLHFPipeline:
    """
    RLHF 完整流程
    
    Stage 1: SFT (Supervised Fine-Tuning)
    Stage 2: Reward Model Training
    Stage 3: PPO/GRPO/DPO Training
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    # ========== Stage 1: SFT ==========
    def sft_train(self, sft_dataset, epochs=3):
        """
        监督微调：在高质量数据上训练
        """
        self.policy = self.base_model
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            for batch in sft_dataset:
                loss = self.compute_sft_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.policy
    
    def compute_sft_loss(self, batch):
        """SFT Loss = Next Token Prediction"""
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        outputs = self.policy(input_ids)
        logits = outputs.logits
        
        # Shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss
    
    # ========== Stage 2: Reward Model ==========
    def train_reward_model(self, preference_dataset, epochs=1):
        """
        训练 Reward Model
        
        输入: (prompt, chosen, rejected) 三元组
        目标: R(chosen) > R(rejected)
        """
        self.reward_model = RewardModel(self.base_model.config)
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            for batch in preference_dataset:
                chosen_rewards = self.reward_model(batch['chosen_input_ids'])
                rejected_rewards = self.reward_model(batch['rejected_input_ids'])
                
                # Bradley-Terry Loss
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self.reward_model
    
    # ========== Stage 3: RL Training ==========
    def rl_train_ppo(self, prompts_dataset, epochs=1):
        """PPO 训练"""
        ref_model = copy.deepcopy(self.policy)
        for param in ref_model.parameters():
            param.requires_grad = False
        
        ppo = PPO(self.policy, self.value_net)
        
        for epoch in range(epochs):
            for prompts in prompts_dataset:
                # 1. 生成 response
                responses, old_log_probs = self.policy.sample(prompts)
                
                # 2. 计算 reward
                rewards = self.reward_model(prompts, responses)
                
                # 3. KL 惩罚
                ref_log_probs = ref_model.log_prob(prompts, responses)
                kl_penalty = self.kl_coef * (old_log_probs - ref_log_probs)
                rewards = rewards - kl_penalty
                
                # 4. PPO 更新
                ppo.update(prompts, responses, old_log_probs, rewards)
    
    def rl_train_dpo(self, preference_dataset, epochs=1):
        """DPO 训练（更简单）"""
        ref_model = copy.deepcopy(self.policy)
        dpo = DPOTrainer(self.policy, ref_model)
        
        for epoch in range(epochs):
            for batch in preference_dataset:
                dpo.train_step(batch)


class RewardModel(nn.Module):
    """Reward Model 结构"""
    
    def __init__(self, config):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.reward_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]
        
        # 输出标量奖励
        reward = self.reward_head(last_hidden).squeeze(-1)
        
        return reward
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
