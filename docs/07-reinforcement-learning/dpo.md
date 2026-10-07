## DPO

### 🎯 核心思想

DPO 跳过 Reward Model，直接从偏好数据优化 Policy：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**直觉**：让模型更偏好 chosen response，同时不偏离 reference model 太远。

> 🤔 **Q: DPO 的 β 是什么意思？怎么选？**
>
> β 控制"离 reference 多远"：
> - β 大: 惩罚大，不能偏离 ref 太多（保守）
> - β 小: 惩罚小，可以大胆偏离 ref（激进）
>
> 通常 β ∈ [0.1, 0.5]，需要调参。

### 📝 实现代码

```python
class DPOTrainer:
    """
    Direct Preference Optimization 🔥🔥🔥
    
    优点：
    - 不需要训练 Reward Model
    - 不需要 PPO 的复杂采样
    - 训练更稳定
    
    缺点：
    - 依赖高质量偏好数据
    - 没有在线探索，可能过拟合
    """
    
    def __init__(
        self,
        policy_model,
        ref_model,  # 冻结的参考模型
        beta: float = 0.1,
        lr: float = 1e-6
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta
        
        # 冻结 reference model
        for param in self.ref.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr
        )
    
    def get_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列的 log 概率
        
        input_ids: [batch, seq_len] prompt + response
        labels: [batch, seq_len] 只有 response 部分有效（prompt 部分为 -100）
        """
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # Shift: logits[:, :-1] 预测 labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # 计算每个 token 的 log prob
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 取出真实 token 的 log prob
        # [batch, seq_len-1]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask 掉 padding 和 prompt 部分
        mask = (shift_labels != -100).float()
        
        # 求和得到序列 log prob
        seq_log_probs = (token_log_probs * mask).sum(dim=-1)
        
        return seq_log_probs
    
    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        """
        计算 DPO 损失
        
        batch 包含:
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
        """
        # ========== Policy model log probs ==========
        policy_chosen_logps = self.get_log_probs(
            self.policy,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels']
        )
        policy_rejected_logps = self.get_log_probs(
            self.policy,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels']
        )
        
        # ========== Reference model log probs ==========
        with torch.no_grad():
            ref_chosen_logps = self.get_log_probs(
                self.ref,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels']
            )
            ref_rejected_logps = self.get_log_probs(
                self.ref,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels']
            )
        
        # ========== DPO Loss ==========
        # log(π/π_ref) 差值
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO loss = -log(sigmoid(β * (pi_logratio - ref_logratio)))
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # 计算指标
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        info = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_rewards': chosen_rewards.mean().item(),
            'rejected_rewards': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }
        
        return loss, info
    
    def train_step(self, batch):
        loss, info = self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return info
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [gae](gae.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：GAE 中 lambda 的作用？**

答：控制多步回报与 bootstrap 的偏差/方差权衡。

**问：DPO 与 PPO 最大区别？**

答：DPO 用偏好对做直接目标，无需在线 PPO 更新和单独训练的奖励模型。

[返回本章目录](README.md)
