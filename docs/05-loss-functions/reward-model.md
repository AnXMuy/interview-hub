## Reward Model Loss

### 🎯 核心思想

Reward Model 学习人类偏好：给定同一 prompt 的两个 response，判断哪个更好。

$$\mathcal{L} = -\log\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $y_w$ 是 chosen（更好的），$y_l$ 是 rejected（更差的）。

> 🤔 **Q: 为什么用 `r_chosen - r_rejected` 而不是直接让 chosen 得分高？**
>
> 这是 Bradley-Terry 模型：我们关心的是**相对质量**，不是绝对分数。
>
> 如果直接让 chosen 得分高，模型会让所有分数越来越高（分数膨胀）。
>
> 用差值后，模型只需要学会“谁更好”，不用关心绝对分数。

### 📝 实现代码

```python
def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor
) -> torch.Tensor:
    """
    Reward Model Loss (Bradley-Terry Model)
    
    Args:
        chosen_rewards: [batch] chosen response 的奖励分数
        rejected_rewards: [batch] rejected response 的奖励分数
    
    Returns:
        loss: 标量
    """
    # Bradley-Terry: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    # Loss = -log(P) = -log(sigmoid(r_chosen - r_rejected))
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    
    return loss.mean()


def reward_model_loss_with_margin(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.0
) -> torch.Tensor:
    """
    带 margin 的 Reward Model Loss
    
    要求 chosen 比 rejected 高出至少 margin
    """
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards - margin)
    return loss.mean()


class RewardModel(torch.nn.Module):
    """
    Reward Model 示例结构
    """
    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base = base_model  # 预训练 LLM
        self.reward_head = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # 获取最后一个 token 的 hidden state
        outputs = self.base(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # 取最后一个非 padding token
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        last_token_hidden = last_hidden[
            torch.arange(batch_size),
            seq_lengths
        ]
        
        # 输出标量奖励
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        return reward


def train_reward_model_step(
    model,
    chosen_input_ids,
    chosen_attention_mask,
    rejected_input_ids,
    rejected_attention_mask,
    optimizer
):
    """Reward Model 训练一步"""
    # 前向传播
    chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
    rejected_rewards = model(rejected_input_ids, rejected_attention_mask)
    
    # 计算 loss
    loss = reward_model_loss(chosen_rewards, rejected_rewards)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 计算准确率
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss.item(), accuracy.item()
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [cross-entropy](cross-entropy.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为什么直接算 log(softmax) 不稳定？**

答：exp 可能溢出或下溢；用 logsumexp/log_softmax。

**问：padding token 如何处理？**

答：用 ignore_index 排除且平均值只除有效 token 数。

[返回本章目录](README.md)
