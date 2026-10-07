## Gradient Accumulation

### 🎯 核心思想

用小 batch 模拟大 batch：累积多个 mini-batch 的梯度后再更新。

**等效 batch size** = mini_batch_size × accumulation_steps

### 📝 实现代码

```python
class GradientAccumulationTrainer:
    """
    梯度累积训练
    
    用小 batch 模拟大 batch
    """
    
    def __init__(
        self,
        model,
        optimizer,
        accumulation_steps: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def train_step(self, inputs, labels):
        """
        累积梯度的训练步骤
        """
        # 计算 loss（除以累积步数，保证梯度量级正确）
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss = loss / self.accumulation_steps
        
        # 反向传播（梯度累积）
        loss.backward()
        
        self.step_count += 1
        
        # 每 accumulation_steps 步更新一次
        if self.step_count % self.accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return True  # 发生了更新
        
        return False  # 只累积，未更新
    
    def train_epoch(self, dataloader):
        """
        完整的一个 epoch
        """
        self.model.train()
        total_loss = 0
        update_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels) / self.accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.accumulation_steps
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                update_count += 1
        
        # 处理最后不足 accumulation_steps 的 batch
        if (batch_idx + 1) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / len(dataloader)
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [lora](lora.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：LoRA 为什么能省参数？**

答：冻结原权重，只训练低秩矩阵 A、B。

**问：梯度累积要注意什么？**

答：每个 microbatch 的 loss 按累积步数缩放，最后才更新和清梯度。

[返回本章目录](README.md)
