## Mixed Precision Training

### 🎯 核心思想

使用 FP16/BF16 加速计算，用 FP32 保证精度：
- **前向/反向**: FP16（快、省内存）
- **参数更新**: FP32（保证精度）
- **Loss Scaling**: 防止 FP16 梯度下溢

### 📝 实现代码

```python
class MixedPrecisionTrainer:
    """
    混合精度训练
    
    核心组件：
    1. GradScaler: 梯度缩放，防止 FP16 下溢
    2. autocast: 自动选择 FP16/FP32
    """
    
    def __init__(self, model, optimizer, init_scale=65536.0):
        self.model = model
        self.optimizer = optimizer
        
        # 梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
    
    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        
        # 使用 autocast 自动混合精度
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
        
        # 反向传播（使用缩放后的 loss）
        self.scaler.scale(loss).backward()
        
        # 先 unscale 梯度，然后裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数（scaler 会检查是否有 inf/nan）
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()


class GradScaler:
    """
    手动实现 Gradient Scaler
    
    为什么需要？
    - FP16 的最小正数是 ~6e-8
    - 深度网络的梯度可能小于这个值，导致下溢为 0
    - 缩放 loss 使梯度变大，更新时再缩放回来
    """
    
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        
        self.step_count = 0
        self.inf_count = 0
    
    def scale_loss(self, loss):
        """缩放 loss"""
        return loss * self.scale
    
    def unscale_grads(self, optimizer):
        """反缩放梯度"""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def step(self, optimizer):
        """
        执行优化器 step，并处理 inf/nan
        """
        # 检查是否有 inf/nan 梯度
        has_inf = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        has_inf = True
                        break
        
        if has_inf:
            # 跳过这次更新，减小 scale
            self.inf_count += 1
            self.scale *= self.backoff_factor
            print(f"发现 inf/nan，scale 降低到 {self.scale}")
            return False
        else:
            # 正常更新
            optimizer.step()
            self.step_count += 1
            
            # 定期增大 scale
            if self.step_count % self.growth_interval == 0:
                self.scale *= self.growth_factor
            
            return True
    
    def update(self):
        """更新内部状态"""
        pass  # 在 step 中已处理
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
