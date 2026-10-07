## Learning Rate Scheduler

### 🎯 核心思想

学习率调度：训练过程中动态调整学习率。

常见策略：
- **Warmup**: 开始时从小学习率逐渐增大
- **Cosine Decay**: 余弦曲线下降
- **Linear Decay**: 线性下降

> 🤔 **Q: 为什么需要 Warmup？不 warmup 会怎样？**
>
> 初始时参数随机，梯度方向不稳定。如果直接用大学习率：
> 1. 参数更新幅度大，可能跳过好的局部最小值
> 2. Adam 的 m 和 v 还没积累好，估计不稳定
> 3. BatchNorm/LayerNorm 的统计量还不准
>
> Warmup 让模型先"走稳"，再"跑快"。

### 📝 实现代码

```python
import math

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing
    
    LLM 训练的标准学习率调度
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        # 获取初始学习率
        self.base_lr = optimizer.lr if hasattr(optimizer, 'lr') else optimizer.param_groups[0]['lr']
        
        self.current_step = 0
    
    def get_lr(self) -> float:
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # Warmup: 线性增加
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            # Cosine: 从 1 下降到 0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self.get_lr()
        
        # 更新 optimizer 的学习率
        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        return lr


class WarmupLinearScheduler:
    """
    Warmup + Linear Decay
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # Linear decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            return self.base_lr - (self.base_lr - self.min_lr) * progress
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ==================== 可视化 ====================
def visualize_schedulers():
    """可视化学习率调度"""
    import matplotlib.pyplot as plt
    
    total_steps = 10000
    warmup_steps = 1000
    base_lr = 1e-4
    
    # 模拟
    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': base_lr}]
    
    opt = FakeOptimizer()
    
    schedulers = {
        'Cosine': WarmupCosineScheduler(opt, warmup_steps, total_steps),
        'Linear': WarmupLinearScheduler(opt, warmup_steps, total_steps),
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, scheduler in schedulers.items():
        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_lr())
            scheduler.step()
        
        plt.plot(lrs, label=name)
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers')
    plt.legend()
    plt.axvline(x=warmup_steps, color='gray', linestyle='--', label='Warmup End')
    plt.savefig('lr_schedulers.png')
    print("已保存: lr_schedulers.png")


if __name__ == "__main__":
    visualize_schedulers()
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：AdamW 的衰减为什么要解耦？**

答：使权重衰减不经过自适应梯度缩放。

**问：偏差修正的目的？**

答：修正从零初始化的动量在初始步骤偏向零的问题。

[返回本章目录](README.md)
