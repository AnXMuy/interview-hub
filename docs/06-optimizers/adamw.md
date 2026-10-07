## AdamW

### 🎯 核心思想

AdamW 修复了 Adam 中 weight decay 的实现问题：

- **Adam**: `grad = grad + λ * w`，然后用 adaptive lr
- **AdamW**: `w = w - λ * w`，直接从权重中减去

**区别**：AdamW 的 weight decay 不会被 adaptive learning rate 缩放。

### 📝 实现代码

```python
class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay
    
    与 Adam 的区别：weight decay 直接作用于权重，
    而不是加到梯度上后被 adaptive lr 缩放。
    
    这是 LLM 训练的标准优化器！
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # ========== AdamW 的关键：weight decay 独立于 Adam ==========
                # 先应用 weight decay（不经过 Adam 的处理）
                if self.weight_decay != 0:
                    param.mul_(1 - self.lr * self.weight_decay)
                
                # 然后正常的 Adam 更新
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# Adam vs AdamW 对比
def compare_adam_adamw():
    """展示 Adam 和 AdamW 的区别"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║               Adam vs AdamW 区别                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Adam with L2:                                               ║
    ║    grad_new = grad + λ * w                                   ║
    ║    w = w - lr * adaptive(grad_new)                           ║
    ║    问题: L2 惩罚被 adaptive lr 缩放了！                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  AdamW (Decoupled):                                          ║
    ║    w = w - lr * λ * w  (直接 decay)                         ║
    ║    w = w - lr * adaptive(grad)                               ║
    ║    正确: weight decay 独立于 adaptive lr                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  为什么重要？                                                ║
    ║  - 高学习率的参数会获得更大的 adaptive scaling               ║
    ║  - Adam 中这些参数的 L2 惩罚也被放大，导致过度正则化         ║
    ║  - AdamW 确保所有参数受到一致的 weight decay                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。网页保存后可直接运行本模块的本地测评。

## 常见面试问答

**问：AdamW 的衰减为什么要解耦？**

答：使权重衰减不经过自适应梯度缩放。

**问：偏差修正的目的？**

答：修正从零初始化的动量在初始步骤偏向零的问题。

[返回本章目录](README.md)
