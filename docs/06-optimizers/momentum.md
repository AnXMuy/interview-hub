## SGD with Momentum

### 🎯 核心思想

引入"动量"，积累历史梯度方向，加速收敛并减少震荡。

$$v_t = \gamma v_{t-1} + \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

**物理直觉**：像小球滚下山坡，有惯性。

### 📝 实现代码

```python
class SGDMomentum:
    """
    SGD with Momentum
    
    动量帮助：
    1. 加速：在一致方向上累积速度
    2. 减震：减少在局部最小值附近的震荡
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = False
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        
        # 初始化速度（每个参数一个）
        self.velocities = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                v = self.velocities[i]
                
                # 更新速度: v = momentum * v + grad
                v.mul_(self.momentum).add_(grad)
                
                if self.nesterov:
                    # Nesterov: 先"预测"位置，再计算梯度
                    # θ = θ - lr * (momentum * v + grad)
                    param -= self.lr * (self.momentum * v + grad)
                else:
                    # 标准 Momentum
                    param -= self.lr * v
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [adamw](adamw.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：AdamW 的衰减为什么要解耦？**

答：使权重衰减不经过自适应梯度缩放。

**问：偏差修正的目的？**

答：修正从零初始化的动量在初始步骤偏向零的问题。

[返回本章目录](README.md)
