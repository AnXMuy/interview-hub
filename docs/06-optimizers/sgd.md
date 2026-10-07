## SGD

### 🎯 核心思想

最基础的梯度下降：沿着梯度反方向更新参数。

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$$

### 📝 实现代码

```python
import torch
import torch.nn as nn

class SGD:
    """
    Stochastic Gradient Descent
    """
    
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        """执行一步参数更新"""
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    # θ = θ - lr * grad
                    param -= self.lr * param.grad
    
    def zero_grad(self):
        """清空梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ==================== 使用示例 ====================
def test_sgd():
    # 简单线性回归
    torch.manual_seed(42)
    X = torch.randn(100, 2)
    y = X @ torch.tensor([2.0, 3.0]) + 1.0 + torch.randn(100) * 0.1
    
    # 模型参数
    w = torch.randn(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    optimizer = SGD([w, b], lr=0.1)
    
    for epoch in range(100):
        # 前向
        y_pred = X @ w + b
        loss = ((y_pred - y) ** 2).mean()
        
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print(f"学到的权重: w={w.data}, b={b.data}")


if __name__ == "__main__":
    test_sgd()
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
