## Adam

### 🎯 核心思想

Adam = Adaptive Moment Estimation，结合了：
- **Momentum**（一阶矩：梯度均值）
- **RMSprop**（二阶矩：梯度方差）

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(一阶矩)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(二阶矩)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(偏差修正)}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

> 🤔 **Q: 为什么除以 √v 而不是直接用 v？**
>
> v 是梯度平方的均值，量纲是 "梯度²"。
>
> 我们要除的是梯度的"幅度"，所以要开根号得到"梯度”量纲。
>
> 这样 m/√v 的量纲是“梯度/梯度=无量纲”，再乘 lr 才正确。

### 📝 实现代码

```python
class Adam:
    """
    Adam Optimizer 🔥 面试必考
    
    核心思想：
    1. m: 梯度的指数移动平均（一阶矩，类似 momentum）
    2. v: 梯度平方的指数移动平均（二阶矩，自适应学习率）
    3. 偏差修正：修复初始阶段 m, v 偏向 0 的问题
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay  # L2 正则化
        
        self.t = 0  # 时间步
        
        # 初始化一阶和二阶矩
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        self.t += 1
        
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                
                grad = param.grad
                
                # L2 正则化（注意：这是 Adam 的方式，不是 AdamW）
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * param
                
                # ========== Step 1: 更新一阶矩（梯度均值）==========
                # m = β1 * m + (1 - β1) * grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                
                # ========== Step 2: 更新二阶矩（梯度方差）==========
                # v = β2 * v + (1 - β2) * grad^2
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
                
                # ========== Step 3: 偏差修正 ==========
                # 为什么需要？初始时 m, v 接近 0，估计偏低
                # 例如 t=1 时，m = (1-β1)*g1，期望应该是 g1
                # 除以 (1-β1^t) 修正这个偏差
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # ========== Step 4: 更新参数 ==========
                # θ = θ - lr * m_hat / (sqrt(v_hat) + eps)
                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ==================== 验证与 PyTorch 对比 ====================
def test_adam():
    torch.manual_seed(42)
    
    # 创建两份相同的参数
    w1 = torch.randn(3, 3, requires_grad=True)
    w2 = w1.clone().detach().requires_grad_(True)
    
    # 我的 Adam
    my_adam = Adam([w1], lr=0.01)
    
    # PyTorch Adam
    torch_adam = torch.optim.Adam([w2], lr=0.01)
    
    # 训练几步
    for step in range(5):
        # 计算相同的 loss
        loss1 = (w1 ** 2).sum()
        loss2 = (w2 ** 2).sum()
        
        # 反向传播
        my_adam.zero_grad()
        loss1.backward()
        my_adam.step()
        
        torch_adam.zero_grad()
        loss2.backward()
        torch_adam.step()
        
        diff = (w1 - w2).abs().max().item()
        print(f"Step {step}: 参数差异 = {diff:.8f}")
    
    assert diff < 1e-6, "实现有误!"
    print("✅ Adam 测试通过!")


if __name__ == "__main__":
    test_adam()
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
