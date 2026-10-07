## Batch Normalization

### 🎯 核心思想

对**每个特征在 batch 维度**进行归一化。主要用于 CNN，在 Transformer 中较少使用。

> 🤔 **Q: 为什么 BatchNorm 训练和推理表现不一致？**
>
> 训练时用当前 batch 的 mean/var，推理时用积累的 running_mean/running_var。
>
> 如果训练数据和推理数据分布不同，会有偏差。这也是 LLM 不用 BatchNorm 的原因之一。

### 📝 实现代码

```python
class BatchNorm1d(nn.Module):
    """
    Batch Normalization 手动实现
    
    对 batch 维度归一化（每个特征通道独立）
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        
        if track_running_stats:
            # running stats 不参与梯度计算
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_features] 或 [batch, num_features, length]
        """
        # 对于 3D 输入，先 flatten 为 2D 计算统计量
        original_shape = x.shape
        if x.dim() == 3:
            # [batch, features, length] -> [batch * length, features]
            x = x.permute(0, 2, 1).reshape(-1, self.num_features)
            reshape_back = True
        else:
            reshape_back = False
        
        if self.training:
            # 训练时：使用当前 batch 的统计量
            # x: [N, features]，对 dim=0 取均值得 [features]
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新 running stats（指数移动平均）
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
        else:
            # 推理时：使用 running stats
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias
        
        if reshape_back:
            # [batch * length, features] -> [batch, features, length]
            batch_size, _, length = original_shape
            x_normalized = x_normalized.reshape(batch_size, length, -1).permute(0, 2, 1)
        
        return x_normalized
```

### 💡 BatchNorm vs LayerNorm

| 特性 | BatchNorm | LayerNorm |
|:---|:---|:---|
| 归一化维度 | Batch 维度 | 特征维度 |
| 依赖 batch size | 是（需要足够大） | 否 |
| 训练/推理一致性 | 不同（running stats） | 相同 |
| 适用场景 | CNN | Transformer, RNN |
| 序列长度变化 | 需要特殊处理 | 天然支持 |

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [rms-norm](rms-norm.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：LN 与 RMSNorm 有何差别？**

答：RMSNorm 不减均值，只除均方根；均沿特征维归一化。

**问：为什么小 batch 不利于 BN？**

答：统计量由 batch 估计，样本少时方差较大。

[返回本章目录](README.md)
