## Temperature Sampling

### 🎯 核心思想

Temperature 控制输出分布的"锐度"：
- **T → 0**: 分布趋近 one-hot（贪婪）
- **T = 1**: 原始分布
- **T → ∞**: 趋近均匀分布（完全随机）

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

### 📝 实现代码

```python
import torch
import torch.nn.functional as F

def temperature_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Temperature Sampling
    
    Args:
        logits: [batch, vocab_size] 或 [vocab_size] 模型输出的 logits
        temperature: 温度参数
            - T < 1: 更确定（分布更尖锐）
            - T = 1: 原始分布
            - T > 1: 更随机（分布更平滑）
    
    Returns:
        sampled_token: 采样得到的 token id
        probs: softmax 后的概率分布
    """
    if temperature == 0:
        # 温度为 0 时退化为 argmax（贪婪解码）
        sampled_token = logits.argmax(dim=-1)
        probs = F.one_hot(sampled_token, logits.size(-1)).float()
        return sampled_token, probs
    
    # Step 1: 温度缩放
    scaled_logits = logits / temperature
    
    # Step 2: 数值稳定的 softmax
    # 减去最大值防止 exp 溢出
    scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True)[0]
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Step 3: 多项式采样
    sampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return sampled_token, probs


# ==================== 可视化温度效果 ====================
def visualize_temperature_effect():
    """展示不同温度对分布的影响"""
    import matplotlib.pyplot as plt
    
    # 模拟 logits（假设有 10 个候选词）
    logits = torch.tensor([3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0])
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    plt.figure(figsize=(12, 6))
    for T in temperatures:
        probs = F.softmax(logits / T, dim=-1)
        plt.plot(probs.numpy(), label=f'T={T}', marker='o')
    
    plt.xlabel('Token Index')
    plt.ylabel('Probability')
    plt.title('Temperature Effect on Probability Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('temperature_effect.png')
    print("已保存: temperature_effect.png")


if __name__ == "__main__":
    visualize_temperature_effect()
```

### 💡 面试追问

**Q: Temperature 为什么能控制多样性？**

> 除以 T 相当于放大/缩小 logits 的差异。T 小时差异放大，softmax 后高概率更高；T 大时差异缩小，分布更均匀。

> 🤔 **Q: Temperature=0 和 Greedy 有什么区别？**
>
> 完全一样！当 T→0 时，`softmax(logits/T)` 趋近于 one-hot，采样结果就是 argmax。
>
> 所以代码中通常特判 `if temperature == 0: return argmax(logits)`

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

[返回本章目录](README.md)
