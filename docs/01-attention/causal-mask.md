## Causal Mask

### 🎯 核心思想

因果掩码确保每个位置**只能看到自己和之前的位置**，这是自回归生成的基础。

### 📝 实现代码

```python
def create_causal_mask(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    创建因果掩码（下三角矩阵）
    
    返回的 mask 中:
    - 1 表示可以看到（保留）
    - 0 表示不能看到（mask 掉）
    
    示例 (seq_len=4):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    """
    # 方法1: 使用 torch.tril（推荐）
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    
    return mask


def create_causal_mask_additive(
    seq_len: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    创建加性因果掩码（直接加到 attention scores 上）
    
    返回的 mask 中:
    - 0 表示可以看到
    - -inf 表示不能看到
    
    这种形式可以直接和 scores 相加: scores = scores + mask
    """
    # 上三角为 -inf，其他为 0
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device) * float('-inf'),
        diagonal=1  # 从对角线上方一格开始
    ).to(dtype)
    
    return mask


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    滑动窗口掩码（Longformer/Mistral 使用）
    
    每个位置只能看到前 window_size 个位置
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 1
    
    return mask


# ==================== 可视化 ====================
def visualize_mask(mask: torch.Tensor, title: str = "Attention Mask"):
    """打印 mask 矩阵"""
    print(f"\n{title}:")
    print(mask.int().numpy())


if __name__ == "__main__":
    # 因果掩码
    causal = create_causal_mask(5)
    visualize_mask(causal, "Causal Mask")
    
    # 滑动窗口掩码
    sliding = create_sliding_window_mask(5, window_size=3)
    visualize_mask(sliding, "Sliding Window Mask (window=3)")
```

输出示例:
```
Causal Mask:
[[1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [1 1 1 1 0]
 [1 1 1 1 1]]

Sliding Window Mask (window=3):
[[1 0 0 0 0]
 [1 1 0 0 0]
 [1 1 1 0 0]
 [0 1 1 1 0]
 [0 0 1 1 1]]
```

> 🤔 **Q: mask 为什么用 0 和 1，不直接用 -inf？**
>
> 两种风格都可以！上面代码用 0/1 是布尔掩码风格（`mask==0` 的位置填 -inf）。
>
> 也可以用加性掩码：`scores = scores + mask`，其中 mask 直接是 0 或 -inf。两种等价，看团队习惯。

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
