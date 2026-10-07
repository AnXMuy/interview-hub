## 残差连接与层归一化

### 🎯 Post-Norm vs Pre-Norm

```python
def post_norm_block(x, sublayer, norm):
    """
    Post-Norm (原始 Transformer)
    
    x -> Sublayer -> Add -> Norm
    
    问题: 深层网络梯度不稳定
    """
    return norm(x + sublayer(x))


def pre_norm_block(x, sublayer, norm):
    """
    Pre-Norm (现代 LLM)
    
    x -> Norm -> Sublayer -> Add
    
    优势:
    1. 梯度直通路径 (residual 不经过 norm)
    2. 训练更稳定
    3. 不需要 warmup
    """
    return x + sublayer(norm(x))


# 梯度流分析
def analyze_gradient_flow():
    """
    Pre-Norm 的梯度流分析
    """
    print("""
    Post-Norm 梯度流:
    ∂L/∂x = ∂L/∂y · ∂Norm/∂(x + sublayer(x))
    → 梯度必须经过 Norm，可能被缩放或消失
    
    Pre-Norm 梯度流:
    ∂L/∂x = ∂L/∂y · (∂x/∂x + ∂sublayer(Norm(x))/∂x)
          = ∂L/∂y · (1 + ...)
    → 有恒等路径，梯度可以直接回传
    
    这就是为什么 Pre-Norm 更稳定！
    """)
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：为什么 FFN 在 token 维度独立？**

答：同一组参数对每个 token 的最后一维逐位置变换。

**问：Pre-Norm 何时归一化？**

答：在注意力/FFN 子层之前归一化，再经过残差相加。

[返回本章目录](README.md)
