## Flash Attention 原理

### 🎯 核心思想

Flash Attention 通过**分块计算 + Online Softmax**，避免存储完整的 N×N Attention 矩阵，大幅减少内存访问。

**关键 insight**：GPU 的瓶颈往往是内存带宽而非计算能力。

### 📝 伪代码实现

```python
def flash_attention_forward_naive(Q, K, V, block_size=64):
    """
    Flash Attention 前向传播（简化版伪代码）
    
    核心思想：
    1. 分块处理，避免存储完整的 N×N attention 矩阵
    2. 使用 Online Softmax 在块间增量更新
    
    标准 Attention 内存：O(N²) - 存储完整 attention 矩阵
    Flash Attention 内存：O(N) - 只存储输出和中间统计量
    """
    N, d = Q.shape
    
    # 输出和统计量初始化
    O = torch.zeros_like(Q)           # 输出
    L = torch.zeros(N, 1)             # logsumexp (softmax 分母的 log)
    M = torch.full((N, 1), float('-inf'))  # running max (数值稳定性)
    
    # 分块遍历 K, V
    for j in range(0, N, block_size):
        j_end = min(j + block_size, N)
        Kj = K[j:j_end]  # 当前 K 块
        Vj = V[j:j_end]  # 当前 V 块
        
        # 分块遍历 Q
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            Qi = Q[i:i_end]  # 当前 Q 块
            
            # === 在 SRAM 中计算（不写回 HBM）===
            
            # 1. 计算局部 attention scores
            Sij = Qi @ Kj.T  # [block_size, block_size]
            
            # 2. Online Softmax 更新
            # 找到新的 max
            Mij = Sij.max(dim=-1, keepdim=True).values
            M_new = torch.max(M[i:i_end], Mij)
            
            # 计算缩放因子
            exp_old = torch.exp(M[i:i_end] - M_new)  # 旧结果的缩放
            exp_new = torch.exp(Sij - M_new)         # 新块的 exp
            
            # 更新 logsumexp
            L_new = exp_old * L[i:i_end] + exp_new.sum(dim=-1, keepdim=True)
            
            # 3. 更新输出（增量更新）
            # 关键公式：旧输出要乘以旧的归一化因子 L_old，再除以新的 L_new
            O[i:i_end] = (exp_old * L[i:i_end] * O[i:i_end] + exp_new @ Vj) / L_new
            
            # 更新统计量
            M[i:i_end] = M_new
            L[i:i_end] = L_new
    
    return O


def explain_flash_attention_io():
    """Flash Attention IO 复杂度分析"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Flash Attention IO 分析                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  标准 Attention HBM 访问:                                    ║
    ║  1. 读 Q, K, V:        3 × N × d                            ║
    ║  2. 写 S = QK^T:       N × N        ← 瓶颈!                 ║
    ║  3. 读 S 做 softmax:   N × N                                ║
    ║  4. 写 P = softmax(S): N × N                                ║
    ║  5. 读 P, V:           2 × N × d                            ║
    ║  总计: O(N²) HBM 访问                                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Flash Attention HBM 访问:                                   ║
    ║  1. 读 Q, K, V: 3 × N × d                                   ║
    ║  2. 分块计算，中间结果 S, P 在 SRAM，不写 HBM                ║
    ║  3. 写输出 O:  N × d                                        ║
    ║  总计: O(N × d) HBM 访问                                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  加速比: 约 2-4x (取决于序列长度和 GPU)                      ║
    ║  内存节省: O(N²) → O(N)                                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
```

### 💡 面试追问

**Q: Flash Attention 为什么能加速？**

> 1. 减少 HBM 访问（GPU 内存带宽是瓶颈）
> 2. 分块计算充分利用 SRAM（L1/L2 cache，速度快 10-100 倍）
> 3. 不存储 N×N 的 attention 矩阵

**Q: Online Softmax 是什么？**

> 增量计算 softmax：维护 running max 和 running sum，每处理一个块就更新，无需等到看完所有数据。

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [scaled-dot-product-attention](scaled-dot-product-attention.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：为何除以 sqrt(d_k)？**

答：使独立分量点积的方差不随 d_k 线性增长，避免 softmax 过早饱和。

**问：GQA 怎样减少缓存？**

答：多个 Q 头共享一组 KV 头，KV cache 随 KV 头数下降。

[返回本章目录](README.md)
