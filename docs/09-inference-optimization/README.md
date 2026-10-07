# ⚡ 推理优化

> 面试频率：🔥🔥🔥🔥🔥 | 难度：⭐⭐⭐⭐

推理优化决定 LLM 服务的成本和体验。KV Cache 是必考题！

## 本章速览

## 方法一览对比

> 💡 **一句话区分**：推理优化的核心目标是**提高吞吐量**和**降低延迟**

| 方法 | 解决问题 | 核心思想 | 效果 |
|:---|:---|:---|:---|
| **KV Cache** | 重复计算 | 缓存历史 K/V | 计算减少 n 倍 |
| **PagedAttention** | 显存碎片 | 分页管理显存 | 显存利用率提升 |
| **Speculative Decoding** | 生成太慢 | 小模型推测+大模型验证 | 加速 2-3x |
| **Continuous Batching** | batch 利用率低 | 动态插入新请求 | 吞吐量提升 2-5x |

```python
# 推理的两个阶段，优化点不同！

# 1. Prefill（首次，compute-bound）
# - 处理整个 prompt，生成第一个 token
# - 矩阵乘法多，GPU 利用率高
# - 优化: 并行、量化

# 2. Decode（递增，memory-bound）
# - 每次只生成 1 个 token，但要读取整个 KV Cache
# - 维度小，内存带宽是瓶颈
# - 优化: KV Cache、batching、speculative
```

> 🤔 **Q: 为什么 Decode 阶段是 memory-bound？**
>
> Decode 时，每次只生成 1 个 token（维度 [1, hidden]），
> 但要读取整个 KV Cache（维度 [seq_len, hidden]）。
>
> 计算量很小，但内存读写很大，所以是内存带宽瓶颈。

---

## 单题目录

- [KV Cache](kv-cache.md)（可测评）
- [PagedAttention](paged-attention.md)
- [Speculative Decoding](speculative-decoding.md)
- [Continuous Batching](continuous-batching.md)

## 高频面试问答

**问：KV Cache 缓存什么？**

答：缓存每层先前 token 的投影 K/V，不缓存 Q。

**问：Prefill 和 decode 瓶颈？**

答：前者偏计算密集，后者常受访存和缓存读取制约。

## 测评

本章练习：[kv-cache](kv-cache.md)。接口与运行命令见 [测评说明](../../grader/README.md)。
