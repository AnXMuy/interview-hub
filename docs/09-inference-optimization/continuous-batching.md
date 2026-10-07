## Continuous Batching

### 🎯 核心思想

序列完成后立即加入新请求，而不是等整个 batch 完成。

**问题**：静态 batching 中，短序列要等长序列完成
**解决**：动态管理 batch，序列独立调度

### 📝 实现代码

```python
class ContinuousBatchingScheduler:
    """
    Continuous Batching 调度器
    
    核心思想：
    - 每个序列独立跟踪状态
    - 序列完成后立即移出，新序列立即加入
    - 最大化 GPU 利用率
    """
    
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        
        # 活跃序列
        self.active_sequences = {}  # seq_id -> SequenceState
        
        # 等待队列
        self.waiting_queue = []
    
    def add_request(self, request):
        """添加新请求到等待队列"""
        self.waiting_queue.append(request)
    
    def schedule_step(self):
        """
        调度一个推理步骤
        
        Returns:
            batch: 本步骤要处理的序列
        """
        # 移除已完成的序列
        completed = [
            seq_id for seq_id, state in self.active_sequences.items()
            if state.is_finished()
        ]
        for seq_id in completed:
            del self.active_sequences[seq_id]
        
        # 从等待队列添加新序列（填满 batch）
        while len(self.active_sequences) < self.max_batch_size and self.waiting_queue:
            request = self.waiting_queue.pop(0)
            seq_id = self.create_sequence(request)
            self.active_sequences[seq_id] = SequenceState(request)
        
        # 返回当前 batch
        return list(self.active_sequences.values())
    
    def create_sequence(self, request):
        """创建新序列"""
        return id(request)  # 简化：用对象 id 作为 seq_id


class SequenceState:
    """序列状态"""
    
    def __init__(self, request):
        self.request = request
        self.generated_tokens = []
        self.max_tokens = request.get('max_tokens', 100)
    
    def is_finished(self):
        """检查是否完成"""
        if len(self.generated_tokens) >= self.max_tokens:
            return True
        if self.generated_tokens and self.generated_tokens[-1] == EOS_TOKEN:
            return True
        return False
    
    def add_token(self, token):
        """添加生成的 token"""
        self.generated_tokens.append(token)


EOS_TOKEN = 2  # 假设的结束符
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [kv-cache](kv-cache.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：KV Cache 缓存什么？**

答：缓存每层先前 token 的投影 K/V，不缓存 Q。

**问：Prefill 和 decode 瓶颈？**

答：前者偏计算密集，后者常受访存和缓存读取制约。

[返回本章目录](README.md)
