## PagedAttention

### 🎯 核心思想

借鉴操作系统的虚拟内存：将 KV Cache 分成固定大小的"页"，按需分配。

**问题**：预分配连续内存浪费（max_seq_len 通常用不满）
**解决**：分页管理，只分配实际需要的页

### 📝 实现代码

```python
class PagedKVCache:
    """
    Paged KV Cache (vLLM 核心技术)
    
    将 KV Cache 分成固定大小的页，按需分配
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,  # 每页存储的 token 数
        max_pages: int = 1000,  # 最大页数
        dtype: torch.dtype = torch.float16,
        device: torch.device = None
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.device = device
        
        # 页表：page_id -> (k_page, v_page)
        # 实际实现中会预分配所有页，这里简化
        self.pages = {}
        
        # 空闲页列表
        self.free_pages = list(range(max_pages))
        
        # 每个序列的页表：seq_id -> [page_ids]
        self.seq_page_table = {}
    
    def allocate_page(self) -> int:
        """分配一个新页"""
        if not self.free_pages:
            raise RuntimeError("Out of pages!")
        
        page_id = self.free_pages.pop(0)
        
        # 分配实际内存
        self.pages[page_id] = {
            'k': torch.zeros(
                self.num_layers, self.num_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device
            ),
            'v': torch.zeros(
                self.num_layers, self.num_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device
            ),
            'used_slots': 0
        }
        
        return page_id
    
    def free_page(self, page_id: int):
        """释放页"""
        if page_id in self.pages:
            del self.pages[page_id]
            self.free_pages.append(page_id)
    
    def append_token(self, seq_id: int, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        为序列追加一个 token 的 KV
        
        k, v: [num_heads, 1, head_dim]
        """
        if seq_id not in self.seq_page_table:
            self.seq_page_table[seq_id] = []
        
        page_ids = self.seq_page_table[seq_id]
        
        # 检查是否需要新页
        if not page_ids or self.pages[page_ids[-1]]['used_slots'] >= self.page_size:
            new_page_id = self.allocate_page()
            page_ids.append(new_page_id)
        
        # 写入最后一页
        current_page_id = page_ids[-1]
        slot_idx = self.pages[current_page_id]['used_slots']
        
        self.pages[current_page_id]['k'][layer_idx, :, slot_idx, :] = k.squeeze(1)
        self.pages[current_page_id]['v'][layer_idx, :, slot_idx, :] = v.squeeze(1)
        self.pages[current_page_id]['used_slots'] += 1
    
    def get_kv(self, seq_id: int, layer_idx: int) -> tuple:
        """
        获取序列的完整 KV
        
        Returns:
            k: [num_heads, seq_len, head_dim]
            v: [num_heads, seq_len, head_dim]
        """
        if seq_id not in self.seq_page_table:
            return None, None
        
        page_ids = self.seq_page_table[seq_id]
        k_list, v_list = [], []
        
        for page_id in page_ids:
            page = self.pages[page_id]
            used = page['used_slots']
            
            k_list.append(page['k'][layer_idx, :, :used, :])
            v_list.append(page['v'][layer_idx, :, :used, :])
        
        k = torch.cat(k_list, dim=1)  # [heads, total_len, head_dim]
        v = torch.cat(v_list, dim=1)
        
        return k, v
    
    def free_sequence(self, seq_id: int):
        """释放序列的所有页"""
        if seq_id in self.seq_page_table:
            for page_id in self.seq_page_table[seq_id]:
                self.free_page(page_id)
            del self.seq_page_table[seq_id]


def paged_attention_advantages():
    """PagedAttention 的优势"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              PagedAttention 优势                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  1. 内存利用率提升                                           ║
    ║     - 传统: 预分配 max_seq_len，浪费 50-90%                  ║
    ║     - Paged: 按需分配，接近 0 浪费                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  2. 支持更大的 batch size                                    ║
    ║     - 相同内存下，batch size 可增加 2-4x                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  3. 灵活的内存共享                                           ║
    ║     - Beam Search: 多个候选共享相同 prefix 的页              ║
    ║     - Parallel Sampling: 共享 prompt 的 KV                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  4. 高效的内存回收                                           ║
    ║     - 序列结束后立即释放页                                   ║
    ║     - 不需要等待整个 batch 完成                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
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
