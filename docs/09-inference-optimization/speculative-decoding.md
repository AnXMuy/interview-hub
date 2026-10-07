## Speculative Decoding

### 🎯 核心思想

用小模型"猜"多个 token，大模型验证。如果猜对了，相当于一次前向生成多个 token。

**加速比**：理论上可达 2-3x（取决于小模型的准确率）

### 📝 实现代码

```python
def speculative_decode(
    draft_model,      # 小模型（快但不那么准）
    target_model,     # 大模型（慢但准）
    input_ids: torch.Tensor,
    gamma: int = 4,   # 每次猜几个 token
    max_new_tokens: int = 100
):
    """
    Speculative Decoding 🔥🔥
    
    原理：
    1. 用小模型快速生成 γ 个 token（草稿）
    2. 用大模型一次性验证这 γ 个 token
    3. 接受部分或全部草稿，拒绝的位置从大模型重采样
    
    为什么快？
    - 大模型的瓶颈是内存带宽而非计算
    - 验证 γ 个 token 和生成 1 个 token 耗时接近
    - 如果草稿全部正确，一次前向 = γ 个 token
    """
    device = input_ids.device
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens // gamma):
        # ========== Step 1: 小模型生成草稿 ==========
        draft_tokens = []
        draft_probs = []
        
        draft_input = generated.clone()
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(token)
                draft_probs.append(probs)
                draft_input = torch.cat([draft_input, token], dim=-1)
        
        draft_tokens = torch.cat(draft_tokens, dim=-1)  # [batch, gamma]
        
        # ========== Step 2: 大模型验证 ==========
        # 一次性计算所有位置的分布
        verify_input = torch.cat([generated, draft_tokens], dim=-1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # 取对应位置的分布
        # target_logits[:, generated_len-1] 预测第一个草稿位置
        # target_logits[:, generated_len] 预测第二个草稿位置
        # ...
        start_pos = generated.size(1) - 1
        target_probs_list = [
            F.softmax(target_logits[:, start_pos + i, :], dim=-1)
            for i in range(gamma + 1)
        ]
        
        # ========== Step 3: 接受/拒绝 ==========
        accepted_count = 0
        
        for i in range(gamma):
            draft_token = draft_tokens[:, i]
            
            # 获取草稿和目标分布
            p_draft = draft_probs[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            p_target = target_probs_list[i].gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            
            # 接受概率: min(1, p_target / p_draft)
            accept_prob = torch.clamp(p_target / p_draft, max=1.0)
            
            if torch.rand(1, device=device).item() < accept_prob.item():
                # 接受，添加这个 token
                generated = torch.cat([generated, draft_token.unsqueeze(1)], dim=-1)
                accepted_count += 1
            else:
                # 拒绝，从调整后的分布采样
                # 调整分布: max(0, p_target - p_draft) 然后归一化
                adjusted_probs = torch.clamp(
                    target_probs_list[i] - draft_probs[i],
                    min=0
                )
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True)
                
                new_token = torch.multinomial(adjusted_probs, num_samples=1)
                generated = torch.cat([generated, new_token], dim=-1)
                break
        
        # 如果全部接受，从大模型最后的分布额外采样一个
        if accepted_count == gamma:
            bonus_token = torch.multinomial(target_probs_list[-1], num_samples=1)
            generated = torch.cat([generated, bonus_token], dim=-1)
    
    return generated


def analyze_speculative_speedup(draft_accuracy: float, gamma: int):
    """分析 Speculative Decoding 的加速比"""
    
    # 期望每次验证接受的 token 数
    # 如果每个草稿独立，期望 = sum(accuracy^i for i in 1..gamma)
    expected_accepted = sum(draft_accuracy ** i for i in range(1, gamma + 1))
    
    # 考虑全部正确时的 bonus token
    expected_with_bonus = expected_accepted + draft_accuracy ** gamma
    
    # 加速比 = 期望生成的 token 数 / 大模型前向次数
    speedup = expected_with_bonus / 1  # 每次大模型只前向一次
    
    print(f"=== Speculative Decoding 加速分析 ===")
    print(f"草稿模型准确率: {draft_accuracy:.0%}")
    print(f"每次猜测数量 γ: {gamma}")
    print(f"期望接受的 token: {expected_accepted:.2f}")
    print(f"期望生成的 token (含 bonus): {expected_with_bonus:.2f}")
    print(f"理论加速比: {speedup:.2f}x")


if __name__ == "__main__":
    analyze_speculative_speedup(draft_accuracy=0.8, gamma=4)
```

---

## 动手练习

本模块的具体实现题会显示在网页右侧；题面明确函数或类接口、输入、输出、边界条件和具体任务。

## 常见面试问答

**问：KV Cache 缓存什么？**

答：缓存每层先前 token 的投影 K/V，不缓存 Q。

**问：Prefill 和 decode 瓶颈？**

答：前者偏计算密集，后者常受访存和缓存读取制约。

[返回本章目录](README.md)
