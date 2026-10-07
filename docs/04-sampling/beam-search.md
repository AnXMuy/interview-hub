## Beam Search

### 🎯 核心思想

维护 k 个最优候选序列，每步扩展所有候选，保留得分最高的 k 个。

> 🤔 **Q: Beam Search 和 Greedy 有什么区别？为什么不总是用 Beam？**
>
> Greedy 是 Beam width=1 的特例。Beam 可以纠正局部次优选择。
>
> 但 Beam Search 有问题：
> 1. 计算量大（每步维护 k 个序列）
> 2. 倾向于生成短句（短序列分数更高）
> 3. 缺乏多样性（往往生成 "safe" 但无聊的句子）
>
> 所以现代 LLM 对话多用 Sampling，翻译等需要稳定输出时才用 Beam。

### 📝 实现代码

```python
def beam_search(
    model,
    input_ids: torch.Tensor,
    beam_width: int = 4,
    max_length: int = 50,
    eos_token_id: int = 2,
    length_penalty: float = 1.0
) -> torch.Tensor:
    """
    Beam Search 解码
    
    Args:
        model: 语言模型
        input_ids: [batch, seq_len] 输入序列（通常 batch=1）
        beam_width: beam 宽度
        max_length: 最大生成长度
        eos_token_id: 结束符 ID
        length_penalty: 长度惩罚（> 1 鼓励长句，< 1 鼓励短句）
    
    Returns:
        best_sequence: [batch, seq_len] 最优序列
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 复制 input_ids 到所有 beam
    # [batch, seq] -> [batch * beam, seq]
    beam_input_ids = input_ids.unsqueeze(1).repeat(1, beam_width, 1)
    beam_input_ids = beam_input_ids.view(batch_size * beam_width, -1)
    
    # 初始化 beam 分数
    # [batch, beam]
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_scores[:, 1:] = float('-inf')  # 初始只有第一个 beam 是活的
    
    # 完成的序列
    done = [False] * batch_size
    
    for step in range(max_length):
        # 获取模型输出
        with torch.no_grad():
            outputs = model(beam_input_ids)
            # 取最后一个 token 的 logits
            next_token_logits = outputs[:, -1, :]  # [batch*beam, vocab]
        
        vocab_size = next_token_logits.size(-1)
        
        # 转换为 log prob
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        
        # 重塑: [batch*beam, vocab] -> [batch, beam, vocab]
        next_token_log_probs = next_token_log_probs.view(batch_size, beam_width, -1)
        
        # 计算新的分数: [batch, beam, 1] + [batch, beam, vocab]
        # -> [batch, beam, vocab]
        next_scores = beam_scores.unsqueeze(-1) + next_token_log_probs
        
        # 重塑为 [batch, beam * vocab]
        next_scores = next_scores.view(batch_size, -1)
        
        # 选择 top beam_width 个
        # [batch, beam]
        next_scores, next_tokens = torch.topk(next_scores, beam_width, dim=-1)
        
        # 解码: 哪个 beam + 哪个 token
        beam_indices = next_tokens // vocab_size  # [batch, beam]
        token_indices = next_tokens % vocab_size  # [batch, beam]
        
        # 构建新的序列
        # 从对应的旧 beam 复制，然后添加新 token
        new_beam_input_ids = []
        for batch_idx in range(batch_size):
            batch_beams = []
            for beam_idx in range(beam_width):
                # 找到源 beam
                source_beam = beam_indices[batch_idx, beam_idx]
                source_idx = batch_idx * beam_width + source_beam
                
                # 新 token
                new_token = token_indices[batch_idx, beam_idx]
                
                # 拼接
                new_seq = torch.cat([
                    beam_input_ids[source_idx],
                    new_token.unsqueeze(0)
                ])
                batch_beams.append(new_seq)
            
            new_beam_input_ids.append(torch.stack(batch_beams))
        
        beam_input_ids = torch.cat(new_beam_input_ids, dim=0)
        beam_scores = next_scores
        
        # 检查 EOS
        # 简化：这里只检查第一个 batch
        if (token_indices[0] == eos_token_id).any():
            break
    
    # 应用长度惩罚
    seq_len = beam_input_ids.size(1)
    length_penalty_factor = ((5 + seq_len) / 6) ** length_penalty
    final_scores = beam_scores / length_penalty_factor
    
    # 选择最优 beam
    best_beam_idx = final_scores.argmax(dim=-1)  # [batch]
    
    # 提取最优序列
    best_sequences = []
    for batch_idx in range(batch_size):
        idx = batch_idx * beam_width + best_beam_idx[batch_idx]
        best_sequences.append(beam_input_ids[idx])
    
    return torch.stack(best_sequences)


# ==================== 简化版 Beam Search ====================
def beam_search_simple(
    get_next_probs,  # 函数: input_ids -> log_probs
    start_token: int,
    beam_width: int = 3,
    max_length: int = 10,
    eos_token: int = 2
):
    """
    简化版 Beam Search（面试手写）
    
    返回 (最优序列, 分数)
    """
    # 初始化: [(序列, 分数)]
    beams = [([start_token], 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == eos_token:
                # 已结束的序列直接保留
                all_candidates.append((seq, score))
                continue
            
            # 获取下一个 token 的 log prob
            log_probs = get_next_probs(seq)  # [vocab_size]
            
            # 扩展所有可能的下一个 token
            for token_id in range(len(log_probs)):
                new_seq = seq + [token_id]
                new_score = score + log_probs[token_id]
                all_candidates.append((new_seq, new_score))
        
        # 保留 top beam_width
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
        
        # 如果所有 beam 都结束了
        if all(seq[-1] == eos_token for seq, _ in beams):
            break
    
    # 返回最优
    return beams[0]
```

---

## 动手练习

本节先用原文示例复现概念；本章的自动测评题在 [top-p](top-p.md)，只需实现该题对应的一个接口。

## 常见面试问答

**问：top-k 与 top-p 区别？**

答：前者保留固定数量，后者保留覆盖累计概率阈值的最小集合。

**问：温度等于零如何处理？**

答：走贪心 argmax，不要做除零。

[返回本章目录](README.md)
