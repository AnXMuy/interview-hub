"""Interview-style implementation contracts for every Markdown module."""

CORE = {
    '00-torch-basics/torch-basics': {
        'name': '实现 masked_mean', 'signature': 'masked_mean(x, mask) -> Tensor',
        'input': 'x: [B,T,D] 浮点张量；mask: [B,T] bool，True 表示有效 token。',
        'output': '返回 [B,D]；每行只对有效 token 求均值，全 False 行返回零向量。',
        'tasks': ['正确广播 mask 并沿 T 维归约', '处理全屏蔽样本且不产生 NaN', '保留 autograd，不能 detach 或转 NumPy'],
    },
    '01-attention/scaled-dot-product-attention': {
        'name': '实现缩放点积注意力', 'signature': 'scaled_dot_product_attention(q, k, v, mask=None) -> Tensor',
        'input': 'q/k/v: [B,H,T,D]；mask 可广播到 [B,H,Tq,Tk]，True 表示可见。',
        'output': '返回 [B,H,Tq,Dv]，计算 softmax(qk^T / sqrt(D))v。',
        'tasks': ['支持 Tq 与 Tk 不相等', '正确处理 None 和广播 mask', '屏蔽位置不能参与 softmax'],
    },
    '02-normalization/rms-norm': {
        'name': '实现 RMSNorm', 'signature': 'class RMSNorm(nn.Module): RMSNorm(dim, eps=1e-6)',
        'input': '输入 x 的最后一维大小为 dim，支持任意前置 batch 维。',
        'output': '按最后一维做 RMS 归一化并乘可训练 weight，输出形状与 x 相同。',
        'tasks': ['初始化 weight 为全 1', '处理 eps 防止除零', '输入和 weight 都必须可反向传播'],
    },
    '03-position-encoding/rope': {
        'name': '实现 RoPE 旋转', 'signature': 'apply_rope(x, offset=0, base=10000.0) -> Tensor',
        'input': 'x: [B,H,T,D]，D 为偶数；offset 为当前 cache 起始位置。',
        'output': '返回同形状张量，对相邻维度 (0,1)、(2,3) 按位置旋转。',
        'tasks': ['使用偶数/奇数维配对', '支持 offset 位置偏移', '保持 dtype、device 和梯度'],
    },
    '04-sampling/top-p': {
        'name': '实现 Top-p 过滤', 'signature': 'top_p_filter(logits, p) -> Tensor',
        'input': 'logits: [B,V]；p 在 (0,1] 内，沿词表维计算 softmax 累积概率。',
        'output': '保留达到 nucleus 概率所需的最小 token 前缀，其余位置写为 -inf。',
        'tasks': ['按 logits 降序排序后累计概率', '至少保留一个 token', '保留原始词表顺序和输入形状'],
    },
    '05-loss-functions/cross-entropy': {
        'name': '实现稳定交叉熵', 'signature': 'cross_entropy(logits, targets, ignore_index=-100) -> Tensor',
        'input': 'logits: [B,C] 未归一化分数；targets: [B] long；可含 ignore_index。',
        'output': '返回非忽略样本的标量平均损失；全部忽略时返回可反传的零标量。',
        'tasks': ['使用 log-sum-exp 保证数值稳定', '正确忽略指定标签', '梯度只流向有效样本'],
    },
    '06-optimizers/adamw': {
        'name': '实现单步 AdamW', 'signature': 'adamw_step(param, grad, m, v, step, lr=..., beta1=..., beta2=..., eps=..., weight_decay=...)',
        'input': '参数、梯度、一阶/二阶动量形状相同；step 从 1 开始。',
        'output': '返回新的 (param, m, v)，不能原地修改输入。',
        'tasks': ['更新动量并做 bias correction', '使用解耦 weight decay', '处理任意浮点 dtype 且不修改原张量'],
    },
    '07-reinforcement-learning/gae': {
        'name': '实现 GAE 优势估计', 'signature': 'compute_gae(rewards, values, dones, gamma=0.99, lam=0.95) -> Tensor',
        'input': 'rewards/dones: [T]；values: [T+1]；done=1 表示该步结束 episode。',
        'output': '返回 [T] advantages，末端 bootstrap 使用 values[T]。',
        'tasks': ['反向递推 TD residual', 'done 后不能跨 episode bootstrap', '支持 float 输入并保留 dtype'],
    },
    '08-efficient-training/lora': {
        'name': '实现 LoRA Linear', 'signature': 'class LoRALinear(nn.Module): LoRALinear(base, rank, alpha)',
        'input': 'base 为 nn.Linear；rank 为低秩维度；alpha 为缩放系数。',
        'output': '返回 base(x) + (x @ A.T @ B.T) * alpha/rank。',
        'tasks': ['冻结 base 参数', '暴露 A[rank,in] 与 B[out,rank] 可训练参数', 'B 初始化为零且梯度正常流动'],
    },
    '09-inference-optimization/kv-cache': {
        'name': '实现 KV Cache', 'signature': 'class KVCache: append(k, v) -> (k_all, v_all); reset() -> None',
        'input': 'k/v: [B,H,T,D]，每次 append 可追加一个或多个 token。',
        'output': '沿序列维拼接并返回累计 K/V；reset 后下一次追加从空缓存开始。',
        'tasks': ['沿 dim=2 追加', 'K/V 始终配对且不互相污染', '支持 reset 和首次 append'],
    },
    '10-transformer-architecture/ffn': {
        'name': '实现 Transformer FFN', 'signature': 'class FeedForward(nn.Module): FeedForward(dim, hidden_dim)',
        'input': 'x 的最后一维为 dim，支持 [B,T,dim] 或任意前置维度。',
        'output': '执行 Linear(dim,hidden_dim) -> GELU -> Linear(hidden_dim,dim)。',
        'tasks': ['暴露 fc1 和 fc2', '逐 token 处理最后一维', '输入和参数均可反向传播'],
    },
}


GENERIC = {
    'causal-mask': ('build_causal_mask', 'build_causal_mask(seq_len, device=None) -> Tensor', '返回 [seq_len,seq_len] bool 上三角因果 mask；True 表示允许访问当前位置及过去位置。', ['支持 seq_len=0 和 seq_len=1', '返回 bool 且在指定 device 上', '未来位置必须全部屏蔽']),
    'flash-attention': ('flash_attention', 'flash_attention(q, k, v, block_size=128) -> Tensor', '输入 [B,H,T,D]，返回与标准 attention 相同的 [B,H,T,Dv] 结果。', ['分块计算避免显式保存完整 attention 矩阵', '数值结果与参考实现接近', '支持非整除 block_size']),
    'gqa': ('repeat_kv', 'repeat_kv(kv, num_query_heads) -> Tensor', '将 [B,Hkv,T,D] 的 KV 头复制到 [B,Hq,T,D]，要求 Hq 可被 Hkv 整除。', ['按 head 维重复而非序列维', '保持 dtype/device', '非法头数要抛出 ValueError']),
    'mqa': ('share_kv_heads', 'share_kv_heads(k, v) -> tuple[Tensor, Tensor]', '将多 KV 头压缩成 MQA 所需的共享单头表示，输入 [B,H,T,D]。', ['沿 head 维聚合', 'K/V 独立返回', '保持 batch、序列和 hidden 维']),
    'multi-head-attention': ('multi_head_attention', 'multi_head_attention(x, num_heads, q_proj, k_proj, v_proj, out_proj) -> Tensor', '输入 x:[B,T,D]，使用给定投影完成 MHA，返回 [B,T,D]。', ['拆分和合并 head', '缩放点积并应用 causal mask 参数', '输出投影后恢复原维度']),
    'layer-norm': ('layer_norm', 'layer_norm(x, weight, bias, eps=1e-5) -> Tensor', '沿最后一维标准化 x，并应用 weight/bias。', ['均值方差沿最后一维计算', '支持任意前置维度', 'eps 防止零方差']),
    'batch-norm': ('batch_norm_2d', 'batch_norm_2d(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5) -> Tensor', '处理 [N,C,H,W] 特征，训练时按 batch 更新统计量，推理时使用 running 统计量。', ['训练/推理分支正确', '统计量按 channel 计算', '保持输出形状']),
    'pre-vs-post-norm': ('apply_transformer_block', 'apply_transformer_block(x, sublayer, norm, prenorm=True) -> Tensor', '实现 Pre-Norm 与 Post-Norm 两种残差连接。', ['prenorm=True 时先 norm 再 sublayer', 'prenorm=False 时 add 后 norm', '保留残差分支梯度']),
    'sinusoidal': ('sinusoidal_position_encoding', 'sinusoidal_position_encoding(length, dim, device=None) -> Tensor', '生成 [length,dim] 正弦/余弦位置编码。', ['偶数维使用 sin、奇数维使用 cos', '支持奇数 dim', '位置 0 的结果正确']),
    'learnable': ('LearnablePositionEmbedding', 'class LearnablePositionEmbedding(nn.Module): LearnablePositionEmbedding(max_length, dim)', '实现可学习的位置 embedding，输入位置 id 返回 [*,dim]。', ['参数形状 [max_length,dim]', '越界 id 明确报错', '支持 batch 输入']),
    'alibi': ('build_alibi_bias', 'build_alibi_bias(num_heads, seq_len) -> Tensor', '生成 ALiBi 的 [num_heads,seq_len,seq_len] 线性距离 bias。', ['距离越远惩罚越大', '每个 head 使用不同 slope', '返回浮点张量']),
    'temperature': ('temperature_scale', 'temperature_scale(logits, temperature) -> Tensor', '对 logits 做 temperature 缩放，temperature 必须为正。', ['除以 temperature 而非乘法', 'temperature<=0 抛出 ValueError', '不改变形状']),
    'top-k': ('top_k_filter', 'top_k_filter(logits, k) -> Tensor', '保留每行最大的 k 个 logits，其余位置设为 -inf。', ['k=1 和 k=词表大小正确', '保留原始顺序', '非法 k 抛出 ValueError']),
    'beam-search': ('beam_search_step', 'beam_search_step(log_probs, beam_scores, beam_size) -> tuple[Tensor,Tensor]', '从候选 log probability 和 beam 分数中选择下一步 top beams。', ['累加 beam 分数', '返回排序后的 beam id 和分数', '结果数量不超过 beam_size']),
    'greedy': ('greedy_decode_step', 'greedy_decode_step(logits) -> Tensor', '沿最后一维返回每个位置最大 logit 的 token id。', ['返回 long dtype', '支持任意 batch/序列前置维', '处理并列最大值稳定']),
    'combined-sampling': ('sample_next_token', 'sample_next_token(logits, temperature=1.0, top_k=None, top_p=None) -> Tensor', '按 temperature、top-k、top-p 顺序过滤并采样下一个 token。', ['过滤顺序明确', '至少保留一个 token', '支持 batch 输入']),
    'stable-softmax': ('stable_softmax', 'stable_softmax(x, dim=-1) -> Tensor', '实现数值稳定 softmax，输入可含很大的正负值。', ['减去最大值', '输出沿 dim 和为 1', '避免 inf/nan']),
    'focal': ('focal_loss', 'focal_loss(logits, targets, gamma=2.0, reduction="mean") -> Tensor', '实现多分类 focal loss，降低容易样本的权重。', ['使用 log_softmax', '支持 mean/sum/none', 'targets 为 long']),
    'kl-divergence': ('kl_divergence', 'kl_divergence(log_p, log_q, reduction="batchmean") -> Tensor', '计算两个 log-probability 分布的 KL(P||Q)。', ['使用 exp(log_p) 加权', '支持 batch 维', 'reduction 语义正确']),
    'mse': ('mse_loss', 'mse_loss(prediction, target, reduction="mean") -> Tensor', '计算预测与目标的均方误差。', ['支持 mean/sum/none', '广播规则与 PyTorch 一致', '保留梯度']),
    'sft': ('sft_loss', 'sft_loss(logits, labels, ignore_index=-100) -> Tensor', '计算自回归 SFT 的 token-level cross entropy，忽略 prompt 标签。', ['沿序列右移 logits/labels', '忽略 -100', '返回标量平均损失']),
    'reward-model': ('pairwise_reward_loss', 'pairwise_reward_loss(chosen_reward, rejected_reward) -> Tensor', '用 logistic pairwise loss 让 chosen reward 高于 rejected reward。', ['计算 -log sigmoid(chosen-rejected)', '支持 batch', '保持数值稳定']),
    'label-smoothing': ('label_smoothed_cross_entropy', 'label_smoothed_cross_entropy(logits, targets, smoothing=0.1) -> Tensor', '实现带 label smoothing 的交叉熵。', ['正确分配平滑概率', 'targets 为 long', '支持 ignore_index']),
    'sgd': ('sgd_step', 'sgd_step(param, grad, lr) -> Tensor', '返回不原地修改参数的 SGD 一步结果。', ['param-lr*grad', '保持 dtype/device', '输入不能被修改']),
    'momentum': ('momentum_step', 'momentum_step(param, grad, velocity, lr, momentum) -> tuple[Tensor,Tensor]', '实现带 momentum buffer 的 SGD 更新。', ['先更新 velocity', '使用更新后的 velocity 更新参数', '不原地修改']),
    'adam': ('adam_step', 'adam_step(param, grad, m, v, step, lr=1e-3) -> tuple[Tensor,Tensor,Tensor]', '实现 Adam 一步，不包含 AdamW 的解耦衰减。', ['一阶/二阶动量', 'bias correction', '不原地修改']),
    'lr-scheduler': ('cosine_lr', 'cosine_lr(step, total_steps, max_lr, min_lr=0.0) -> float', '实现 cosine learning-rate schedule。', ['step=0 返回 max_lr', '末步接近 min_lr', '边界 step 要裁剪']),
    'reinforce': ('reinforce_loss', 'reinforce_loss(log_probs, returns) -> Tensor', '计算 REINFORCE 的负 log-prob 加权回报损失。', ['逐样本求和 log_probs', 'returns 不应反传', '返回 batch 平均']),
    'ppo': ('ppo_clipped_loss', 'ppo_clipped_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2) -> Tensor', '实现 PPO clipped surrogate objective 的负值。', ['计算 ratio', '按 advantage 正负裁剪', '返回标量']),
    'dpo': ('dpo_loss', 'dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1) -> Tensor', '实现 DPO preference loss。', ['构造 policy/ref margin', '使用 log sigmoid', '返回 batch 平均']),
    'grpo': ('group_relative_advantage', 'group_relative_advantage(rewards, group_ids, eps=1e-8) -> Tensor', '按 group 对 reward 做均值方差标准化，返回相对优势。', ['同组独立归一化', '处理组内方差为零', '输出与 rewards 同形状']),
    'rlhf-pipeline': ('compute_rlhf_reward', 'compute_rlhf_reward(reward, kl, kl_coef) -> Tensor', '组合 reward model 分数和 KL 惩罚。', ['reward - kl_coef*kl', '保持输入形状', '支持 batch/sequence']),
    'gradient-checkpointing': ('checkpointed_forward', 'checkpointed_forward(function, x) -> Tensor', '使用 PyTorch checkpoint 调用 function，减少前向激活保存。', ['输出与直接调用一致', '梯度可以回传', '不改变输入']),
    'mixed-precision': ('mixed_precision_step', 'mixed_precision_step(model, optimizer, x, target, scaler) -> Tensor', '实现 autocast + GradScaler 的一次训练步。', ['前向在 autocast 中', 'scaler scale/backward/step/update 顺序正确', '返回 detached loss']),
    'gradient-accumulation': ('accumulate_gradients', 'accumulate_gradients(model, batches, optimizer, accumulation_steps) -> float', '按 accumulation_steps 累积梯度后更新模型。', ['loss 除以累积步数', '只在边界 step optimizer.step', '清理梯度']),
    'kv-cache': ('KVCache', 'class KVCache: append(k, v) -> tuple[Tensor,Tensor]; reset() -> None', '沿序列维追加 K/V 并支持 reset。', ['首次追加、连续追加和 reset', 'K/V 形状匹配', '不能混淆 batch/head 维']),
    'paged-attention': ('paged_attention', 'paged_attention(query, key_pages, value_pages, page_table) -> Tensor', '根据 page table 从分页 KV 中读取并计算 attention。', ['正确映射逻辑序列到物理页', '只读取有效 token', '输出形状正确']),
    'speculative-decoding': ('accept_speculative_tokens', 'accept_speculative_tokens(draft_probs, target_probs, draft_tokens, rng=None) -> tuple[Tensor,int]', '按 speculative decoding 接受/拒绝 draft token，并返回新序列和接受数。', ['按 target/draft 概率比接受', '首次拒绝后停止接受', '边界长度处理正确']),
    'continuous-batching': ('merge_sequences', 'merge_sequences(sequences, max_batch) -> Tensor', '将不同长度 token 序列 padding 成 batch tensor，并返回有效 mask。', ['padding 位置使用 pad_id', '保留原始 token 顺序', '超过 max_batch 明确报错']),
    'overview': ('build_causal_attention', 'build_causal_attention(x, attention, causal=True) -> Tensor', '封装 decoder-only 的 causal self-attention 数据流。', ['causal=True 屏蔽未来 token', '输出形状与 x 相同', '支持关闭 causal']),
    'encoder': ('TransformerEncoderLayer', 'class TransformerEncoderLayer(nn.Module): TransformerEncoderLayer(dim, heads, hidden_dim)', '实现可双向 self-attention 的 encoder layer。', ['残差与 norm 顺序明确', '支持 padding mask', '输出形状不变']),
    'decoder': ('TransformerDecoderLayer', 'class TransformerDecoderLayer(nn.Module): TransformerDecoderLayer(dim, heads, hidden_dim)', '实现带 causal self-attention 和 cross-attention 的 decoder layer。', ['self/cross attention 顺序正确', 'causal mask 生效', '输出形状不变']),
    'gpt-decoder': ('GPTDecoderBlock', 'class GPTDecoderBlock(nn.Module): GPTDecoderBlock(dim, heads, hidden_dim)', '实现 pre-norm GPT decoder block。', ['attention 与 FFN 残差', '使用 causal mask', '支持 batch-first 输入']),
    'residual-norm': ('pre_norm_block', 'pre_norm_block(x, sublayer, norm) -> Tensor', '实现 x + sublayer(norm(x)) 的 pre-norm 残差块。', ['先 norm', '保留恒等残差路径', '不修改 x']),
    'full-transformer': ('TransformerLM', 'class TransformerLM(nn.Module): TransformerLM(vocab_size, dim, layers, heads)', '实现最小 decoder-only language model，输入 token id 返回 logits。', ['embedding/blocks/lm_head 串联', 'causal attention', '输出 [B,T,vocab_size]']),
}


def get_spec(module, page, label=None):
    key = f'{module}/{page}'
    if key in CORE:
        return CORE[key]
    slug = page
    if slug not in GENERIC:
        return {'name': f'{label or page} 实现题', 'signature': f'implement_{slug}(...) -> Tensor', 'input': '按照本模块理论定义输入张量和参数。', 'output': '返回题目要求的张量或对象。', 'tasks': ['保持输入输出形状契约', '处理边界条件', '保留梯度和 dtype/device']}
    function, signature, output, tasks = GENERIC[slug]
    return {'name': f'实现 {function}', 'signature': signature, 'input': '具体输入约定见函数签名和本模块理论。', 'output': output, 'tasks': tasks}
