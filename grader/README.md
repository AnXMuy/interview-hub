# 本地测评说明

在仓库根目录执行。仅依赖 Python 3.10+ 与本地 PyTorch，无需服务器、GPU、联网、模型权重或 pytest。

```bash
./.venv/bin/python -m grader                              # 全部 11 个精确测评
./.venv/bin/python -m grader 01-attention                 # 测一个章节的精确题
./.venv/bin/python -m grader 01-attention multi-head-attention  # 测一个普通模块
./.venv/bin/python -m grader --help                       # 查看帮助
```

核心题模板在 `solutions/`，普通模块模板在 `solutions/pages/{章节}__{模块}.py`。每个模块都使用明确的函数或类接口，不再生成模糊的 `solve(*args, **kwargs)`；网页第一次打开普通模块时会按题目契约初始化具体模板。没有写完会显示 `[未实现]`，断言失败显示 `[失败]` 及具体检查项；全部通过时退出码为 0，否则为 1。文档中的完整代码属于学习示例，并不会被测评程序导入。

| 主题 | 编辑文件 | 唯一接口 | 重点 |
|---|---|---|---|
| torch 基础 | `solutions/torch_basics.py` | `masked_mean` | 广播、空行、梯度 |
| Attention | `solutions/attention.py` | `scaled_dot_product_attention` | 缩放、mask、交叉长度 |
| 归一化 | `solutions/normalization.py` | `RMSNorm` | 参数、最后一维 |
| 位置编码 | `solutions/position_encoding.py` | `apply_rope` | 相邻配对、位置偏移 |
| 采样 | `solutions/sampling.py` | `top_p_filter` | 累积概率边界 |
| 损失 | `solutions/loss_functions.py` | `cross_entropy` | 稳定性、忽略项、反传 |
| 优化器 | `solutions/optimizers.py` | `adamw_step` | 解耦衰减、偏差修正 |
| 强化学习 | `solutions/reinforcement_learning.py` | `compute_gae` | done 与 bootstrap |
| 高效训练 | `solutions/efficient_training.py` | `LoRALinear` | 冻结底座、A/B |
| 推理 | `solutions/inference_optimization.py` | `KVCache` | seq 追加、reset |
| Transformer | `solutions/transformer_architecture.py` | `FeedForward` | fc1/fc2、GELU |

其余 Markdown 模块均有独立练习文件和独立测评命令，例如：

```bash
./.venv/bin/python -m grader 01-attention multi-head-attention
# solutions/pages/01-attention__multi-head-attention.py
```

普通模块的默认测评会检查文件可导入、至少存在一个公开函数或类、并明确拒绝 `NotImplementedError`。每个模块的题目契约由 `app/exercise_specs.py` 维护，并同时显示在网页题面和对应文档页面；需要数值级反馈时，在 `grader/__main__.py` 为该模块加入专门测试即可。

输入输出约定以模板 docstring 为准；测试数据和参考计算在 [__main__.py](__main__.py)，可以离线逐项阅读。测评不修改实现，不下载模型；出错时保留检查名及最大误差便于定位。
