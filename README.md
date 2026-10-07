# PyTorch 与大模型手撕复习

离线 Markdown 教材 + 本地 PyTorch 练习 + 浏览器学习工作台。环境和前端依赖全部安装在当前目录下的 `.venv/`、`node_modules/`，启动时不依赖全局项目环境。

## 一键启动

```bash
./setup_env.sh   # 首次执行：在当前目录创建 .venv，并安装 PyTorch、网页服务和离线前端资源
./start.sh       # 启动本地网页并自动打开浏览器
```

启动后访问终端显示的 `http://127.0.0.1:8765`。网页以“一个 Markdown 模块一页”为最小学习单位：左侧目录可展开，讲解区按标题左右翻页，同一章节的模块也用左右箭头切换，右侧编辑器始终绑定当前模块的独立代码和测评。需要指定 Python 时使用 `PYTHON=/path/to/python ./setup_env.sh`；后续启动永远使用当前目录下的 `.venv/bin/python`。

## 从这里开始

1. 执行 `./setup_env.sh`，它会把 Python 依赖装入当前目录 `.venv/`，把 KaTeX 和 Lucide 装入当前目录 `node_modules/`。
2. 从 [torch 常用语法速查](docs/00-torch-basics/torch-basics.md) 开始。每个章节的 `README.md` 有总述、目录和高频八股；每个 Markdown 模块都对应一份独立练习文件。
3. 网页进入任意模块后，直接编辑右侧显示的 `solutions/...` 文件，点击保存和运行测评。核心 11 个模块有数值、梯度和边界测试；其余模块先执行独立文件导入与接口检查，后续可按相同契约补充专门断言。
4. 命令行测评仍可运行 `./.venv/bin/python -m grader 01-attention`；网页模块测评会调用 `./.venv/bin/python -m grader 01-attention multi-head-attention`。`[通过]`、`[未实现]`、`[失败]` 会分别显示在网页和终端。

## 章节路线

| 顺序 | 章节（总述及高频八股） | 模块数 | 精确测评模块 |
|---|---|---:|---|
| 00 | [PyTorch 基础](docs/00-torch-basics/README.md) | 1 | [masked_mean](docs/00-torch-basics/torch-basics.md) |
| 01 | [Attention](docs/01-attention/README.md) | 6 | [Scaled Dot-Product Attention](docs/01-attention/scaled-dot-product-attention.md) |
| 02 | [归一化](docs/02-normalization/README.md) | 4 | [RMSNorm](docs/02-normalization/rms-norm.md) |
| 03 | [位置编码](docs/03-position-encoding/README.md) | 4 | [RoPE](docs/03-position-encoding/rope.md) |
| 04 | [采样策略](docs/04-sampling/README.md) | 6 | [Top-p](docs/04-sampling/top-p.md) |
| 05 | [损失函数](docs/05-loss-functions/README.md) | 8 | [Cross Entropy](docs/05-loss-functions/cross-entropy.md) |
| 06 | [优化器](docs/06-optimizers/README.md) | 5 | [AdamW](docs/06-optimizers/adamw.md) |
| 07 | [强化学习](docs/07-reinforcement-learning/README.md) | 6 | [GAE](docs/07-reinforcement-learning/gae.md) |
| 08 | [高效训练](docs/08-efficient-training/README.md) | 4 | [LoRA](docs/08-efficient-training/lora.md) |
| 09 | [推理优化](docs/09-inference-optimization/README.md) | 4 | [KV Cache](docs/09-inference-optimization/kv-cache.md) |
| 10 | [Transformer 架构](docs/10-transformer-architecture/README.md) | 7 | [FFN](docs/10-transformer-architecture/ffn.md) |

网页服务只绑定 `127.0.0.1`，测评接口会执行本地 Python 代码；不要把这个开发服务直接暴露到公网主页。若要挂到公开主页，应另做认证、沙箱和资源隔离。所有新功能均不依赖远端网页、投票站点或模型权重。
