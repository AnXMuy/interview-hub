# torch-basics：常用 PyTorch 语法速查

本文示例默认 `import torch`、`from torch import nn`、`import torch.nn.functional as F`。先看形状，再看 dtype/device，最后看梯度。示例可直接粘贴到本地 Python；不需要在线 Notebook。

## 1. 创建张量、类型与设备

```python
import torch
from torch import nn
import torch.nn.functional as F

x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
z = torch.zeros(2, 3)             # 全零
o = torch.ones_like(z)            # 同形状/dtype/device
r = torch.randn(2, 3)            # 标准正态
i = torch.arange(0, 6).reshape(2, 3)
steps = torch.linspace(0, 1, 5)
ids = torch.tensor([0, 2], dtype=torch.long)  # Embedding/CE 标签为 long
mask = torch.tensor([True, False])            # mask 为 bool
print(x.shape, x.ndim, x.numel(), x.dtype, x.device)

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
x = x.to(device=device, dtype=torch.float32)
scalar = x[0, 0].item()          # 仅标量转 Python 值（会同步设备）
array = x.detach().cpu().numpy() # GPU 张量先搬到 CPU；有梯度先 detach
```

不要在训练中反复 `.item()` 或把有梯度的张量转 NumPy 后继续求导。

## 2. 形状与维度

```python
x = torch.arange(24).reshape(2, 3, 4) # [B,T,D]
a = x.reshape(6, 4)            # 必要时拷贝
b = x.flatten(0, 1)            # [B*T,D]
c = x.unsqueeze(1)             # [B,1,T,D]
d = c.squeeze(1)               # 仅压指定大小为 1 的轴
e = x.transpose(1, 2)          # [B,D,T]，通常非连续
f = x.permute(2, 0, 1)          # [D,B,T]
g = e.contiguous().view(2, 4, 3)  # view 需要兼容步长
left, right = x.chunk(2, dim=0)
stacked = torch.stack([x, x], dim=0) # 新增轴 [2,B,T,D]
joined = torch.cat([x, x], dim=1)    # 连接旧轴 [B,2*T,D]
```

`reshape` 不保证共享存储；`transpose`/`permute` 可能得到非连续视图。`squeeze()` 不指定轴时会连 batch=1 一起删掉。

## 3. 索引、掩码与广播

```python
x = torch.arange(12).reshape(2, 3, 2).float()
token0 = x[:, 0, :]                       # [2,2]
selected = x[:, [0, 2], :]               # 高级索引 [2,2,2]
valid = torch.tensor([[True, False, True], [False, True, True]])
masked = x.masked_fill(~valid[..., None], 0) # [2,3,1] 广播到 [2,3,2]
bias = torch.randn(2, 1, 2)
y = x + bias                             # 广播到 [2,3,2]
row = torch.arange(2)
picked = x[row, torch.tensor([0, 2])]   # [2,2]，逐行选 token
```

广播从最后一维向左对齐；每个轴必须相等或有一个是 1。布尔 mask 要明确 `True` 是有效还是屏蔽，尤其 PyTorch 不同注意力 API 的约定可能不同。

## 4. 归约、矩阵运算与概率

```python
x = torch.randn(2, 3, 4)
mean = x.mean(dim=-1, keepdim=True)     # [2,3,1]
total = x.sum(dim=(0, 1))               # [4]
max_values, indices = x.max(dim=-1)     # [2,3] 各自返回值和下标
sorted_values, sorted_ids = x.sort(dim=-1, descending=True)
top_values, top_ids = x.topk(2, dim=-1)
q = torch.randn(2, 3, 4)
k = torch.randn(2, 5, 4)
scores = q @ k.transpose(-2, -1)        # [2,3,5]，批次维广播
prob = scores.softmax(dim=-1)
log_prob = F.log_softmax(scores, dim=-1) # 优于先 softmax 再 log
log_z = torch.logsumexp(scores, dim=-1)  # 稳定 log(sum(exp))
clip = scores.clamp(min=-5, max=5)
```

注意 `sum/mean` 的维度和 `keepdim`。Cross Entropy 输入的是 logits，通常无需先做 softmax。

## 5. Autograd 与原地操作

```python
x = torch.tensor([1., 2.], requires_grad=True)
w = torch.tensor([3., 4.], requires_grad=True)
loss = (x * w).sum()
loss.backward()                  # x.grad = w；标量可直接 backward
print(x.grad, w.grad)
x.grad = None                    # 下次 backward 前清掉累积梯度
grad_w = torch.autograd.grad((x * w).sum(), w)[0]  # 返回梯度而不累积在 w.grad
with torch.no_grad():
    w -= 0.01 * grad_w           # 参数更新不进入计算图
frozen = x.detach()             # 与当前图断开，但可能共享存储
```

`.backward()` 默认累积梯度；不要对仍用于反向传播的叶子张量做危险的原地修改。`detach()` 断梯度，不等于深拷贝。

## 6. nn.Module、参数与常见层

```python
class TinyLM(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, vocab)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.embed(ids))       # [B,T,D]
        return self.fc(F.gelu(h))            # [B,T,V] logits

model = TinyLM(100, 32)
print(sum(p.numel() for p in model.parameters()))
model.register_buffer('counter', torch.zeros(1)) # 随设备移动，不由优化器更新
for name, parameter in model.named_parameters():
    print(name, parameter.shape, parameter.requires_grad)
```

参数要注册成 `nn.Parameter` 或放进子 `nn.Module`；普通张量属性不会被优化器自动更新。`nn.Dropout` 在 train/eval 模式行为不同。

## 7. 损失、反向传播与训练循环

```python
model = TinyLM(100, 32).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
ids = torch.randint(0, 100, (2, 6), device=device)
targets = ids.roll(-1, dims=1)
targets[:, -1] = -100             # 忽略无下一词的末位

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(ids)               # [B,T,V]
loss = F.cross_entropy(logits.reshape(-1, 100), targets.reshape(-1), ignore_index=-100)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

model.eval()
with torch.inference_mode():
    next_id = model(ids)[:, -1].argmax(dim=-1)
```

`.eval()` 只改变模块行为，不关闭梯度；推理用 `inference_mode()`。梯度累积时每个 microbatch 的 loss 要按步数缩放，并且只在更新前清零一次。

## 8. Dataset、随机性与保存

```python
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
dataset = TensorDataset(torch.randint(0, 100, (16, 6)))
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for (batch_ids,) in loader:
    print(batch_ids.shape)         # [4,6]
    break

torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'checkpoint.pt')
# 只加载自己信任的本地文件；新版本可对仅含权重的文件使用 weights_only=True
state = torch.load('checkpoint.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state['model'])
```

`state_dict()` 存参数及 buffer，不存 Python 类定义；加载时先创建同结构模型。运行最后一节会在当前目录产生 `checkpoint.pt`。

## 9. 典型错误清单

| 现象 | 定位 |
|---|---|
| matmul 维度不匹配 | 在 `@` 前打印两侧 shape，核对倒数第二、第一维 |
| `view` 报 stride 错误 | `transpose/permute` 后用 `reshape` 或 `contiguous().view` |
| 没有梯度 | 检查 `detach`、`no_grad`、`.item()`、参数是否注册 |
| CUDA/CPU 类型不一致 | 模型、输入和 mask 移到同一 device |
| 交叉熵异常 | 传 logits、整型标签；类别维放最后再展平 |
| 全 mask softmax 得 NaN | 为每行保留至少一个可见位置，或单独处理空行 |

## 动手练习

只实现 `solutions/torch_basics.py` 的 `masked_mean(x, mask)`：输入 `x:[B,T,D]`、布尔 `mask:[B,T]`；对每个样本有效 token 求均值，全部无效则输出零，保持梯度。运行 `python -m grader 00-torch-basics`。

## 常见面试问答

**问：`detach`、`no_grad`、`inference_mode` 各是什么？**

答：`detach` 对指定张量断图；`no_grad` 在上下文中不记录计算图；`inference_mode` 专用于推理，可进一步减少 autograd 开销，产生的 inference tensor 不应直接回到需记录梯度的训练图。

**问：为什么 `zero_grad` 必须明确调用？**

答：PyTorch 默认累积梯度；除非要累积多个 microbatch，否则每次优化前需清零。

**问：什么时候用 `stack`，什么时候用 `cat`？**

答：`stack` 新增维度；`cat` 在已有维度拼接。

[返回本章目录](README.md)
