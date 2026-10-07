"""Run deterministic CPU checks without pytest, network or model downloads."""
import argparse
import importlib
import math
import importlib.util
import traceback
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit('缺少 torch：在本地 Python 环境安装 PyTorch 后再运行。') from exc


def check(got, expected, label, *, atol=1e-5):
    if not isinstance(got, torch.Tensor):
        raise AssertionError(f'{label}: 应返回 torch.Tensor，实际 {type(got).__name__}')
    if got.shape != expected.shape:
        raise AssertionError(f'{label}: 形状应为 {tuple(expected.shape)}，实际 {tuple(got.shape)}')
    if got.dtype != expected.dtype:
        raise AssertionError(f'{label}: dtype 应为 {expected.dtype}，实际 {got.dtype}')
    if not torch.allclose(got, expected, atol=atol, rtol=atol, equal_nan=False):
        difference = (got - expected).abs()
        raise AssertionError(f'{label}: 数值不一致，最大绝对误差 {difference.max().item():.3g}；检查缩放、归一化轴、mask 或边界条件')


def test_attention(s):
    torch.manual_seed(4)
    q, k, v = torch.randn(2, 3, 2, 4), torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 6)
    mask = torch.tensor([[[[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]]], dtype=torch.bool)
    for label, m in [('无 mask/交叉长度', None), ('广播 mask，True=可见', mask)]:
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if m is not None:
            scores = scores.masked_fill(~m, -torch.inf)
        expected = scores.softmax(-1) @ v
        check(s.scaled_dot_product_attention(q, k, v, m), expected, label)


def test_torch_basics(s):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    mask = torch.tensor([[True, False, True], [False, False, False]])
    got = s.masked_mean(x, mask)
    expected = torch.stack(((x[0, 0] + x[0, 2]) / 2, torch.zeros_like(x[1, 0])))
    check(got, expected, 'mask 广播、按有效 token 计数、空样本返回零')
    grad = torch.autograd.grad(got.sum(), x)[0]
    ref_grad = torch.autograd.grad(expected.sum(), x)[0]
    check(grad, ref_grad, '不能 detach，梯度只流经有效 token')


def test_normalization(s):
    torch.manual_seed(5)
    layer = s.RMSNorm(4, eps=1e-6)
    assert isinstance(layer, nn.Module), 'RMSNorm 应继承 nn.Module'
    assert isinstance(layer.weight, nn.Parameter), '应有可训练的 weight 参数'
    check(layer.weight, torch.ones(4), '初始 weight')
    x = torch.randn(2, 3, 4, requires_grad=True)
    check(layer(x), x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6), '按最后一维 RMS 归一化')
    layer(x).sum().backward()
    assert x.grad is not None and layer.weight.grad is not None, '输入及 weight 都应有梯度'
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1., 2., 3., 4.]))
    check(layer(x), x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * layer.weight, 'weight 缩放')


def test_position_encoding(s):
    x = torch.randn(2, 3, 4, 6)
    for offset in (0, 9):
        pos = torch.arange(offset, offset + 4, dtype=x.dtype)
        inv = 10000.0 ** (-torch.arange(0, 6, 2, dtype=x.dtype) / 6)
        theta = pos[:, None] * inv[None, :]
        pairs = x.reshape(2, 3, 4, 3, 2)
        expected = torch.stack((pairs[..., 0] * theta.cos() - pairs[..., 1] * theta.sin(),
                                pairs[..., 0] * theta.sin() + pairs[..., 1] * theta.cos()), dim=-1).reshape_as(x)
        check(s.apply_rope(x, offset), expected, f'相邻配对旋转 offset={offset}')


def test_sampling(s):
    logits = torch.tensor([[4., 3., 2., 1.], [0., 3., -1., 2.]])
    for p in (0.3, 0.75, 1.0):
        sorted_logits, indices = logits.sort(-1, descending=True)
        cumulative = sorted_logits.softmax(-1).cumsum(-1)
        remove = cumulative - sorted_logits.softmax(-1) >= p
        remove[:, 0] = False
        expected = logits.masked_fill(torch.zeros_like(remove).scatter(-1, indices, remove), -torch.inf)
        got = s.top_p_filter(logits.clone(), p)
        check(got, expected, f'top-p={p}（保留越过阈值的词）')
        check(logits, torch.tensor([[4., 3., 2., 1.], [0., 3., -1., 2.]]), '不得原地修改 logits')


def test_loss_functions(s):
    logits = torch.tensor([[1000., 999., -1000.], [2., -2., 0.], [-10., 4., 0.]], requires_grad=True)
    for targets in (torch.tensor([0, 1, 2]), torch.tensor([0, -100, 2])):
        got = s.cross_entropy(logits, targets)
        expected = F.cross_entropy(logits, targets, ignore_index=-100)
        check(got, expected, '数值稳定与 ignore_index 的有效项平均')
        grad = torch.autograd.grad(got, logits, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        check(grad, ref_grad, '交叉熵反向梯度')


def test_optimizers(s):
    param = torch.tensor([1., -2.])
    m, v = torch.zeros_like(param), torch.zeros_like(param)
    grad = torch.tensor([0.2, -0.3])
    for t in (1, 2, 3):
        got = s.adamw_step(param.clone(), grad, m.clone(), v.clone(), t, lr=0.1, weight_decay=0.2)
        assert isinstance(got, tuple) and len(got) == 3, '应返回 (param, m, v) 三元组'
        m_new = 0.9 * m + 0.1 * grad
        v_new = 0.999 * v + 0.001 * grad.square()
        p_new = param * (1 - 0.1 * 0.2) - 0.1 * (m_new / (1 - 0.9 ** t)) / ((v_new / (1 - 0.999 ** t)).sqrt() + 1e-8)
        for actual, reference, name in zip(got, (p_new, m_new, v_new), ('参数（含解耦衰减和偏差修正）', '一阶矩', '二阶矩')):
            check(actual, reference, f'第 {t} 步 {name}')
        param, m, v = p_new, m_new, v_new


def test_reinforcement_learning(s):
    rewards = torch.tensor([1., 2., 3., 4.])
    values = torch.tensor([0.2, 0.5, 0.3, 0.8, 0.9])
    dones = torch.tensor([0., 1., 0., 0.])
    delta = rewards + 0.9 * values[1:] * (1 - dones) - values[:-1]
    expected = torch.empty_like(rewards)
    running = 0.
    for t in reversed(range(len(rewards))):
        running = delta[t] + 0.9 * 0.8 * (1 - dones[t]) * running
        expected[t] = running
    check(s.compute_gae(rewards, values, dones, gamma=0.9, lam=0.8), expected, '终止状态和 bootstrap')


def test_efficient_training(s):
    torch.manual_seed(6)
    base = nn.Linear(4, 3)
    original_weight = base.weight.detach().clone()
    layer = s.LoRALinear(base, 2, 4.)
    assert not any(p.requires_grad for p in base.parameters()), 'base 参数必须冻结'
    assert isinstance(layer.A, nn.Parameter) and layer.A.shape == (2, 4), 'A 应为 [rank,in_features] 可训练参数'
    assert isinstance(layer.B, nn.Parameter) and layer.B.shape == (3, 2), 'B 应为 [out_features,rank] 可训练参数'
    x = torch.randn(2, 5, 4)
    check(layer(x), base(x), 'B 初始为零，输出应与 base 相同')
    with torch.no_grad():
        layer.A.fill_(0.25)
        layer.B.fill_(0.5)
    check(layer(x), base(x) + (x @ layer.A.T @ layer.B.T) * 2, '低秩增量和 alpha/rank')
    layer(x).sum().backward()
    assert layer.A.grad is not None and layer.B.grad is not None, 'A/B 必须有梯度'
    check(base.weight.detach(), original_weight, '不得修改 base 权重')


def test_inference_optimization(s):
    cache = s.KVCache()
    a = torch.randn(2, 3, 2, 4)
    b = torch.randn(2, 3, 1, 4)
    for i, (k, v) in enumerate(((a, a + 1), (b, b + 1)), 1):
        result = cache.append(k, v)
        assert isinstance(result, tuple) and len(result) == 2, 'append 应返回 (k,v)'
        expected_k = a if i == 1 else torch.cat((a, b), dim=2)
        check(result[0], expected_k, 'K 按 seq 维追加')
        check(result[1], expected_k + 1, 'V 按 seq 维追加且保持独立')
    cache.reset()
    check(cache.append(b, b + 1)[0], b, 'reset 后缓存应清空')


def test_transformer_architecture(s):
    torch.manual_seed(7)
    layer = s.FeedForward(4, 9)
    assert isinstance(layer.fc1, nn.Linear) and isinstance(layer.fc2, nn.Linear), '应暴露 fc1、fc2 两层 Linear'
    assert (layer.fc1.in_features, layer.fc1.out_features, layer.fc2.in_features, layer.fc2.out_features) == (4, 9, 9, 4), '线性层维度不符'
    x = torch.randn(2, 3, 4, requires_grad=True)
    check(layer(x), layer.fc2(F.gelu(layer.fc1(x))), '逐 token 的 Linear-GELU-Linear')
    layer(x).sum().backward()
    assert x.grad is not None, '输入梯度应可反传'


TESTS = {
    '00-torch-basics': ('torch_basics', test_torch_basics),
    '01-attention': ('attention', test_attention),
    '02-normalization': ('normalization', test_normalization),
    '03-position-encoding': ('position_encoding', test_position_encoding),
    '04-sampling': ('sampling', test_sampling),
    '05-loss-functions': ('loss_functions', test_loss_functions),
    '06-optimizers': ('optimizers', test_optimizers),
    '07-reinforcement-learning': ('reinforcement_learning', test_reinforcement_learning),
    '08-efficient-training': ('efficient_training', test_efficient_training),
    '09-inference-optimization': ('inference_optimization', test_inference_optimization),
    '10-transformer-architecture': ('transformer_architecture', test_transformer_architecture),
}


def main():
    parser = argparse.ArgumentParser(description='本地手撕 LLM 测评')
    parser.add_argument('topic', nargs='?', default='all')
    parser.add_argument('page', nargs='?', help='章节内的具体 Markdown 页面；用于网页端逐题测评')
    args = parser.parse_args()
    torch.set_num_threads(1)
    passed = 0
    if args.page and args.topic != 'all' and args.topic not in TESTS:
        print(f'[失败] {args.topic}/{args.page}: 未知章节')
        return 1
    if args.page and args.topic in TESTS:
        legacy_pages = {
            '00-torch-basics': 'torch-basics', '01-attention': 'scaled-dot-product-attention',
            '02-normalization': 'rms-norm', '03-position-encoding': 'rope',
            '04-sampling': 'top-p', '05-loss-functions': 'cross-entropy',
            '06-optimizers': 'adamw', '07-reinforcement-learning': 'gae',
            '08-efficient-training': 'lora', '09-inference-optimization': 'kv-cache',
            '10-transformer-architecture': 'ffn',
        }
        if args.page != legacy_pages[args.topic]:
            return generic_page_check(args.topic, args.page)
    selected = TESTS if args.topic == 'all' else {args.topic: TESTS[args.topic]}
    for topic, (module, test) in selected.items():
        try:
            test(importlib.import_module(f'solutions.{module}'))
        except NotImplementedError:
            print(f'[未实现] {topic}: 请打开 solutions/{module}.py 完成接口')
        except (AssertionError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
            print(f'[失败] {topic}: {exc}')
        except Exception:
            print(f'[异常] {topic}:')
            traceback.print_exc(limit=3)
        else:
            print(f'[通过] {topic}')
            passed += 1
    print(f'通过 {passed}/{len(selected)} 题')
    return 0 if passed == len(selected) else 1


def generic_page_check(topic, page):
    path = Path(__file__).resolve().parents[1] / 'solutions' / 'pages' / f'{topic}__{page}.py'
    if not path.is_file():
        print(f'[未实现] {topic}/{page}: 练习文件尚未初始化')
        return 1
    spec = importlib.util.spec_from_file_location(f'page_{topic}_{page}', path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        public = [value for name, value in vars(module).items() if not name.startswith('_') and callable(value)]
        if not public:
            raise AssertionError('必须实现一个公开函数或类')
        source = path.read_text(encoding='utf-8')
        if 'NotImplementedError' in source:
            raise NotImplementedError
    except NotImplementedError:
        print(f'[未实现] {topic}/{page}: 请根据对应 Markdown 完成核心接口')
        return 1
    except Exception as exc:
        print(f'[失败] {topic}/{page}: 导入或接口检查失败：{exc}')
        return 1
    print(f'[通过] {topic}/{page}: 独立练习文件可导入，公开接口已实现')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
