"""Single source of truth for modules, lesson pages, and local exercises."""
from dataclasses import dataclass
from pathlib import Path
import re

from .exercise_specs import get_spec

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SOLUTIONS = ROOT / 'solutions'


@dataclass(frozen=True)
class Exercise:
    module: str
    file: str
    page: str
    name: str
    signature: str


EXERCISES = {
    '00-torch-basics/torch-basics': Exercise('00-torch-basics', 'torch_basics.py', 'torch-basics.md', 'Masked Mean', 'masked_mean(x, mask)'),
    '01-attention/scaled-dot-product-attention': Exercise('01-attention', 'attention.py', 'scaled-dot-product-attention.md', 'Scaled Dot-Product Attention', 'scaled_dot_product_attention(q, k, v, mask)'),
    '02-normalization/rms-norm': Exercise('02-normalization', 'normalization.py', 'rms-norm.md', 'RMSNorm', 'RMSNorm(dim, eps)'),
    '03-position-encoding/rope': Exercise('03-position-encoding', 'position_encoding.py', 'rope.md', 'RoPE', 'apply_rope(x, offset, base)'),
    '04-sampling/top-p': Exercise('04-sampling', 'sampling.py', 'top-p.md', 'Top-p Sampling', 'top_p_filter(logits, p)'),
    '05-loss-functions/cross-entropy': Exercise('05-loss-functions', 'loss_functions.py', 'cross-entropy.md', 'Cross Entropy', 'cross_entropy(logits, targets, ignore_index)'),
    '06-optimizers/adamw': Exercise('06-optimizers', 'optimizers.py', 'adamw.md', 'AdamW', 'adamw_step(param, grad, m, v, step, ...)'),
    '07-reinforcement-learning/gae': Exercise('07-reinforcement-learning', 'reinforcement_learning.py', 'gae.md', 'GAE', 'compute_gae(rewards, values, dones, gamma, lam)'),
    '08-efficient-training/lora': Exercise('08-efficient-training', 'efficient_training.py', 'lora.md', 'LoRALinear', 'LoRALinear(base, rank, alpha)'),
    '09-inference-optimization/kv-cache': Exercise('09-inference-optimization', 'inference_optimization.py', 'kv-cache.md', 'KV Cache', 'KVCache().append(k, v) / reset()'),
    '10-transformer-architecture/ffn': Exercise('10-transformer-architecture', 'transformer_architecture.py', 'ffn.md', 'Feed Forward Network', 'FeedForward(dim, hidden_dim)'),
}


def catalog():
    modules = []
    for folder in sorted(DOCS.iterdir()):
        if not folder.is_dir() or not (folder / 'README.md').is_file():
            continue
        module = folder.name
        content = (folder / 'README.md').read_text(encoding='utf-8')
        title = re.search(r'^#\s+(.+)$', content, re.M).group(1).strip()
        pages = []
        for label, filename in re.findall(r'^- \[([^\]]+)\]\(([^)]+\.md)\)', content, re.M):
            slug = filename.removesuffix('.md')
            pages.append({'id': slug, 'title': label, 'graded': True,
                          'exercise': exercise_payload(EXERCISES.get(f'{module}/{slug}'), module, filename, label)})
        if not pages:
            filename = 'torch-basics.md'
            pages = [{'id': 'torch-basics', 'title': 'PyTorch 常用语法', 'graded': True,
                      'exercise': exercise_payload(EXERCISES['00-torch-basics/torch-basics'], module, filename, 'PyTorch 常用语法')}]
        modules.append({'id': module, 'title': title, 'pages': pages})
    return modules


def exercise_payload(exercise, module, filename, label):
    slug = filename.removesuffix('.md')
    spec = get_spec(module, slug, label)
    if exercise:
        return {'id': f'{module}/{slug}', 'name': spec['name'],
                'signature': spec['signature'], 'file': 'solutions/' + exercise.file,
                'page': slug, 'legacy': True, 'spec': spec}
    return {'id': f'{module}/{slug}', 'name': spec['name'],
            'signature': spec['signature'],
            'file': f'solutions/pages/{module}__{slug}.py', 'page': slug,
            'legacy': False, 'spec': spec}


def get_exercise(module, page):
    for item in catalog():
        if item['id'] == module:
            for lesson in item['pages']:
                if lesson['id'] == page:
                    return lesson['exercise']
    return None
