"""Loopback-only document browser and bounded local exercise runner."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import subprocess
import sys
import threading
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from .catalog import DOCS, EXERCISES, ROOT, SOLUTIONS, catalog, get_exercise

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
MARKDOWN = MarkdownIt('commonmark', {'html': False}).enable('table')
STATIC = ROOT / 'app' / 'static'
KATEX = ROOT / 'node_modules' / 'katex' / 'dist'
LOCK = threading.Lock()
RUNNERS = ThreadPoolExecutor(max_workers=1)
LIMIT = 64_000
if KATEX.is_dir():
    app.mount('/vendor/katex', StaticFiles(directory=KATEX), name='katex')


class Source(BaseModel):
    code: str


def validate_local(request: Request):
    # Loopback only: arbitrary Python supplied to solutions/ must never be a public API.
    if request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        raise HTTPException(403, '练习接口只允许本机访问')
    origin = request.headers.get('origin')
    if origin and (urlsplit(origin).hostname not in ('127.0.0.1', 'localhost') or
                   urlsplit(origin).netloc != request.headers.get('host')):
        raise HTTPException(403, '跨站请求被拒绝')


def resolve_page(module: str, page: str) -> Path:
    if not re.fullmatch(r'[a-z0-9-]+|README', page):
        raise HTTPException(404, '未找到章节')
    path = DOCS / module / f'{page}.md'
    if not path.is_file():
        raise HTTPException(404, '未找到文档')
    return path


def render_document(source: Path) -> str:
    # Resolve links to other local docs into internal routes; no remote scripts or HTML.
    tokens = MARKDOWN.parse(source.read_text(encoding='utf-8'))
    for token in tokens:
        if token.type != 'inline' or not token.children:
            continue
        for child in token.children:
            if child.type != 'link_open':
                continue
            target = child.attrGet('href')
            if not target or target.startswith(('#', 'http:', 'https:')):
                continue
            url = urlsplit(target)
            resolved = (source.parent / unquote(url.path)).resolve()
            if resolved.is_relative_to(DOCS) and resolved.suffix == '.md' and resolved.is_file():
                child.attrSet('href', '#/' + resolved.parent.name + '/' + resolved.stem +
                              (('#' + url.fragment) if url.fragment else ''))
            else:
                child.attrSet('href', '#')
    return MARKDOWN.renderer.render(tokens, MARKDOWN.options, {})


@app.get('/api/catalog')
def get_catalog():
    return {'modules': catalog()}


@app.get('/api/document/{module}/{page}')
def get_document(module: str, page: str):
    return {'html': render_document(resolve_page(module, page))}


def solution_path(exercise):
    path = SOLUTIONS / exercise['file'].removeprefix('solutions/')
    if not path.exists() and not exercise.get('legacy'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '"""本模块独立练习：保留一个函数或类的实现，不要修改接口。"""\n\n'
            'def solve(*args, **kwargs):\n'
            '    """请根据对应 Markdown 完成核心逻辑。"""\n'
            '    raise NotImplementedError\n', encoding='utf-8')
    return path


@app.get('/api/exercise/{module}/{page}')
def get_page_exercise(module: str, page: str, request: Request):
    validate_local(request)
    exercise = get_exercise(module, page)
    if not exercise:
        raise HTTPException(404, '未找到练习')
    path = solution_path(exercise)
    return {'code': path.read_text(encoding='utf-8'), 'file': exercise['file'],
            'signature': exercise['signature'], 'name': exercise['name'],
            'legacy': exercise.get('legacy', False)}


@app.put('/api/exercise/{module}/{page}')
def save_exercise(module: str, page: str, source: Source, request: Request):
    validate_local(request)
    exercise = get_exercise(module, page)
    if not exercise:
        raise HTTPException(404, '未找到练习')
    if len(source.code.encode('utf-8')) > LIMIT:
        raise HTTPException(413, '代码不能超过 64 KB')
    try:
        compile(source.code, exercise.file, 'exec')
    except SyntaxError as exc:
        raise HTTPException(422, f'第 {exc.lineno} 行语法错误：{exc.msg}') from exc
    path = solution_path(exercise)
    with LOCK:
        temporary = path.with_suffix('.py.tmp')
        temporary.write_text(source.code, encoding='utf-8')
        os.replace(temporary, path)
    return {'saved': True}


def run_check(module: str, page: str):
    with LOCK:
        try:
            process = subprocess.run([sys.executable, '-m', 'grader', module, page], cwd=ROOT,
                                     capture_output=True, text=True, timeout=25,
                                     env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'output': '测评超时（25 秒）。检查死循环或耗时计算。'}
    output = (process.stdout + process.stderr)[-12_000:]
    if '[通过]' in output and process.returncode == 0:
        status = 'passed'
    elif '[未实现]' in output:
        status = 'pending'
    else:
        status = 'failed'
    return {'status': status, 'output': output, 'exit_code': process.returncode}


@app.post('/api/grade/{module}/{page}')
def grade(module: str, page: str, request: Request):
    validate_local(request)
    if not get_exercise(module, page):
        raise HTTPException(404, '未找到练习')
    try:
        return RUNNERS.submit(run_check, module, page).result(timeout=30)
    except TimeoutError:
        raise HTTPException(503, '测评队列繁忙') from None


@app.get('/')
def home():
    return FileResponse(STATIC / 'index.html')


@app.get('/assets/{name}')
def asset(name: str):
    if name not in ('app.css', 'app.js'):
        raise HTTPException(404)
    return FileResponse(STATIC / name, media_type='text/css' if name.endswith('.css') else 'text/javascript')


@app.get('/icons/{name}')
def icon(name: str):
    name = name.removesuffix('.svg')
    if name not in ('book-open', 'search', 'chevron-down', 'chevron-right',
                    'panel-left-close', 'panel-left-open', 'panel-right-close',
                    'panel-right-open', 'play', 'save', 'check', 'circle-check',
                    'terminal', 'arrow-right', 'arrow-left', 'menu', 'x', 'code',
                    'file-text', 'rotate-ccw', 'external-link', 'circle-help', 'list-filter'):
        raise HTTPException(404)
    return FileResponse(ROOT / 'node_modules' / 'lucide-static' / 'icons' / (name + '.svg'),
                        media_type='image/svg+xml')
