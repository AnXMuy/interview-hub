(() => {
  const state = {
    catalog: [], module: null, page: null, lessonPages: [], lessonIndex: 0,
    explainIndex: 0, exercise: null, savedCode: '',
    passed: new Set(JSON.parse(localStorage.getItem('llm-passed-lessons') || '[]')),
  };
  const $ = (id) => document.getElementById(id);
  const icon = (name) => `<img src="/icons/${name}.svg" alt="" aria-hidden="true">`;
  const moduleOf = (id) => state.catalog.find((item) => item.id === id);
  const pageOf = (module, page) => moduleOf(module)?.pages.find((item) => item.id === page);
  const keyOf = (module, page) => `${module}/${page}`;
  const currentPage = () => pageOf(state.module, state.page);

  function renderIcons() { document.querySelectorAll('[data-icon]').forEach((node) => { node.innerHTML = icon(node.dataset.icon); }); }
  function isPassed(module, page) { return state.passed.has(keyOf(module, page)); }
  function lessonNumber() { return String(state.lessonIndex + 1).padStart(2, '0'); }

  function renderNav(filter = '') {
    const query = filter.trim().toLowerCase();
    $('course-nav').innerHTML = state.catalog.map((mod, index) => {
      const pages = mod.pages.filter((page) => !query || `${mod.title} ${page.title}`.toLowerCase().includes(query));
      if (query && !pages.length) return '';
      const done = mod.pages.filter((page) => isPassed(mod.id, page.id)).length;
      return `<details class="module" data-module="${mod.id}" ${mod.id === state.module || query ? 'open' : ''}>
        <summary class="module-summary"><span class="chevron">${icon('chevron-right')}</span><span class="module-number">${String(index).padStart(2, '0')}</span><span class="module-title">${mod.title}</span><span class="module-score">${done}/${mod.pages.length}</span></summary>
        <div class="page-list">${pages.map((page) => `<a class="page-link ${mod.id === state.module && page.id === state.page ? 'active' : ''} ${isPassed(mod.id, page.id) ? 'passed' : ''}" href="#/${mod.id}/${page.id}"><span class="graded-dot"></span><span>${page.title}</span></a>`).join('')}</div>
      </details>`;
    }).join('');
    $('course-nav').querySelectorAll('.page-link').forEach((link) => link.addEventListener('click', () => { if (innerWidth <= 800) $('sidebar').classList.remove('open'); }));
    $('chapter-count').textContent = `${state.catalog.reduce((n, mod) => n + mod.pages.length, 0)} 个模块`;
  }

  function renderProgress() {
    const total = state.catalog.reduce((n, mod) => n + mod.pages.length, 0);
    const done = state.catalog.reduce((n, mod) => n + mod.pages.filter((page) => isPassed(mod.id, page.id)).length, 0);
    $('progress-bar').style.width = `${total ? done / total * 100 : 0}%`;
    $('course-count').textContent = `${String(done).padStart(2, '0')} / ${String(total).padStart(2, '0')}`;
  }

  function splitExplanation() {
    const article = $('document');
    const blocks = [...article.children];
    const pages = []; let current = [];
    blocks.forEach((block) => {
      if (/^H[12]$/.test(block.tagName) && current.length) { pages.push(current); current = []; }
      current.push(block);
    });
    if (current.length) pages.push(current);
    state.lessonPages = pages.length ? pages : [[...blocks]];
    state.explainIndex = 0;
    showExplanation();
  }

  function showExplanation() {
    const article = $('document');
    [...article.children].forEach((child) => { child.hidden = true; });
    (state.lessonPages[state.explainIndex] || []).forEach((child) => { child.hidden = false; });
    $('page-indicator').textContent = `讲解 ${state.explainIndex + 1} / ${state.lessonPages.length}`;
    $('prev-page').disabled = state.explainIndex === 0;
    $('next-page').disabled = state.explainIndex === state.lessonPages.length - 1;
  }

  async function loadDocument(module, page) {
    const response = await fetch(`/api/document/${module}/${page}`);
    if (!response.ok) throw new Error('无法载入文档');
    $('document').innerHTML = (await response.json()).html;
    if (window.renderMathInElement) renderMathInElement($('document'), { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false }], throwOnError: false });
    $('document').querySelectorAll('a[href^="#/"]').forEach((link) => link.addEventListener('click', (event) => { event.preventDefault(); navigate(link.getAttribute('href').slice(2)); }));
    splitExplanation();
  }

  async function loadExercise(module, page) {
    const response = await fetch(`/api/exercise/${module}/${page}`);
    if (!response.ok) throw new Error('无法载入练习');
    const data = await response.json(); state.exercise = data; state.savedCode = data.code;
    $('code-editor').value = data.code; $('file-name').textContent = data.file; $('exercise-title').textContent = data.name; $('exercise-signature').textContent = data.signature;
    $('exercise-number').textContent = `${state.lessonIndex + 1} / ${String(moduleOf(module).pages.length).padStart(2, '0')}`;
    $('section-number').textContent = `${lessonNumber()} / ${String(state.catalog.length).padStart(2, '0')}`;
    $('workbench-subtitle').textContent = data.legacy ? '精确测评 · 只修改当前接口' : '独立练习 · 可导入检查';
    updateLineNumbers(); setDirty(false); setResult('idle', '等待运行', '完成当前模块后运行本地测评，结果会显示在这里。');
    $('open-exercise-doc').onclick = () => { $('reader').focus(); };
  }

  async function navigate(path, replace = false) {
    const [module, page] = path.split('/'); const mod = moduleOf(module); if (!mod) return;
    const lesson = pageOf(module, page) || mod.pages[0]; state.module = module; state.page = lesson.id; state.lessonIndex = mod.pages.findIndex((item) => item.id === lesson.id);
    if (replace) history.replaceState(null, '', `#/${module}/${lesson.id}`); else if (location.hash !== `#/${module}/${lesson.id}`) history.pushState(null, '', `#/${module}/${lesson.id}`);
    $('crumb-module').textContent = mod.title; $('crumb-page').textContent = lesson.title; $('reader-topic').textContent = `${mod.pages.length} 个模块 · 当前第 ${state.lessonIndex + 1} 个`;
    renderNav($('search').value); renderProgress(); renderLessonControls();
    $('document').innerHTML = '<div class="loading">正在载入讲解…</div>'; await Promise.all([loadDocument(module, lesson.id), loadExercise(module, lesson.id)]);
  }

  function renderLessonControls() {
    const mod = moduleOf(state.module); const first = state.lessonIndex === 0; const last = state.lessonIndex === mod.pages.length - 1;
    $('lesson-indicator').textContent = `模块 ${state.lessonIndex + 1} / ${mod.pages.length}`; $('prev-lesson').disabled = first; $('next-lesson').disabled = last;
  }
  function moveLesson(delta) { const mod = moduleOf(state.module); const next = mod.pages[state.lessonIndex + delta]; if (next) navigate(`${state.module}/${next.id}`); }
  function moveExplanation(delta) { const next = state.explainIndex + delta; if (next >= 0 && next < state.lessonPages.length) { state.explainIndex = next; showExplanation(); } }
  function updateLineNumbers() { $('line-numbers').innerHTML = $('code-editor').value.split('\n').map((_, index) => `<div>${index + 1}</div>`).join(''); }
  function setDirty(dirty) { $('unsaved-dot').classList.toggle('show', dirty); $('save-status').textContent = dirty ? '有未保存的修改' : '代码只保存在本机'; }
  function setResult(kind, label, output) { $('result-badge').textContent = label; $('result-badge').className = `result-badge ${kind}`; $('result-output').textContent = output; }
  function toast(message) { const node = $('toast'); node.textContent = message; node.classList.add('show'); setTimeout(() => node.classList.remove('show'), 2600); }

  async function saveCode(showToast = true) {
    const code = $('code-editor').value; const response = await fetch(`/api/exercise/${state.module}/${state.page}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || '保存失败'); state.savedCode = code; setDirty(false); if (showToast) toast('当前模块代码已保存到本机');
  }
  async function runGrade() {
    try { if ($('code-editor').value !== state.savedCode) await saveCode(false); } catch (error) { setResult('fail', '保存失败', error.message); return; }
    const button = $('run-code'); button.disabled = true; button.querySelector('span').textContent = '测评中…'; setResult('pending', '运行中', '正在运行当前模块的本地测评，请稍候…');
    try {
      const response = await fetch(`/api/grade/${state.module}/${state.page}`, { method: 'POST' }); const result = await response.json(); if (!response.ok) throw new Error(result.detail || '测评失败');
      const kind = result.status === 'passed' ? 'pass' : result.status === 'pending' ? 'pending' : 'fail'; const label = result.status === 'passed' ? '通过' : result.status === 'pending' ? '未实现' : result.status === 'timeout' ? '超时' : '有错误'; setResult(kind, label, result.output || '没有输出');
      if (result.status === 'passed') { state.passed.add(keyOf(state.module, state.page)); localStorage.setItem('llm-passed-lessons', JSON.stringify([...state.passed])); renderNav($('search').value); renderProgress(); toast('当前模块已通过'); }
    } catch (error) { setResult('fail', '请求失败', error.message); } finally { button.disabled = false; button.querySelector('span').textContent = '运行测评'; }
  }

  async function boot() {
    renderIcons(); const response = await fetch('/api/catalog'); state.catalog = (await response.json()).modules;
    const [hashModule, hashPage] = location.hash.replace(/^#\/?/, '').split('/'); const first = state.catalog[0]; const mod = moduleOf(hashModule) || first; await navigate(`${mod.id}/${hashPage || mod.pages[0].id}`, true);
    $('search').addEventListener('input', (event) => renderNav(event.target.value)); $('toggle-sidebar').onclick = () => $('sidebar').classList.toggle('open'); $('close-sidebar').onclick = () => $('sidebar').classList.remove('open');
    $('prev-page').onclick = () => moveExplanation(-1); $('next-page').onclick = () => moveExplanation(1); $('prev-lesson').onclick = () => moveLesson(-1); $('next-lesson').onclick = () => moveLesson(1);
    $('toggle-editor').onclick = () => { const hidden = $('workbench').style.display === 'none'; $('workbench').style.display = hidden ? '' : 'none'; $('toggle-editor').innerHTML = icon(hidden ? 'panel-right-close' : 'panel-right-open'); };
    $('show-reader').onclick = () => { $('workspace').classList.remove('editor-view'); $('show-reader').classList.add('active'); $('show-editor').classList.remove('active'); }; $('show-editor').onclick = () => { $('workspace').classList.add('editor-view'); $('show-editor').classList.add('active'); $('show-reader').classList.remove('active'); };
    $('code-editor').addEventListener('input', () => { updateLineNumbers(); setDirty($('code-editor').value !== state.savedCode); }); $('code-editor').addEventListener('scroll', () => { $('line-numbers').scrollTop = $('code-editor').scrollTop; }); $('code-editor').addEventListener('keydown', (event) => { if (event.key === 'Tab') { event.preventDefault(); const start = event.target.selectionStart; event.target.value = event.target.value.slice(0, start) + '    ' + event.target.value.slice(event.target.selectionEnd); event.target.selectionStart = event.target.selectionEnd = start + 4; updateLineNumbers(); setDirty(true); } });
    $('save-code').onclick = () => saveCode().catch((error) => toast(error.message)); $('reset-code').onclick = () => loadExercise(state.module, state.page).catch((error) => toast(error.message)); $('run-code').onclick = runGrade;
    const syncHash = () => { const [module, page] = location.hash.replace(/^#\/?/, '').split('/'); if (module) navigate(`${module}/${page || moduleOf(module)?.pages[0].id}`, true); };
    window.addEventListener('popstate', syncHash); window.addEventListener('hashchange', syncHash); window.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') { event.preventDefault(); $('save-code').click(); } if (event.key === 'ArrowLeft' && !['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) moveLesson(-1); if (event.key === 'ArrowRight' && !['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) moveLesson(1); if (event.key === '/' && document.activeElement !== $('code-editor') && document.activeElement !== $('search')) { event.preventDefault(); $('search').focus(); } });
  }
  boot().catch((error) => { $('document').innerHTML = `<div class="loading">${error.message}</div>`; });
})();
