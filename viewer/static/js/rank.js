// Ranking page: drag-and-drop thumbnail ordering + hover preview + click-to-comment
// Saved to working/<run_id>/human/ranking.json

const projectName = window.location.pathname.split('/')[2];
const runId = window.location.pathname.split('/')[4];

let iterations = [];
let ranking = []; // array of iteration_number strings
let notes = {};   // { "<iteration_number>": "text" }
let selected = null; // iteration_number string
let hovered = null;  // iteration_number string

function esc(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

async function fetchIterations() {
  const resp = await fetch(`/api/project/${projectName}/run/${runId}/iterations`);
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return data.iterations || [];
}

async function fetchExistingRanking() {
  const resp = await fetch(`/api/project/${projectName}/run/${runId}/human/ranking`);
  const data = await resp.json();
  return data || { ranking: [], notes: {} };
}

function defaultRankingFromIterations(iters) {
  // Default: best score first, then iteration number descending as tie-breaker.
  const sorted = [...iters].sort((a, b) => {
    const sa = a.evaluation?.overall_score || 0;
    const sb = b.evaluation?.overall_score || 0;
    if (sb !== sa) return sb - sa;
    return (b.iteration_number || 0) - (a.iteration_number || 0);
  });
  return sorted.map(it => String(it.iteration_number));
}

function iterationByNumberMap(iters) {
  const m = {};
  iters.forEach(it => {
    if (it && typeof it.iteration_number === 'number') {
      m[String(it.iteration_number)] = it;
    }
  });
  return m;
}

function setPreview(iterStr) {
  const byNum = iterationByNumberMap(iterations);
  const it = iterStr ? byNum[iterStr] : null;

  const img = document.getElementById('previewImg');
  const rankEl = document.getElementById('previewRank');
  const scoreEl = document.getElementById('previewScore');
  const iterEl = document.getElementById('previewIter');
  if (!img || !rankEl || !scoreEl || !iterEl) return;

  if (!it) {
    img.removeAttribute('src');
    img.alt = 'Preview image';
    rankEl.textContent = '#–';
    scoreEl.textContent = '–%';
    iterEl.textContent = 'Iter –';
    return;
  }

  const idx = ranking.indexOf(iterStr);
  const score = it.evaluation?.overall_score ?? 0;
  img.src = `/api/project/${projectName}/run/${runId}/image/iteration_${iterStr}.png`;
  img.alt = `Iteration ${iterStr}`;
  rankEl.textContent = idx >= 0 ? `#${idx + 1}` : '#–';
  scoreEl.textContent = `${score}%`;
  iterEl.textContent = `Iter ${iterStr}`;
}

function setSelected(iterStr) {
  selected = iterStr;
  hovered = null;
  setPreview(iterStr);

  // Update editor value
  const editor = document.getElementById('notesEditor');
  if (editor) {
    editor.value = notes[iterStr] || '';
    editor.focus();
  }
  highlightSelectedThumb();
}

function highlightSelectedThumb() {
  const grid = document.getElementById('thumbGrid');
  if (!grid) return;
  grid.querySelectorAll('.thumb').forEach(el => {
    if (el.dataset.iteration === selected) el.classList.add('selected');
    else el.classList.remove('selected');
  });
}

let dragSrcEl = null;

function addDnDHandlersThumb(el) {
  el.addEventListener('dragstart', (e) => {
    dragSrcEl = el;
    el.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', el.dataset.iteration);
  });

  el.addEventListener('dragend', () => {
    el.classList.remove('dragging');
    dragSrcEl = null;
  });

  el.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  });

  el.addEventListener('drop', (e) => {
    e.preventDefault();
    const src = e.dataTransfer.getData('text/plain');
    const tgt = el.dataset.iteration;
    if (!src || !tgt || src === tgt) return;
    moveRankingItem(src, tgt);
  });
}

function moveRankingItem(srcIter, targetIter) {
  const srcIdx = ranking.indexOf(srcIter);
  const tgtIdx = ranking.indexOf(targetIter);
  if (srcIdx < 0 || tgtIdx < 0) return;
  ranking.splice(srcIdx, 1);
  ranking.splice(tgtIdx, 0, srcIter);
  renderThumbGrid();
  setStatus('Unsaved changes', 'warn');
}

function renderThumbGrid() {
  const grid = document.getElementById('thumbGrid');
  if (!grid) return;
  const byNum = iterationByNumberMap(iterations);
  grid.innerHTML = '';

  ranking.forEach((numStr, idx) => {
    const it = byNum[numStr];
    if (!it) return;
    const score = it.evaluation?.overall_score ?? 0;

    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'thumb';
    el.draggable = true;
    el.dataset.iteration = numStr;
    el.title = `#${idx + 1} • Iter ${numStr} • ${score}%`;

    el.innerHTML = `
      <div class="thumb-badges">
        <span class="thumb-rank">#${idx + 1}</span>
        <span class="thumb-score">${esc(String(score))}%</span>
      </div>
      <img class="thumb-img" src="/api/project/${projectName}/run/${runId}/image/iteration_${numStr}.png" alt="Iteration ${esc(numStr)}">
      <div class="thumb-label">Iter ${esc(numStr)}</div>
    `;

    addDnDHandlersThumb(el);

    el.addEventListener('mouseenter', () => {
      hovered = numStr;
      setPreview(numStr);
    });
    el.addEventListener('mouseleave', () => {
      hovered = null;
      setPreview(selected);
    });

    el.addEventListener('click', () => setSelected(numStr));

    grid.appendChild(el);
  });

  // Keep selection valid
  if (!selected && ranking.length > 0) {
    setSelected(ranking[0]);
  } else if (selected && !ranking.includes(selected) && ranking.length > 0) {
    setSelected(ranking[0]);
  } else {
    setPreview(hovered || selected);
    highlightSelectedThumb();
  }
}

function setStatus(text, kind) {
  const el = document.getElementById('saveStatus');
  if (!el) return;
  el.textContent = text || '';
  el.classList.remove('ok', 'warn', 'err');
  if (kind) el.classList.add(kind);
}

async function save() {
  const payload = { ranking, notes };
  setStatus('Saving…', null);
  const resp = await fetch(`/api/project/${projectName}/run/${runId}/human/ranking`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  setStatus('Saved', 'ok');
}

async function init() {
  try {
    iterations = await fetchIterations();
    const existing = await fetchExistingRanking();

    const allNums = iterations
      .map(it => String(it.iteration_number))
      .filter(x => x && x !== 'undefined' && x !== 'null');

    // If existing ranking is valid, keep it; otherwise default ranking.
    const existingRanking = Array.isArray(existing.ranking) ? existing.ranking.map(String) : [];
    const setAll = new Set(allNums);
    const filteredExisting = existingRanking.filter(x => setAll.has(x));
    const missing = allNums.filter(x => !filteredExisting.includes(x));

    if (filteredExisting.length > 0) {
      ranking = filteredExisting.concat(missing);
    } else {
      ranking = defaultRankingFromIterations(iterations);
    }

    notes = (existing.notes && typeof existing.notes === 'object') ? existing.notes : {};
    renderThumbGrid();

    document.getElementById('rankLoading').classList.add('is-hidden');
    document.getElementById('rankContent').classList.remove('is-hidden');
    setStatus(existing.updated_at ? 'Loaded existing ranking' : '', null);

    // Notes editor (writes to selected)
    const editor = document.getElementById('notesEditor');
    if (editor) {
      editor.addEventListener('input', (e) => {
        if (!selected) return;
        notes[selected] = e.target.value;
        setStatus('Unsaved changes', 'warn');
      });
    }

    document.getElementById('saveRanking').addEventListener('click', async () => {
      try {
        await save();
      } catch (e) {
        setStatus(`Save failed: ${e.message}`, 'err');
      }
    });
  } catch (e) {
    document.getElementById('rankLoading').textContent = `Error: ${e.message}`;
  }
}

init();

