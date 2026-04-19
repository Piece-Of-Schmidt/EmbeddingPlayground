// GloVe Explorer — main application controller
// Uses ES modules; requires a local HTTP server (python3 -m http.server 8080)

import { parse, describeQuery, MODE_FARTHEST } from './lib/parser.js';
import { computeQuery, knn, kfarthest }        from './lib/algebra.js';
import { loadEmbeddings, loadModelList }        from './lib/embeddings.js';
import { pca2D }                                from './lib/pca.js';

// ─── State ───────────────────────────────────────────────────────────────────

let store = null;
let activeTab = 'neighbors';
const chartInstances = {};
const queryHistory = [];   // [{expr, timestamp}], max 20 entries

// ─── Boot ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Register datalabels globally but off by default — only Scatter enables it
  Chart.register(ChartDataLabels);
  Chart.defaults.set('plugins.datalabels', { display: false });

  setupTabs();
  setupInput();
  setupSimilarity();
  setupHistory();
  startLoading();
});

async function startLoading(modelId) {
  const isSwitch  = !!modelId;  // true = model switch, false = initial boot
  const overlay   = document.getElementById('loading-overlay');
  const bar       = document.getElementById('loading-bar');
  const loadMsg   = document.getElementById('loading-msg');
  const errEl     = document.getElementById('loading-error');

  // Always reset error state before attempting
  errEl.textContent = '';
  errEl.classList.add('hidden');
  bar.style.width = '0%';

  if (isSwitch) {
    _setStatus('loading', 'Switching model …');
  } else {
    overlay.classList.remove('hidden');
  }

  try {
    if (!modelId) {
      const models = await loadModelList();
      _populateModelDropdown(models);
      modelId = models[0].id;
    }

    console.log('[GloVe] Loading model:', modelId);
    store = await loadEmbeddings(modelId, (fraction) => {
      bar.style.width = Math.round(fraction * 100) + '%';
      if (!isSwitch) loadMsg.textContent = `Loading embeddings … ${Math.round(fraction * 100)}%`;
    });

    // Update header info
    document.getElementById('vocab-size').textContent = store.size.toLocaleString('en-US');
    document.getElementById('dims-count').textContent = store.dims;
    console.log('[GloVe] Loaded:', store.modelPath, '— words:', store.size, 'dims:', store.dims);
    const pathEl = document.getElementById('model-path');
    if (pathEl) pathEl.textContent = store.modelPath ?? '';
    _setStatus('ready', 'Ready');

    if (!isSwitch) overlay.classList.add('hidden');

    // Re-run last query with new model if there was one
    const input = document.getElementById('query-input');
    if (input.value.trim()) runQuery(input.value);
    else input.focus();

  } catch (err) {
    _setStatus('error', 'Error');
    if (isSwitch) {
      // Show error inline in header, not as blocking overlay
      const pathEl = document.getElementById('model-path');
      if (pathEl) pathEl.textContent = 'Load failed: ' + err.message;
    } else {
      loadMsg.textContent = '';
      errEl.textContent = err.message;
      errEl.classList.remove('hidden');
    }
  }
}

function _setStatus(state, text) {
  const dot  = document.getElementById('status-dot');
  const span = document.getElementById('status-text');
  dot.className  = 'status-dot' + (state === 'ready' ? ' ready' : state === 'error' ? ' error' : '');
  span.textContent = text;
}

function _populateModelDropdown(models) {
  const select = document.getElementById('model-select');
  if (!select) return;
  select.innerHTML = '';
  for (const { id, label } of models) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = label;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => {
    console.log('[GloVe] Model dropdown changed to:', select.value);
    clearResults();
    startLoading(select.value);
  });
  // Show the model selector only if there's more than one model
  const wrapper = document.getElementById('model-select-wrapper');
  if (wrapper) wrapper.style.display = models.length > 1 ? 'flex' : 'none';
}

// ─── Input & Query ────────────────────────────────────────────────────────────

function setupInput() {
  const input = document.getElementById('query-input');
  let debounceTimer = null;

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      clearTimeout(debounceTimer);
      runQuery(input.value);
    }
  });

  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => runQuery(input.value), 400);
  });

  // Example chips
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.query;
      runQuery(chip.dataset.query);
    });
  });
}

function runQuery(raw) {
  if (!store) return;
  raw = raw.trim();
  if (!raw) { clearResults(); return; }

  // Split on commas → one result object per expression
  const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
  const results = [];

  for (const part of parts) {
    const parsed = parse(part);
    if (parsed.terms.length === 0) continue;

    const { resultVec, normalizedVec, steps, unknownWords } = computeQuery(parsed.terms, store);
    const queryWords = parsed.terms.map(t => t.word);
    const neighbors = parsed.mode === MODE_FARTHEST
      ? kfarthest(normalizedVec, store, 12, queryWords)
      : knn(normalizedVec, store, 12, queryWords);

    results.push({
      expr: describeQuery(parsed),
      parsed,
      resultVec,
      normalizedVec,
      steps,
      unknownWords,
      neighbors,
    });
  }

  if (results.length === 0) { clearResults(); return; }

  pushHistory(raw);
  renderNeighborsTab(results);
  renderStepsTab(results);
  renderVectorsTab(results);
  renderScatterTab(results);
  renderHeatmapTab(results);
}

function clearResults() {
  document.getElementById('tab-neighbors').innerHTML     = '<p class="empty-hint">Enter an expression above.</p>';
  document.getElementById('tab-steps').innerHTML         = '<p class="empty-hint">No calculation yet.</p>';
  document.getElementById('tab-vectors').innerHTML       = '<p class="empty-hint">Shows the raw embedding vectors for all query words and the result.</p>';
  document.getElementById('tab-scatter').innerHTML       = '<p class="empty-hint">2D PCA projection of query words, result, and neighbors.</p>';
  document.getElementById('tab-heatmap').innerHTML       = '<p class="empty-hint">Heatmap of all dimensions. Blue = positive, Red = negative.</p>';
  destroyChart('chart-vectors');
  destroyChart('chart-scatter');
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────

function setupTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const pane = document.getElementById('tab-' + btn.dataset.tab);
      if (pane) pane.classList.add('active');
      activeTab = btn.dataset.tab;
    });
  });
}

// ─── Sidebar: Query History ───────────────────────────────────────────────────

function setupHistory() {
  document.getElementById('history-clear').addEventListener('click', () => {
    queryHistory.length = 0;
    renderHistory();
  });
}

function pushHistory(expr) {
  // Avoid consecutive duplicates
  if (queryHistory.length > 0 && queryHistory[0].expr === expr) return;
  queryHistory.unshift({ expr });
  if (queryHistory.length > 20) queryHistory.pop();
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById('history-list');
  if (queryHistory.length === 0) {
    list.innerHTML = '<p class="empty-hint">No searches yet.</p>';
    return;
  }
  list.innerHTML = queryHistory.map((h, i) =>
    `<li class="history-item" data-index="${i}">${_esc(h.expr)}</li>`
  ).join('');
  list.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const expr = queryHistory[+item.dataset.index].expr;
      document.getElementById('query-input').value = expr;
      runQuery(expr);
    });
  });
}

// ─── Sidebar: Similarity Calculator ──────────────────────────────────────────

function setupSimilarity() {
  const inputA = document.getElementById('sim-a');
  const inputB = document.getElementById('sim-b');
  let timer = null;

  const compute = () => {
    if (!store) return;
    const rawA = inputA.value.trim();
    const rawB = inputB.value.trim();
    const hint  = document.getElementById('sim-hint');
    const score = document.getElementById('sim-score');
    const bar   = document.getElementById('sim-bar');

    if (!rawA || !rawB) {
      hint.textContent = 'Enter two words or expressions';
      score.textContent = '—';
      bar.style.width = '0%';
      bar.className = 'sim-bar';
      return;
    }

    // Each side can be a full expression (e.g. "king - man + woman")
    const vecA = _resolveSimVec(rawA);
    const vecB = _resolveSimVec(rawB);

    if (!vecA) { hint.textContent = `Unknown: "${rawA}"`; score.textContent = '—'; bar.style.width = '0%'; return; }
    if (!vecB) { hint.textContent = `Unknown: "${rawB}"`; score.textContent = '—'; bar.style.width = '0%'; return; }

    // Cosine similarity (both vecs already normalized)
    let sim = 0;
    for (let d = 0; d < vecA.length; d++) sim += vecA[d] * vecB[d];
    sim = Math.max(-1, Math.min(1, sim));

    score.textContent = sim.toFixed(3);
    // Map [-1,1] → [0,100]% bar width, color by value
    const pct = Math.round(((sim + 1) / 2) * 100);
    bar.style.width = pct + '%';
    bar.className = 'sim-bar ' + (sim >= 0.5 ? 'high' : sim >= 0.2 ? 'mid' : 'low');
    hint.textContent = _simLabel(sim);
  };

  [inputA, inputB].forEach(el => {
    el.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(compute, 300); });
    el.addEventListener('keydown', e => { if (e.key === 'Enter') { clearTimeout(timer); compute(); } });
  });
}

function _resolveSimVec(raw) {
  const parsed = parse(raw);
  if (parsed.terms.length === 0) return null;
  const { normalizedVec, unknownWords } = computeQuery(parsed.terms, store);
  if (unknownWords.length === parsed.terms.length) return null; // all unknown
  return normalizedVec;
}

function _simLabel(sim) {
  if (sim > 0.85) return 'Nearly identical';
  if (sim > 0.65) return 'Very similar';
  if (sim > 0.45) return 'Somewhat related';
  if (sim > 0.20) return 'Weakly related';
  if (sim > -0.1) return 'Unrelated';
  return 'Opposite meaning';
}

// ─── Sidebar: stub (no longer used) ──────────────────────────────────────────

function renderSidebarNeighbors(_results) { /* removed — sidebar now shows history + sim */ }

// ─── Tab: Neighbors ───────────────────────────────────────────────────────────

function renderNeighborsTab(results) {
  const pane = document.getElementById('tab-neighbors');
  if (results.length === 1) {
    const r = results[0];
    pane.innerHTML = _neighborsHTML(r.neighbors, r.parsed.mode, r.unknownWords, r.expr);
  } else {
    pane.innerHTML = `<div class="neighbors-columns">${
      results.map(r => `<div class="neighbors-column">${_neighborsHTML(r.neighbors, r.parsed.mode, r.unknownWords, r.expr)}</div>`).join('')
    }</div>`;
  }
  _bindNeighborClicks(pane);
}

function _neighborsHTML(neighbors, mode, unknownWords, expr) {
  const modeLabel = mode === MODE_FARTHEST ? 'Farthest Neighbors' : 'Nearest Neighbors';
  const maxScore  = Math.max(...neighbors.map(n => Math.abs(n.score)), 0.01);

  let html = `<div class="neighbors-header">
    <span class="neighbors-expr">${_esc(expr)}</span>
    <span class="neighbors-mode ${mode === MODE_FARTHEST ? 'farthest' : 'nearest'}">${modeLabel}</span>
  </div>`;

  if (unknownWords.length > 0) {
    html += `<div class="unknown-words">Unknown words: ${unknownWords.map(_esc).map(w => `<code>${w}</code>`).join(', ')}</div>`;
  }

  html += '<ol class="neighbor-list">';
  for (const { word, score } of neighbors) {
    const pct  = Math.round((Math.abs(score) / maxScore) * 100);
    const cls  = score >= 0 ? 'pos' : 'neg';
    html += `<li class="neighbor-item" data-word="${_esc(word)}">
      <span class="neighbor-word">${_esc(word)}</span>
      <span class="neighbor-score-wrap">
        <span class="neighbor-bar ${cls}" style="width:${pct}%"></span>
        <span class="neighbor-score">${score.toFixed(3)}</span>
      </span>
    </li>`;
  }
  html += '</ol>';
  return html;
}

function _bindNeighborClicks(container) {
  container.querySelectorAll('.neighbor-item').forEach(item => {
    item.addEventListener('click', () => {
      const word = item.dataset.word;
      document.getElementById('query-input').value = word;
      runQuery(word);
    });
  });
}

// ─── Tab: Steps ───────────────────────────────────────────────────────────────

function renderStepsTab(results) {
  const pane = document.getElementById('tab-steps');
  const withSteps = results.filter(r => r.steps.length > 1);

  if (withSteps.length === 0) {
    pane.innerHTML = '<p class="empty-hint">No arithmetic steps — enter an expression like <code>king - man + woman</code>.</p>';
    return;
  }

  pane.innerHTML = '';
  let cardIndex = 0;

  for (const result of withSteps) {
    if (withSteps.length > 1) {
      const heading = document.createElement('p');
      heading.className = 'steps-group-label';
      heading.textContent = result.expr;
      pane.appendChild(heading);
    }

    const timeline = document.createElement('div');
    timeline.className = 'steps-timeline';
    pane.appendChild(timeline);

    result.steps.forEach((step, i) => {
      const card = document.createElement('div');
      card.className = 'step-card';
      const canvasId = `step-canvas-${cardIndex++}`;
      card.innerHTML = `
        <div class="step-header">
          <span class="step-label ${step.sign === -1 ? 'minus' : i === 0 ? 'first' : 'plus'}">${_esc(step.label)}</span>
          ${step.changedDims.length > 0 ? `<span class="step-changed">Top-Δ dims: ${step.changedDims.join(', ')}</span>` : ''}
        </div>
        <canvas id="${canvasId}" class="step-canvas"></canvas>
        <div class="step-sublabel">Running sum after this step</div>
      `;
      timeline.appendChild(card);
      _renderMiniBar(canvasId, step.runningVec, step.changedDims);
    });

    const resCard = document.createElement('div');
    resCard.className = 'step-card result-card';
    const resCanvasId = `step-canvas-${cardIndex++}`;
    resCard.innerHTML = `
      <div class="step-header"><span class="step-label result">= Result (normalized)</span></div>
      <canvas id="${resCanvasId}" class="step-canvas"></canvas>
      <div class="step-sublabel">This vector is used for the nearest-neighbor search</div>
    `;
    timeline.appendChild(resCard);
    _renderMiniBar(resCanvasId, result.steps[result.steps.length - 1].runningVec, []);
  }
}

function _renderMiniBar(canvasId, vec, highlightDims) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  destroyChart(canvasId);

  const dims = vec.length;
  const highlight = new Set(highlightDims);
  const data   = Array.from(vec);
  const colors = data.map((v, i) => {
    if (highlight.has(i)) return v >= 0 ? '#f59e0b' : '#f97316';
    return v >= 0 ? '#3b82f6' : '#ef4444';
  });

  chartInstances[canvasId] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: data.map((_, i) => i),
      datasets: [{ data, backgroundColor: colors, borderWidth: 0 }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { title: ctx => `Dim ${ctx[0].label}`, label: ctx => (+ctx.parsed.y).toFixed(3) }
        },
      },
      scales: {
        x: { display: false, grid: { display: false } },
        y: { display: true, grid: { color: '#e2e8f0' }, ticks: { font: { size: 10 } } },
      },
    },
  });
}

// ─── Tab: Vectors ─────────────────────────────────────────────────────────────

function renderVectorsTab(results) {
  const pane = document.getElementById('tab-vectors');
  pane.innerHTML = '<canvas id="chart-vectors"></canvas>';
  destroyChart('chart-vectors');

  const palette = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
  const entries = [];
  const isMulti = results.length > 1;

  for (const result of results) {
    if (!isMulti) {
      // Single expression: show individual query words too
      for (const { word } of result.parsed.terms) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) entries.push({ label: word, vec: store.getVector(idx) });
      }
      if (result.steps.length > 1) entries.push({ label: '= result', vec: result.normalizedVec });
      else entries.push({ label: result.expr, vec: result.normalizedVec });
      // Top neighbor for reference
      if (result.neighbors.length > 0) {
        const top = result.neighbors[0];
        const idx = store.getWordIndex(top.word);
        if (idx !== -1) entries.push({ label: `~ ${top.word}`, vec: store.getVector(idx) });
      }
    } else {
      // Multi: one entry per expression result
      entries.push({ label: result.expr, vec: result.normalizedVec });
    }
  }

  const dims   = entries[0].vec.length;
  const labels = Array.from({ length: dims }, (_, i) => i);
  const datasets = entries.map(({ label, vec }, ei) => ({
    label,
    data: Array.from(vec),
    backgroundColor: palette[ei % palette.length] + '99',
    borderColor:     palette[ei % palette.length],
    borderWidth: 1,
  }));

  chartInstances['chart-vectors'] = new Chart(
    document.getElementById('chart-vectors'), {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          legend: { display: true, position: 'top' },
          tooltip: {
            callbacks: {
              title: ctx => `Dimension ${ctx[0].label}`,
              label: ctx => `${ctx.dataset.label}: ${(+ctx.parsed.y).toFixed(3)}`,
            },
          },
        },
        scales: {
          x: {
            display: true,
            grid: { display: false },
            ticks: {
              callback: (_, index) => index % 10 === 0 ? index : null,
              maxRotation: 0, font: { size: 10 }, color: '#94a3b8',
            },
          },
          y: {
            display: true,
            grid: { color: '#e2e8f0' },
            ticks: { callback: v => (+v).toFixed(2), font: { size: 10 } },
          },
        },
      },
    }
  );
}

// ─── Tab: Scatter (PCA) ───────────────────────────────────────────────────────

function renderScatterTab(results) {
  const pane = document.getElementById('tab-scatter');
  pane.innerHTML = '<canvas id="chart-scatter"></canvas>';
  destroyChart('chart-scatter');

  const palette = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316'];
  const isMulti = results.length > 1;
  const allEntries = [];

  for (let ri = 0; ri < results.length; ri++) {
    const result = results[ri];
    if (!isMulti) {
      for (const { word } of result.parsed.terms) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) allEntries.push({ label: word, vec: Array.from(store.getVector(idx)), group: 'query' });
      }
      if (result.steps.length > 1)
        allEntries.push({ label: 'result', vec: Array.from(result.normalizedVec), group: 'result' });
      for (const { word } of result.neighbors.slice(0, 10)) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) allEntries.push({ label: word, vec: Array.from(store.getVector(idx)), group: 'neighbor' });
      }
    } else {
      // Per-expression coloring: result vector + top-5 neighbors
      allEntries.push({ label: result.expr, vec: Array.from(result.normalizedVec), group: `e${ri}`, main: true });
      for (const { word } of result.neighbors.slice(0, 5)) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) allEntries.push({ label: word, vec: Array.from(store.getVector(idx)), group: `e${ri}`, main: false });
      }
    }
  }

  if (allEntries.length < 2) {
    pane.innerHTML = '<p class="empty-hint">Not enough points for PCA.</p>';
    return;
  }

  const projected = pca2D(allEntries.map(e => e.vec));

  let datasets;
  if (!isMulti) {
    const roleStyle = {
      query:    { color: '#3b82f6', radius: 8,  border: '#1d4ed8' },
      result:   { color: '#f59e0b', radius: 10, border: '#d97706' },
      neighbor: { color: '#94a3b8', radius: 6,  border: '#64748b' },
    };
    const roleNames = { query: 'Query words', result: 'Result', neighbor: 'Neighbors' };
    const groups = {};
    allEntries.forEach((e, i) => {
      if (!groups[e.group]) groups[e.group] = { points: [], labels: [] };
      groups[e.group].points.push(projected[i]);
      groups[e.group].labels.push(e.label);
    });
    datasets = Object.entries(groups).map(([g, { points, labels }]) => {
      const s = roleStyle[g];
      return { label: roleNames[g], data: points, _labels: labels,
        backgroundColor: s.color + 'cc', borderColor: s.border,
        borderWidth: 2, pointRadius: s.radius, pointHoverRadius: s.radius + 2 };
    });
  } else {
    // One dataset per expression group
    const groups = {};
    allEntries.forEach((e, i) => {
      if (!groups[e.group]) groups[e.group] = { points: [], labels: [], ri: parseInt(e.group.slice(1)) };
      groups[e.group].points.push(projected[i]);
      groups[e.group].labels.push(e.label);
    });
    datasets = Object.entries(groups).map(([g, { points, labels, ri }]) => {
      const col = palette[ri % palette.length];
      return { label: results[ri].expr, data: points, _labels: labels,
        backgroundColor: col + 'bb', borderColor: col,
        borderWidth: 2, pointRadius: 7, pointHoverRadius: 9 };
    });
  }

  chartInstances['chart-scatter'] = new Chart(
    document.getElementById('chart-scatter'), {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 200 },
        plugins: {
          legend: { display: true, position: 'top' },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const lbl = ctx.dataset._labels?.[ctx.dataIndex] ?? '';
                return `${lbl} (${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`;
              },
            },
          },
          datalabels: {
            formatter: (_, ctx) => ctx.dataset._labels?.[ctx.dataIndex] ?? '',
            color: '#1e293b', font: { size: 11, weight: '500' },
            anchor: 'end', align: 'top', offset: 4,
          },
        },
        scales: {
          x: { grid: { color: '#e2e8f0' }, ticks: { display: false }, title: { display: true, text: 'PC 1' } },
          y: { grid: { color: '#e2e8f0' }, ticks: { display: false }, title: { display: true, text: 'PC 2' } },
        },
      },
    }
  );
}

// ─── Tab: Heatmap ─────────────────────────────────────────────────────────────

function renderHeatmapTab(results) {
  const pane = document.getElementById('tab-heatmap');
  pane.innerHTML = '';

  const isMulti = results.length > 1;
  const wordEntries = [];

  for (const result of results) {
    if (!isMulti) {
      for (const { word } of result.parsed.terms) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) wordEntries.push({ label: word, vec: store.getVector(idx), isMain: true });
      }
      if (result.steps.length > 1) wordEntries.push({ label: '= result', vec: result.normalizedVec, isMain: true });
      for (const { word } of result.neighbors.slice(0, 5)) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) wordEntries.push({ label: `~ ${word}`, vec: store.getVector(idx), isMain: false });
      }
    } else {
      wordEntries.push({ label: result.expr, vec: result.normalizedVec, isMain: true });
      for (const { word } of result.neighbors.slice(0, 3)) {
        const idx = store.getWordIndex(word);
        if (idx !== -1) wordEntries.push({ label: `~ ${word}`, vec: store.getVector(idx), isMain: false });
      }
    }
  }

  if (wordEntries.length === 0) return;
  const dims = results[0].normalizedVec.length;

  const CELL_W = 18, CELL_H = 44, PADDING = 8;
  const canvasW = dims * CELL_W + PADDING;
  const canvasH = wordEntries.length * CELL_H + 20; // 20px for axis tick labels

  // ── Outer container: flex row, labels sticky left ──────────────────────────
  const outer = document.createElement('div');
  outer.className = 'heatmap-outer';

  // Left: sticky label column (HTML so it can be position:sticky)
  const labelCol = document.createElement('div');
  labelCol.className = 'heatmap-labels';
  for (const { label, isMain } of wordEntries) {
    const el = document.createElement('div');
    el.className = 'heatmap-row-label' + (isMain ? ' heatmap-row-label--main' : '');
    el.textContent = label;
    el.style.height = CELL_H + 'px';
    labelCol.appendChild(el);
  }
  // Spacer for tick row
  const spacer = document.createElement('div');
  spacer.style.height = '20px';
  labelCol.appendChild(spacer);

  // Right: scrollable canvas
  const scrollArea = document.createElement('div');
  scrollArea.className = 'heatmap-scroll';

  const canvas = document.createElement('canvas');
  canvas.width  = canvasW;
  canvas.height = canvasH;
  scrollArea.appendChild(canvas);

  outer.appendChild(labelCol);
  outer.appendChild(scrollArea);
  pane.appendChild(outer);

  const info = document.createElement('p');
  info.className = 'heatmap-info';
  info.textContent = `${dims} dimensions × ${wordEntries.length} words — Blue = positive, Red = negative`;
  pane.appendChild(info);

  // ── Draw canvas ────────────────────────────────────────────────────────────
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvasW, canvasH);

  let maxAbs = 0;
  for (const { vec } of wordEntries) for (const v of vec) maxAbs = Math.max(maxAbs, Math.abs(v));
  if (maxAbs === 0) maxAbs = 1;

  for (let wi = 0; wi < wordEntries.length; wi++) {
    const { vec, isMain } = wordEntries[wi];
    const y = wi * CELL_H;

    // Highlight main rows with a subtle background
    if (isMain) {
      ctx.fillStyle = '#f0f9ff';
      ctx.fillRect(0, y, canvasW, CELL_H);
    }

    for (let d = 0; d < dims; d++) {
      const val = vec[d] / maxAbs;
      ctx.fillStyle = _divergingColor(val);
      ctx.fillRect(d * CELL_W, y + 2, CELL_W - 1, CELL_H - 4);
    }
  }

  // Axis tick marks every 10 dims
  ctx.fillStyle = '#94a3b8';
  ctx.font = '9px system-ui, sans-serif';
  ctx.textAlign = 'center';
  const tickY = wordEntries.length * CELL_H + 4;
  for (let d = 0; d < dims; d += 10) {
    ctx.fillText(d, d * CELL_W + CELL_W / 2, tickY + 8);
  }
}

/** Map value in [-1,1] to a diverging blue–white–red color. */
function _divergingColor(t) {
  // t = -1 → full red; t = 0 → white; t = 1 → full blue
  const c = Math.round(Math.abs(t) * 200);
  if (t >= 0) return `rgb(${255 - c}, ${255 - c}, 255)`;       // white → blue
  else        return `rgb(255, ${255 - c}, ${255 - c})`;        // white → red
}

// ─── Chart helpers ────────────────────────────────────────────────────────────

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function _esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
