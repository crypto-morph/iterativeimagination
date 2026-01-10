// Load and display iterations for a run

const projectName = window.location.pathname.split('/')[2];
const runId = window.location.pathname.split('/')[4];
let allIterations = [];
let currentSort = 'timeline';
let timelineOrder = [];
let selectedIterationNumber = null;
let iterationByNumber = {};
let lastFingerprint = '';
let autoRefreshTimer = null;

function computeFingerprint(iterations) {
    if (!Array.isArray(iterations) || iterations.length === 0) return 'empty';
    const nums = iterations
        .map(it => it?.iteration_number)
        .filter(n => typeof n === 'number')
        .sort((a, b) => a - b);
    const maxN = nums.length ? nums[nums.length - 1] : 0;
    const maxTs = Math.max(...iterations.map(it => Number(it?.timestamp || 0)));
    return `${iterations.length}:${maxN}:${maxTs}`;
}

function setAutoRefreshStatus(text) {
    const el = document.getElementById('autoRefreshStatus');
    if (el) el.textContent = text || '';
}

function applyIterations(dataIterations, { preferLatestIfSelectedLatest = true } = {}) {
    const prevLatest = timelineOrder.length ? timelineOrder[timelineOrder.length - 1] : null;
    const wasSelectedLatest = (selectedIterationNumber !== null && prevLatest !== null && selectedIterationNumber === prevLatest);

    allIterations = dataIterations;
    iterationByNumber = {};
    allIterations.forEach(it => {
        if (it && typeof it.iteration_number === 'number') {
            iterationByNumber[it.iteration_number] = it;
        }
    });

    timelineOrder = [...allIterations]
        .map(it => it.iteration_number)
        .filter(n => typeof n === 'number')
        .sort((a, b) => a - b);

    buildIterationButtons();

    if (timelineOrder.length > 0) {
        const latest = timelineOrder[timelineOrder.length - 1];
        if (selectedIterationNumber === null) {
            selectIteration(latest);
        } else if (preferLatestIfSelectedLatest && wasSelectedLatest) {
            selectIteration(latest);
        } else if (iterationByNumber[selectedIterationNumber]) {
            selectIteration(selectedIterationNumber);
        } else {
            selectIteration(latest);
        }
    }

    sortAndDisplayIterations();
}

async function loadIterations() {
    try {
        const response = await fetch(`/api/project/${projectName}/run/${runId}/iterations`);
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('loading').textContent = 'Error: ' + data.error;
            return;
        }
        
        applyIterations(data.iterations || []);
        lastFingerprint = computeFingerprint(allIterations);
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('content').classList.remove('hidden');
    } catch (error) {
        document.getElementById('loading').textContent = 'Error loading iterations: ' + error.message;
    }
}

function buildIterationButtons() {
    const container = document.getElementById('iterationButtons');
    if (!container) return;
    container.innerHTML = '';

    timelineOrder.forEach(n => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'iter-btn';
        btn.textContent = String(n);
        btn.dataset.iteration = String(n);
        btn.addEventListener('click', () => selectIteration(n));
        container.appendChild(btn);
    });

    updateNavButtons();
}

function updateNavButtons() {
    const prevBtn = document.getElementById('prevIteration');
    const nextBtn = document.getElementById('nextIteration');
    if (!prevBtn || !nextBtn) return;

    if (selectedIterationNumber === null || timelineOrder.length === 0) {
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }
    const idx = timelineOrder.indexOf(selectedIterationNumber);
    prevBtn.disabled = idx <= 0;
    nextBtn.disabled = idx < 0 || idx >= timelineOrder.length - 1;
}

function highlightSelectedButton() {
    const container = document.getElementById('iterationButtons');
    if (!container) return;
    container.querySelectorAll('.iter-btn').forEach(btn => {
        const n = Number(btn.dataset.iteration);
        if (n === selectedIterationNumber) btn.classList.add('active');
        else btn.classList.remove('active');
    });
    // Try to keep selection visible
    const active = container.querySelector('.iter-btn.active');
    if (active && typeof active.scrollIntoView === 'function') {
        active.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
}

function selectIteration(n) {
    if (typeof n !== 'number') return;
    const iter = iterationByNumber[n];
    if (!iter) return;

    selectedIterationNumber = n;

    const label = document.getElementById('selectedIterationLabel');
    if (label) label.textContent = String(n);

    const img = document.getElementById('selectedImg');
    if (img) {
        img.src = `/api/project/${projectName}/run/${runId}/image/iteration_${n}.png`;
    }

    const meta = document.getElementById('selectedMeta');
    if (meta) {
        const score = iter.evaluation?.overall_score ?? 'N/A';
        const sim = iter.comparison?.similarity_score;
        const simPct = (typeof sim === 'number') ? `${(sim * 100).toFixed(1)}%` : 'N/A';
        const denoise = iter.parameters_used?.denoise;
        const denoiseText = (typeof denoise === 'number') ? denoise.toFixed(2) : 'N/A';
        const cfg = iter.parameters_used?.cfg ?? 'N/A';
        const steps = iter.parameters_used?.steps ?? 'N/A';
        meta.innerHTML = `<div><strong>Score:</strong> ${escapeHtml(String(score))}% &nbsp; <strong>Similarity:</strong> ${escapeHtml(simPct)}</div>` +
                         `<div><strong>Denoise:</strong> ${escapeHtml(denoiseText)} &nbsp; <strong>CFG:</strong> ${escapeHtml(String(cfg))} &nbsp; <strong>Steps:</strong> ${escapeHtml(String(steps))}</div>`;
    }

    highlightSelectedButton();
    updateNavButtons();
}

function sortAndDisplayIterations() {
    let sorted = [...allIterations];
    
    if (currentSort === 'score') {
        // Sort by score (descending - highest first)
        sorted.sort((a, b) => {
            const scoreA = a.evaluation?.overall_score || 0;
            const scoreB = b.evaluation?.overall_score || 0;
            return scoreB - scoreA;
        });
    } else {
        // Sort by timeline (iteration number - ascending)
        sorted.sort((a, b) => {
            return (a.iteration_number || 0) - (b.iteration_number || 0);
        });
    }
    
    displayIterations(sorted);
}

function displayIterations(iterations) {
    const container = document.getElementById('iterations');
    container.innerHTML = '';
    
    iterations.forEach(iter => {
        const card = createIterationCard(iter);
        container.appendChild(card);
    });
}

function createIterationCard(iter) {
    const card = document.createElement('div');
    card.className = 'iteration-card';
    
    const score = iter.evaluation?.overall_score || 0;
    const scoreClass = score >= 90 ? 'excellent' : score >= 70 ? 'good' : 'poor';
    
    const similarity = iter.comparison?.similarity_score || 0;
    const timestamp = new Date(iter.timestamp * 1000).toLocaleString();
    
    card.innerHTML = `
        <div class="iteration-header">
            <span class="iteration-number">Iteration ${iter.iteration_number}</span>
            <span class="score-badge ${scoreClass}">${score}%</span>
        </div>
        <img src="/api/project/${projectName}/run/${runId}/image/iteration_${iter.iteration_number}.png" 
             alt="Iteration ${iter.iteration_number}" 
             class="iteration-image"
             onclick="this.classList.toggle('fullscreen')">
        <div class="metadata-panel" id="metadata-${iter.iteration_number}">
            <div class="timestamp">${timestamp}</div>
            
            <div class="metadata-section">
                <h4>📊 Scores</h4>
                <div class="metadata-item">
                    <span class="metadata-label">Overall Score:</span>
                    <span class="metadata-value"><strong>${score}%</strong></span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Similarity:</span>
                    <span class="metadata-value">${(similarity * 100).toFixed(1)}%</span>
                </div>
            </div>
            
            <div class="metadata-section">
                <h4>⚙️ Parameters</h4>
                <div class="metadata-item">
                    <span class="metadata-label">Denoise:</span>
                    <span class="metadata-value">${iter.parameters_used?.denoise?.toFixed(2) || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">CFG:</span>
                    <span class="metadata-value">${iter.parameters_used?.cfg || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Steps:</span>
                    <span class="metadata-value">${iter.parameters_used?.steps || 'N/A'}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Seed:</span>
                    <span class="metadata-value">${iter.parameters_used?.seed || 'N/A'}</span>
                </div>
            </div>
            
            <div class="metadata-section">
                <h4>✅ Acceptance Criteria</h4>
                <ul class="criteria-list">
                    ${Object.entries(iter.evaluation?.criteria_results || {}).map(([field, result]) => {
                        const passed = result === true || (typeof result === 'number' && result > 0);
                        return `<li class="criteria-item ${passed ? 'pass' : 'fail'}">
                            <span>${field.replace(/_/g, ' ')}</span>
                            <span>${passed ? '✓' : '✗'}</span>
                        </li>`;
                    }).join('')}
                </ul>
            </div>
            
            ${iter.prompts_used?.positive ? `
            <div class="metadata-section">
                <h4>✨ Positive Prompt</h4>
                <div class="prompt-box">${escapeHtml(iter.prompts_used.positive)}</div>
            </div>
            ` : ''}
            
            ${iter.prompts_used?.negative ? `
            <div class="metadata-section">
                <h4>🚫 Negative Prompt</h4>
                <div class="prompt-box">${escapeHtml(iter.prompts_used.negative)}</div>
            </div>
            ` : ''}
            
            ${iter.comparison?.differences?.length ? `
            <div class="metadata-section">
                <h4>🔍 Differences</h4>
                <ul class="criteria-list">
                    ${iter.comparison.differences.map(diff => `<li style="padding: 4px 0; color: #666;">• ${escapeHtml(diff)}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
        </div>
    `;
    
    // Toggle metadata visibility
    const showMetadata = document.getElementById('showMetadata');
    if (showMetadata) {
        const metadataPanel = card.querySelector('.metadata-panel');
        showMetadata.addEventListener('change', (e) => {
            metadataPanel.style.display = e.target.checked ? 'block' : 'none';
        });
    }
    
    return card;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle original image visibility
document.getElementById('showOriginal')?.addEventListener('change', (e) => {
    document.getElementById('originalImage').style.display = e.target.checked ? 'block' : 'none';
});

// Prev/Next navigation (timeline order)
document.getElementById('prevIteration')?.addEventListener('click', () => {
    if (selectedIterationNumber === null) return;
    const idx = timelineOrder.indexOf(selectedIterationNumber);
    if (idx > 0) selectIteration(timelineOrder[idx - 1]);
});

document.getElementById('nextIteration')?.addEventListener('click', () => {
    if (selectedIterationNumber === null) return;
    const idx = timelineOrder.indexOf(selectedIterationNumber);
    if (idx >= 0 && idx < timelineOrder.length - 1) selectIteration(timelineOrder[idx + 1]);
});

// Sort button handlers
document.getElementById('sortTimeline')?.addEventListener('click', () => {
    currentSort = 'timeline';
    document.getElementById('sortTimeline').classList.add('active');
    document.getElementById('sortScore').classList.remove('active');
    sortAndDisplayIterations();
});

document.getElementById('sortScore')?.addEventListener('click', () => {
    currentSort = 'score';
    document.getElementById('sortScore').classList.add('active');
    document.getElementById('sortTimeline').classList.remove('active');
    sortAndDisplayIterations();
});

// Auto refresh toggle + poll loop
const autoRefresh = document.getElementById('autoRefresh');
if (autoRefresh) {
    autoRefresh.addEventListener('change', () => {
        setAutoRefreshStatus(autoRefresh.checked ? ' (on)' : ' (off)');
    });
    setAutoRefreshStatus(autoRefresh.checked ? ' (on)' : ' (off)');
}

async function pollForUpdates() {
    try {
        if (document.visibilityState !== 'visible') return;
        if (autoRefresh && !autoRefresh.checked) return;
        const response = await fetch(`/api/project/${projectName}/run/${runId}/iterations`, { cache: 'no-store' });
        const data = await response.json();
        if (data.error) return;
        const next = data.iterations || [];
        const fp = computeFingerprint(next);
        if (fp !== lastFingerprint) {
            lastFingerprint = fp;
            applyIterations(next, { preferLatestIfSelectedLatest: true });
            setAutoRefreshStatus(' (updated)');
            setTimeout(() => setAutoRefreshStatus(autoRefresh && autoRefresh.checked ? ' (on)' : ' (off)'), 800);
        }
    } catch (e) {
        // ignore transient failures
    }
}

if (autoRefreshTimer) clearInterval(autoRefreshTimer);
autoRefreshTimer = setInterval(pollForUpdates, 2500);

// Load on page load
loadIterations();
