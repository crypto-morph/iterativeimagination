/* global window, document, fetch */

const project = window.__LIVE_PROJECT__;

// State
let runId = null;
let pollTimer = null;
let currentMask = "";
let currentIteration = null;
let currentSettings = null;
let currentTerms = { must_include: [], ban_terms: [], avoid_terms: [] };
let currentSuggestion = null;

// Utility functions
function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  const el = $("statusText");
  if (el) el.textContent = text || "Ready";
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Diff highlighting (GitHub-style)
function highlightDiff(oldText, newText) {
  const oldWords = oldText.split(/(\s+)/);
  const newWords = newText.split(/(\s+)/);
  const result = [];
  let oldIdx = 0;
  let newIdx = 0;
  
  while (oldIdx < oldWords.length || newIdx < newWords.length) {
    if (oldIdx >= oldWords.length) {
      // Only new words left
      result.push(`<span class="diff-added">${escapeHtml(newWords[newIdx])}</span>`);
      newIdx++;
    } else if (newIdx >= newWords.length) {
      // Only old words left
      result.push(`<span class="diff-removed">${escapeHtml(oldWords[oldIdx])}</span>`);
      oldIdx++;
    } else if (oldWords[oldIdx] === newWords[newIdx]) {
      // Same word
      result.push(`<span class="diff-unchanged">${escapeHtml(oldWords[oldIdx])}</span>`);
      oldIdx++;
      newIdx++;
    } else {
      // Different - try to find next match
      const oldWord = oldWords[oldIdx].trim();
      const newWord = newWords[newIdx].trim();
      
      if (oldWord && !newWord) {
        result.push(`<span class="diff-removed">${escapeHtml(oldWords[oldIdx])}</span>`);
        oldIdx++;
      } else if (!oldWord && newWord) {
        result.push(`<span class="diff-added">${escapeHtml(newWords[newIdx])}</span>`);
        newIdx++;
      } else {
        // Both are non-empty, different words
        result.push(`<span class="diff-removed">${escapeHtml(oldWords[oldIdx])}</span>`);
        result.push(`<span class="diff-added">${escapeHtml(newWords[newIdx])}</span>`);
        oldIdx++;
        newIdx++;
      }
    }
  }
  
  return result.join("");
}

// Load masks dropdown
async function loadMasks() {
  try {
    const res = await fetch(`/api/project/${project}/input/masks`);
    const data = await res.json();
    const masks = (data && data.masks) || [];
    
    const select = $("maskSelect");
    if (!select) return;
    
    select.innerHTML = "";
    
    // Add "global" option
    const globalOpt = document.createElement("option");
    globalOpt.value = "global";
    globalOpt.textContent = "Global (no mask)";
    select.appendChild(globalOpt);
    
    // Add mask options
    for (const mask of masks) {
      const opt = document.createElement("option");
      opt.value = mask.name;
      opt.textContent = mask.name;
      select.appendChild(opt);
    }
    
    // Set default to first mask or global
    if (masks.length > 0) {
      select.value = masks[0].name;
      currentMask = masks[0].name;
    } else {
      select.value = "global";
      currentMask = "global";
    }
    
    // Load data for selected mask (but wait a bit for DOM to be ready)
    setTimeout(() => {
      loadMaskData();
    }, 100);
  } catch (e) {
    console.error("Failed to load masks:", e);
  }
}

// Load mask-specific data (prompts, terms)
async function loadMaskData() {
  if (!currentMask) {
    console.log("No mask selected, skipping loadMaskData");
    return;
  }
  
  console.log(`Loading data for mask: ${currentMask}`);
  
  // Load prompts
  try {
    const res = await fetch(`/api/project/${project}/mask/${currentMask}/prompts`);
    const data = await res.json();
    if (res.ok) {
      const posEl = $("inputPositivePrompt");
      const negEl = $("inputNegativePrompt");
      if (posEl) posEl.value = data.positive || "";
      if (negEl) negEl.value = data.negative || "";
      console.log(`Loaded prompts for ${currentMask}:`, data);
    } else {
      console.error("Failed to load prompts:", data.error);
    }
  } catch (e) {
    console.error("Failed to load mask prompts:", e);
  }
  
  // Load terms
  try {
    const res = await fetch(`/api/project/${project}/mask/${currentMask}/terms`);
    const data = await res.json();
    if (res.ok) {
      currentTerms = {
        must_include: data.must_include || [],
        ban_terms: data.ban_terms || [],
        avoid_terms: data.avoid_terms || []
      };
      console.log(`Loaded terms for ${currentMask}:`, currentTerms);
      renderTerms();
    } else {
      console.error("Failed to load terms:", data.error);
    }
  } catch (e) {
    console.error("Failed to load mask terms:", e);
  }
}

// Render terms
function renderTerms() {
  renderTermList("must_include", currentTerms.must_include);
  renderTermList("ban_terms", currentTerms.ban_terms);
  renderTermList("avoid_terms", currentTerms.avoid_terms);
}

function renderTermList(type, terms) {
  const container = $(type === "must_include" ? "mustIncludeTerms" : type === "ban_terms" ? "banTerms" : "avoidTerms");
  if (!container) return;
  
  container.innerHTML = "";
  
  if (terms.length === 0) {
    container.innerHTML = '<span class="has-text-grey is-size-7">No terms defined</span>';
    return;
  }
  
  terms.forEach((term, idx) => {
    const tag = document.createElement("span");
    tag.className = "tag term-tag";
    tag.innerHTML = `
      <span>${escapeHtml(term)}</span>
      <button class="delete is-small" data-term-type="${type}" data-term-idx="${idx}"></button>
    `;
    container.appendChild(tag);
  });
  
  // Wire delete buttons
  container.querySelectorAll(".delete").forEach(btn => {
    btn.addEventListener("click", () => {
      const termType = btn.dataset.termType;
      const termIdx = parseInt(btn.dataset.termIdx);
      if (termType === "must_include") {
        currentTerms.must_include.splice(termIdx, 1);
      } else if (termType === "ban_terms") {
        currentTerms.ban_terms.splice(termIdx, 1);
      } else if (termType === "avoid_terms") {
        currentTerms.avoid_terms.splice(termIdx, 1);
      }
      renderTerms();
    });
  });
}

// Load settings
async function loadSettings() {
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/settings`);
    const data = await res.json();
    if (res.ok) {
      currentSettings = data;
      renderSettings();
    }
  } catch (e) {
    console.error("Failed to load settings:", e);
  }
}

function renderSettings() {
  const container = $("settingsForm");
  if (!container || !currentSettings) return;
  
  // Common sampler options
  const samplerOptions = [
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m",
    "euler",
    "euler_ancestral",
    "heun",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_sde",
    "dpmpp_2m_sde",
    "ddim",
    "uni_pc"
  ].map(s => `<option value="${s}" ${s === currentSettings.sampler_name ? "selected" : ""}>${s}</option>`).join("");
  
  // Common scheduler options
  const schedulerOptions = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform"
  ].map(s => `<option value="${s}" ${s === currentSettings.scheduler ? "selected" : ""}>${s}</option>`).join("");
  
  container.innerHTML = `
    <div class="field">
      <label class="label is-size-7">Denoise</label>
      <div class="control">
        <input id="settingDenoise" class="input is-small" type="number" step="0.01" min="0" max="1" value="${currentSettings.denoise}">
      </div>
    </div>
    <div class="field">
      <label class="label is-size-7">CFG</label>
      <div class="control">
        <input id="settingCfg" class="input is-small" type="number" step="0.1" min="1" max="30" value="${currentSettings.cfg}">
      </div>
    </div>
    <div class="field">
      <label class="label is-size-7">Steps</label>
      <div class="control">
        <input id="settingSteps" class="input is-small" type="number" min="1" max="100" value="${currentSettings.steps}">
      </div>
    </div>
    <div class="field">
      <label class="label is-size-7">Sampler</label>
      <div class="control">
        <div class="select is-small is-fullwidth">
          <select id="settingSampler" class="is-small">
            ${samplerOptions}
          </select>
        </div>
      </div>
    </div>
    <div class="field">
      <label class="label is-size-7">Scheduler</label>
      <div class="control">
        <div class="select is-small is-fullwidth">
          <select id="settingScheduler" class="is-small">
            ${schedulerOptions}
          </select>
        </div>
      </div>
    </div>
    <div class="field">
      <label class="label is-size-7">Checkpoint</label>
      <div class="control">
        <input id="settingCheckpoint" class="input is-small" type="text" value="${escapeHtml(currentSettings.checkpoint)}" readonly>
      </div>
      <p class="help is-size-7">Read-only (edit in config/AIGen.yaml)</p>
    </div>
    <div class="field">
      <label class="label is-size-7">Workflow</label>
      <div class="control">
        <input id="settingWorkflow" class="input is-small" type="text" value="${escapeHtml(currentSettings.workflow)}" readonly>
      </div>
      <p class="help is-size-7">Read-only (edit in config/AIGen.yaml)</p>
    </div>
  `;
}

async function saveSettings() {
  if (!currentSettings) {
    $("settingsStatus").textContent = "No settings loaded";
    return;
  }
  
  const statusEl = $("settingsStatus");
  statusEl.textContent = "Saving...";
  
  // Get values from form
  const denoise = parseFloat($("settingDenoise").value);
  const cfg = parseFloat($("settingCfg").value);
  const steps = parseInt($("settingSteps").value);
  const sampler_name = $("settingSampler").value;
  const scheduler = $("settingScheduler").value;
  
  // Validate
  if (isNaN(denoise) || denoise < 0 || denoise > 1) {
    statusEl.textContent = "Invalid denoise value (0-1)";
    return;
  }
  if (isNaN(cfg) || cfg < 1 || cfg > 30) {
    statusEl.textContent = "Invalid CFG value (1-30)";
    return;
  }
  if (isNaN(steps) || steps < 1 || steps > 100) {
    statusEl.textContent = "Invalid steps value (1-100)";
    return;
  }
  
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        denoise,
        cfg,
        steps,
        sampler_name,
        scheduler
      })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.error || "Failed to save";
      return;
    }
    
    // Update current settings
    currentSettings.denoise = denoise;
    currentSettings.cfg = cfg;
    currentSettings.steps = steps;
    currentSettings.sampler_name = sampler_name;
    currentSettings.scheduler = scheduler;
    
    statusEl.textContent = "Saved successfully";
    setTimeout(() => { statusEl.textContent = ""; }, 2000);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
}

// Load iteration prompts
async function loadIterationPrompts() {
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, { cache: "no-store" });
    const data = await res.json();
    if (res.ok) {
      $("iterationPositivePrompt").value = data.positive || "";
      $("iterationNegativePrompt").value = data.negative || "";
    }
  } catch (e) {
    console.error("Failed to load iteration prompts:", e);
  }
}

// Save iteration prompts
async function saveIterationPrompts() {
  const positive = $("iterationPositivePrompt").value || "";
  const negative = $("iterationNegativePrompt").value || "";
  
  const statusEl = $("promptsStatus");
  statusEl.textContent = "Saving...";
  
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ positive, negative })
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.error || "Failed to save";
      return;
    }
    statusEl.textContent = "Saved successfully";
    setTimeout(() => { statusEl.textContent = ""; }, 2000);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
}

// Describe images
async function describeInput() {
  const container = $("inputDescription");
  container.innerHTML = '<span class="has-text-grey">Generating description...</span>';
  
  try {
    const res = await fetch(`/api/project/${project}/input/describe`);
    const data = await res.json();
    if (res.ok) {
      container.textContent = data.description || "No description available";
    } else {
      container.innerHTML = `<span class="has-text-danger">${escapeHtml(data.error || "Failed")}</span>`;
    }
  } catch (e) {
    container.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
  }
}

async function describeLatest() {
  if (!runId || currentIteration === null) return;
  
  const container = $("latestDescription");
  container.innerHTML = '<span class="has-text-grey">Generating description...</span>';
  
  try {
    const res = await fetch(`/api/project/${project}/run/${runId}/describe/${currentIteration}`);
    const data = await res.json();
    if (res.ok) {
      container.textContent = data.description || "No description available";
    } else {
      container.innerHTML = `<span class="has-text-danger">${escapeHtml(data.error || "Failed")}</span>`;
    }
  } catch (e) {
    container.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
  }
}

// Generate prompt suggestion
async function generateSuggestion() {
  if (!runId) {
    setStatus("No run active. Click 'Run' first.");
    return;
  }
  
  const btn = $("generateSuggestionBtn");
  const box = $("suggestionBox");
  const placeholder = $("suggestionPlaceholder");
  
  btn.disabled = true;
  btn.innerHTML = '<span class="icon"><i class="fas fa-spinner fa-spin"></i></span><span>Generating...</span>';
  box.classList.add("is-hidden");
  placeholder.textContent = "Generating AI suggestions...";
  
  try {
    const res = await fetch(`/api/project/${project}/run/${runId}/suggest_prompts`);
    const data = await res.json();
    if (!res.ok) {
      placeholder.innerHTML = `<span class="has-text-danger">${escapeHtml(data.error || "Failed")}</span>`;
      return;
    }
    
    currentSuggestion = data;
    
    // Render with diff highlighting
    const posDiff = highlightDiff(data.current_positive || "", data.suggested_positive || "");
    const negDiff = highlightDiff(data.current_negative || "", data.suggested_negative || "");
    
    $("suggestedPositivePrompt").innerHTML = posDiff;
    $("suggestedNegativePrompt").innerHTML = negDiff;
    
    box.classList.remove("is-hidden");
    placeholder.classList.add("is-hidden");
    setStatus("Suggestion generated");
  } catch (e) {
    placeholder.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="icon"><i class="fas fa-magic"></i></span><span>Generate</span>';
  }
}

// Apply suggestion
function applySuggestion() {
  if (!currentSuggestion) return;
  
  // Get plain text from contenteditable divs (strip HTML)
  const posEl = $("suggestedPositivePrompt");
  const negEl = $("suggestedNegativePrompt");
  
  const positive = posEl.textContent || posEl.innerText || "";
  const negative = negEl.textContent || negEl.innerText || "";
  
  $("iterationPositivePrompt").value = positive.trim();
  $("iterationNegativePrompt").value = negative.trim();
  
  setStatus("Suggestion applied to iteration prompts");
  $("promptsStatus").textContent = "Applied from suggestion (not saved yet)";
}

// Reset run
async function resetRun() {
  if (!confirm("Reset the current run? This will clear the checkpoint.")) return;
  
  try {
    const res = await fetch(`/api/project/${project}/live/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: true, max_iterations: parseInt($("maxIterInput").value || "20", 10) })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.error || "Failed to reset");
      return;
    }
    
    runId = data.run_id;
    setStatus(`Run reset: ${runId}`);
    startPolling();
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
}

// Run iteration
async function runIteration() {
  if (!currentMask) {
    setStatus("Please select a mask first");
    return;
  }
  
  // Save prompts first
  await saveIterationPrompts();
  
  // Start/continue run
  const maxIter = parseInt($("maxIterInput").value || "20", 10);
  
  $("runBtn").disabled = true;
  setStatus("Starting run...");
  
  try {
    const res = await fetch(`/api/project/${project}/live/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: false, max_iterations: maxIter })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.error || "Failed to start");
      $("runBtn").disabled = false;
      return;
    }
    
    runId = data.run_id;
    setStatus(`Run started: ${runId}`);
    startPolling();
    
    // Auto-describe images after a delay
    setTimeout(() => {
      describeInput();
      pollOnce().then(() => {
        if (currentIteration) {
          describeLatest();
        }
      });
    }, 2000);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    $("runBtn").disabled = false;
  }
}

// Polling
async function fetchState() {
  if (!runId) return null;
  const res = await fetch(`/api/project/${project}/live/${runId}/state`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Failed to fetch state");
  return data.state;
}

async function fetchIterations() {
  if (!runId) return [];
  const res = await fetch(`/api/project/${project}/run/${runId}/iterations`);
  const data = await res.json();
  if (!res.ok) return [];
  return data.iterations || [];
}

function renderLatest(state, iterations) {
  const latest = iterations.length ? iterations[iterations.length - 1] : null;
  const spinner = $("spinnerOverlay");
  
  if (latest) {
    const n = latest.iteration_number;
    currentIteration = n;
    const latestImg = $("latestImg");
    if (latestImg) {
      latestImg.src = `/api/project/${project}/run/${runId}/image/iteration_${n}.png`;
    }
    
    // Auto-describe if not already described
    const descEl = $("latestDescription");
    if (descEl && descEl.textContent.includes("No iteration")) {
      describeLatest();
    }
  } else {
    currentIteration = null;
  }
  
  if (state.status === "running" || state.status === "starting") {
    if (spinner) spinner.classList.remove("is-hidden");
    $("runBtn").disabled = true;
  } else {
    if (spinner) spinner.classList.add("is-hidden");
    $("runBtn").disabled = false;
  }
  
  setStatus(`${state.status} | iter ${state.current_iteration} | best ${state.best_score} (iter ${state.best_iteration || "-"})`);
}

async function pollOnce() {
  try {
    const state = await fetchState();
    const iterations = await fetchIterations();
    if (!state) return;
    renderLatest(state, iterations);
  } catch (e) {
    setStatus(`poll error: ${e.message}`);
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  pollTimer = setInterval(pollOnce, 2000);
}

// Tab switching
function initTabs() {
  // Prompt tabs
  const promptTabs = document.querySelectorAll('.tabs.is-boxed [data-tab]');
  promptTabs.forEach(tab => {
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      const tabName = tab.dataset.tab;
      
      // Update tab buttons
      tab.parentElement.parentElement.querySelectorAll("li").forEach(li => li.classList.remove("is-active"));
      tab.parentElement.classList.add("is-active");
      
      // Update content
      document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.add("is-hidden");
      });
      const contentEl = $(tabName);
      if (contentEl) {
        contentEl.classList.remove("is-hidden");
      }
    });
  });
  
  // Term tabs
  const termTabs = document.querySelectorAll('.tabs.is-small [data-term-tab]');
  termTabs.forEach(tab => {
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      const tabName = tab.dataset.termTab;
      
      // Update tab buttons
      tab.parentElement.parentElement.querySelectorAll("li").forEach(li => li.classList.remove("is-active"));
      tab.parentElement.classList.add("is-active");
      
      // Update content
      document.querySelectorAll(".term-tab-content").forEach(content => {
        content.classList.add("is-hidden");
      });
      const contentEl = $(tabName);
      if (contentEl) {
        contentEl.classList.remove("is-hidden");
      }
    });
  });
}

// Wire UI
function wireUI() {
  $("resetBtn").addEventListener("click", resetRun);
  $("runBtn").addEventListener("click", runIteration);
  $("maskSelect").addEventListener("change", (e) => {
    currentMask = e.target.value;
    loadMaskData();
  });
  $("savePromptsBtn").addEventListener("click", saveIterationPrompts);
  $("saveSettingsBtn").addEventListener("click", saveSettings);
  $("generateSuggestionBtn").addEventListener("click", generateSuggestion);
  $("applySuggestionBtn").addEventListener("click", applySuggestion);
  
  initTabs();
}

// Initialize
async function init() {
  console.log("Initializing live page...");
  
  // Wire UI first so event handlers are ready
  wireUI();
  
  // Wait a moment for DOM to be fully ready
  await new Promise(resolve => setTimeout(resolve, 100));
  
  // Load masks (this will set currentMask and call loadMaskData)
  await loadMasks();
  
  // Load other data
  await loadSettings();
  await loadIterationPrompts();
  await describeInput();
  
  console.log("Initialization complete");
}

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
