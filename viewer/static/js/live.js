/* global window, document, fetch, PromptsModule, ValidationModule, SuggestionsModule */

const project = window.__LIVE_PROJECT__;

// State
let runId = null;
let pollTimer = null;
let currentMask = "";
let currentIteration = null;
let currentSettings = null;

// Utility functions
function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  const el = $("statusText");
  if (el) el.textContent = text || "Ready";
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
    
    // Load data for selected mask
    await loadMaskData();
  } catch (e) {
    console.error("Failed to load masks:", e);
  }
}

// Load mask-specific data (prompts, terms)
async function loadMaskData() {
  if (!currentMask) {
    currentMask = "global";
  }
  
  console.log(`Loading data for mask: ${currentMask}`);
  
  // Load prompts as terms
  try {
    const res = await fetch(`/api/project/${project}/mask/${currentMask}/prompts`);
    const data = await res.json();
    if (res.ok) {
      // Parse prompts into terms and set them
      PromptsModule.setTermsFromPrompts(data.positive || "", data.negative || "");
    }
  } catch (e) {
    console.error("Failed to load mask prompts:", e);
  }
  
  // Load mask terms
  try {
    const res = await fetch(`/api/project/${project}/mask/${currentMask}/terms`);
    const data = await res.json();
    if (res.ok) {
      PromptsModule.setMaskTerms(
        data.must_include || [],
        data.ban_terms || [],
        data.avoid_terms || []
      );
    }
  } catch (e) {
    console.error("Failed to load mask terms:", e);
  }
  
  // Run validation
  runValidation();
}

// Run validation
function runValidation() {
  const positiveTerms = PromptsModule.getPositiveTerms();
  const negativeTerms = PromptsModule.getNegativeTerms();
  const maskTerms = PromptsModule.getMaskTerms();
  
  const issues = ValidationModule.ValidationRules.validateAll(
    positiveTerms,
    negativeTerms,
    maskTerms
  );
  
  const container = $("validationIssues");
  if (container) {
    if (issues.length > 0) {
      container.style.display = "block";
      ValidationModule.displayValidationIssues(container, issues);
    } else {
      container.style.display = "none";
    }
  }
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

// Describe images as terms (but don't overwrite existing prompts)
async function describeInputTerms(overwrite = false) {
  const container = $("inputDescription");
  const btn = $("loadFromImageBtn");
  if (!container) return;
  
  if (btn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
  }
  container.innerHTML = '<span class="has-text-grey"><i class="fas fa-spinner fa-spin"></i> Generating description...</span>';
  
  try {
    const res = await fetch(`/api/project/${project}/input/describe_terms`);
    const data = await res.json();
    if (res.ok) {
      if (overwrite) {
        // Only overwrite if explicitly requested
        PromptsModule.setTermsFromPrompts(
          (data.positive_terms || []).join(", "),
          (data.negative_terms || []).join(", ")
        );
        container.innerHTML = '<span class="has-text-success"><i class="fas fa-check"></i> Terms loaded from image description</span>';
        runValidation();
      } else {
        // Just show the description without overwriting
        const posCount = (data.positive_terms || []).length;
        const negCount = (data.negative_terms || []).length;
        container.innerHTML = `<span class="has-text-grey"><i class="fas fa-info-circle"></i> AIVis found ${posCount} positive and ${negCount} negative terms (click to load)</span>`;
      }
    } else {
      container.innerHTML = `<span class="has-text-danger">${escapeHtml(data.error || "Failed")}</span>`;
    }
  } catch (e) {
    container.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("is-loading");
    }
  }
}

async function describeLatestTerms() {
  if (!runId || currentIteration === null) return;
  
  const container = $("latestDescription");
  if (!container) return;
  
  container.innerHTML = '<span class="has-text-grey">Generating description...</span>';
  
  try {
    const res = await fetch(`/api/project/${project}/run/${runId}/describe_terms/${currentIteration}`);
    const data = await res.json();
    if (res.ok) {
      container.innerHTML = '<span class="has-text-success"><i class="fas fa-check"></i> Terms loaded from iteration description</span>';
      // Optionally update prompts with iteration terms
      // PromptsModule.setTermsFromPrompts(...)
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
    const placeholder = $("suggestionPlaceholder");
    if (placeholder) {
      placeholder.innerHTML = '<span class="has-text-warning">Please start a run first by clicking "Run"</span>';
    }
    return;
  }
  
  const btn = $("generateSuggestionBtn");
  const placeholder = $("suggestionPlaceholder");
  const panel = $("suggestionsPanel");
  
  if (btn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
  }
  
  if (placeholder) {
    placeholder.textContent = "Generating AI suggestions...";
    placeholder.classList.remove("is-hidden");
  }
  
  if (panel) {
    panel.classList.add("is-hidden");
  }
  
  try {
    console.log(`Generating suggestions for runId: ${runId}, mask: ${currentMask || "global"}`);
    const suggestions = await SuggestionsModule.generateSuggestions(runId, currentMask || "global");
    console.log("Suggestions received:", suggestions);
    
    if (suggestions) {
      SuggestionsModule.renderSuggestions(suggestions, applySuggestion);
      if (placeholder) {
        placeholder.classList.add("is-hidden");
      }
    } else {
      if (placeholder) {
        placeholder.innerHTML = '<span class="has-text-warning">No suggestions generated. Make sure a run has completed at least one iteration.</span>';
      }
    }
  } catch (e) {
    console.error("Error generating suggestions:", e);
    if (placeholder) {
      placeholder.innerHTML = `<span class="has-text-danger">Error: ${escapeHtml(e.message)}</span>`;
    }
    setStatus(`Error generating suggestions: ${e.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("is-loading");
    }
  }
}

// Apply a single suggestion
function applySuggestion(type, action, term) {
  // Handle mask terms
  if (type.startsWith("mask_")) {
    const maskType = type.replace("mask_", ""); // "must_include", "ban_terms", or "avoid_terms"
    const maskTerms = PromptsModule.getMaskTerms();
    let terms = [...(maskTerms[maskType] || [])];
    
    if (action === "add") {
      const lower = term.toLowerCase();
      if (!terms.some(t => t.toLowerCase() === lower)) {
        terms.push(term);
      }
    } else if (action === "remove") {
      terms = terms.filter(t => t.toLowerCase() !== term.toLowerCase());
    }
    
    // Update mask terms
    const newMaskTerms = { ...maskTerms };
    newMaskTerms[maskType] = terms;
    PromptsModule.setMaskTerms(
      newMaskTerms.must_include || [],
      newMaskTerms.ban_terms || [],
      newMaskTerms.avoid_terms || []
    );
    
    // Run validation
    runValidation();
    return;
  }
  
  // Handle regular prompt terms
  let positiveTerms = [...PromptsModule.getPositiveTerms()];
  let negativeTerms = [...PromptsModule.getNegativeTerms()];
  
  if (type === "positive") {
    if (action === "add") {
      // Add to positive
      const lower = term.toLowerCase();
      if (!positiveTerms.some(t => t.toLowerCase() === lower)) {
        positiveTerms.push(term);
      }
    } else if (action === "remove") {
      // Remove from positive
      positiveTerms = positiveTerms.filter(t => t.toLowerCase() !== term.toLowerCase());
    }
  } else if (type === "negative") {
    if (action === "add") {
      // Add to negative
      const lower = term.toLowerCase();
      if (!negativeTerms.some(t => t.toLowerCase() === lower)) {
        negativeTerms.push(term);
      }
    } else if (action === "remove") {
      // Remove from negative
      negativeTerms = negativeTerms.filter(t => t.toLowerCase() !== term.toLowerCase());
    }
  }
  
  // Update prompts
  PromptsModule.setTermsFromPrompts(
    PromptsModule.termsToPrompt(positiveTerms),
    PromptsModule.termsToPrompt(negativeTerms)
  );
  
  // Run validation
  runValidation();
}

// Save prompts
async function savePrompts() {
  const prompts = PromptsModule.getCurrentPrompts();
  const statusEl = $("promptsStatus");
  const btn = $("savePromptsBtn");
  
  if (btn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
  }
  statusEl.textContent = "Saving...";
  
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        positive: prompts.positive,
        negative: prompts.negative
      })
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
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("is-loading");
    }
  }
}

// Save as project default (to rules.yaml)
async function saveAsProjectDefault() {
  if (!confirm("Save current prompts and mask terms as project default in rules.yaml?\n\nThis will overwrite the existing rules.yaml configuration.")) {
    return;
  }
  
  const prompts = PromptsModule.getCurrentPrompts();
  const maskTerms = PromptsModule.getMaskTerms();
  const statusEl = $("promptsStatus");
  statusEl.textContent = "Saving to rules.yaml...";
  
  try {
    // Get current mask name
    const maskName = currentMask === "global" ? null : currentMask;
    
    // Load current rules.yaml structure
    const res = await fetch(`/api/project/${project}/config/rules_struct`);
    const rulesData = await res.json();
    
    if (!res.ok) {
      statusEl.textContent = rulesData.error || "Failed to load rules.yaml";
      return;
    }
    
    // Ensure prompts structure exists
    if (!rulesData.prompts) {
      rulesData.prompts = {};
    }
    
    // Update prompts in rules structure
    if (maskName) {
      // Save to mask-specific prompts
      if (!rulesData.prompts.masks) {
        rulesData.prompts.masks = {};
      }
      if (!rulesData.prompts.masks[maskName]) {
        rulesData.prompts.masks[maskName] = {};
      }
      rulesData.prompts.masks[maskName].positive = prompts.positive;
      rulesData.prompts.masks[maskName].negative = prompts.negative;
    } else {
      // Save to global prompts
      if (!rulesData.prompts.global) {
        rulesData.prompts.global = {};
      }
      rulesData.prompts.global.positive = prompts.positive;
      rulesData.prompts.global.negative = prompts.negative;
    }
    
    // Update mask terms in acceptance criteria
    // Find criteria that match this mask's active_criteria
    if (maskName && rulesData.masking && rulesData.masking.masks) {
      const maskConfig = rulesData.masking.masks.find(m => m.name === maskName);
      if (maskConfig && maskConfig.active_criteria) {
        const activeFields = maskConfig.active_criteria;
        if (rulesData.acceptance_criteria) {
          rulesData.acceptance_criteria.forEach(crit => {
            if (crit.field && activeFields.includes(crit.field)) {
              // Update this criterion's terms
              if (maskTerms.must_include.length > 0) {
                crit.must_include = maskTerms.must_include;
              }
              if (maskTerms.ban_terms.length > 0) {
                crit.ban_terms = maskTerms.ban_terms;
              }
              if (maskTerms.avoid_terms.length > 0) {
                crit.avoid_terms = maskTerms.avoid_terms;
              }
            }
          });
        }
      }
    }
    
    // Save updated rules (endpoint expects {rules: {...}})
    const saveRes = await fetch(`/api/project/${project}/config/rules_struct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rules: rulesData })
    });
    
    const saveData = await saveRes.json();
    if (!saveRes.ok) {
      statusEl.textContent = saveData.error || "Failed to save rules.yaml";
      return;
    }
    
    statusEl.textContent = "Saved to rules.yaml successfully!";
    setTimeout(() => { statusEl.textContent = ""; }, 3000);
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
      PromptsModule.setTermsFromPrompts(data.positive || "", data.negative || "");
      runValidation();
    }
  } catch (e) {
    console.error("Failed to load iteration prompts:", e);
  }
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
  
  const runBtn = $("runBtn");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.classList.add("is-loading");
  }
  
  // Save prompts first
  await savePrompts();
  
  // Start/continue run
  const maxIter = parseInt($("maxIterInput").value || "20", 10);
  
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
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.classList.remove("is-loading");
      }
      return;
    }
    
    runId = data.run_id;
    setStatus(`Run started: ${runId}`);
    startPolling();
    
    // Auto-describe images after a delay
    setTimeout(() => {
      describeInputTerms();
      pollOnce().then(() => {
        if (currentIteration) {
          describeLatestTerms();
        }
      });
    }, 2000);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.classList.remove("is-loading");
    }
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
  const runBtn = $("runBtn");
  const generateBtn = $("generateSuggestionBtn");
  const applyBtn = $("applySuggestionBtn");
  
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
      describeLatestTerms();
    }
  } else {
    currentIteration = null;
  }
  
  if (state.status === "running" || state.status === "starting") {
    if (spinner) spinner.classList.remove("is-hidden");
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.classList.add("is-loading");
    }
    // Disable suggestion buttons during run
    if (generateBtn) {
      generateBtn.disabled = true;
      generateBtn.title = "Wait for run to complete";
    }
    if (applyBtn) {
      applyBtn.disabled = true;
      applyBtn.title = "Wait for run to complete";
    }
  } else {
    if (spinner) spinner.classList.add("is-hidden");
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.classList.remove("is-loading");
    }
    // Enable suggestion buttons when run is complete and we have an iteration
    if (generateBtn) {
      if (currentIteration) {
        generateBtn.disabled = false;
        generateBtn.title = "";
      } else {
        generateBtn.disabled = true;
        generateBtn.title = "No iterations yet - run must complete at least one iteration";
      }
    }
    if (applyBtn) {
      applyBtn.disabled = true; // Only enabled when suggestions are generated
      applyBtn.title = "Generate suggestions first";
    }
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
  // Add null checks for all event listeners
  const resetBtn = $("resetBtn");
  if (resetBtn) resetBtn.addEventListener("click", resetRun);
  
  const runBtn = $("runBtn");
  if (runBtn) runBtn.addEventListener("click", runIteration);
  
  const maskSelect = $("maskSelect");
  if (maskSelect) {
    maskSelect.addEventListener("change", (e) => {
      currentMask = e.target.value;
      loadMaskData();
    });
  }
  
  const savePromptsBtn = $("savePromptsBtn");
  if (savePromptsBtn) savePromptsBtn.addEventListener("click", savePrompts);
  
  const saveAsDefaultBtn = $("saveAsDefaultBtn");
  if (saveAsDefaultBtn) saveAsDefaultBtn.addEventListener("click", saveAsProjectDefault);
  
  const saveSettingsBtn = $("saveSettingsBtn");
  if (saveSettingsBtn) saveSettingsBtn.addEventListener("click", saveSettings);
  
  const generateSuggestionBtn = $("generateSuggestionBtn");
  if (generateSuggestionBtn) generateSuggestionBtn.addEventListener("click", generateSuggestion);
  
  const applySuggestionBtn = $("applySuggestionBtn");
  if (applySuggestionBtn) {
    applySuggestionBtn.addEventListener("click", () => {
      SuggestionsModule.applyAllSuggestions(applySuggestion);
    });
  }
  
  const loadFromImageBtn = $("loadFromImageBtn");
  if (loadFromImageBtn) {
    loadFromImageBtn.addEventListener("click", () => {
      describeInputTerms(true); // Overwrite when user explicitly clicks
    });
  }
  
  // Expose validation function globally so prompts module can call it
  window.runValidation = runValidation;
  
  initTabs();
}

// Initialize
async function init() {
  console.log("Initializing live page...");
  
  // Initialize prompt module first
  PromptsModule.init();
  
  // Wire UI first so event handlers are ready
  wireUI();
  
  // Wait a moment for DOM to be fully ready
  await new Promise(resolve => setTimeout(resolve, 200));
  
  // Load masks (this will set currentMask and call loadMaskData)
  await loadMasks();
  
  // Load other data
  await loadSettings();
  await loadIterationPrompts();
  
  // Don't auto-load AIVis description terms - they would overwrite saved prompts
  // User can manually trigger it if needed
  const descContainer = $("inputDescription");
  if (descContainer) {
    descContainer.innerHTML = '<span class="has-text-grey">Click "Load from Image" to get AIVis terms</span>';
  }
  
  console.log("Initialization complete");
}

// Utility
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// Start initialization when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
