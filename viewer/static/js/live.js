/* global window, document, fetch */

const project = window.__LIVE_PROJECT__;

let runId = null;
let pollTimer = null;
let pendingNudge = { less_random: false, too_similar: false };
let rulesLoadedText = null;
let promptsLoaded = false;

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("statusText").textContent = text || "";
}

function setRulesStatus(text) {
  const el = $("rulesStatus");
  if (el) el.textContent = text || "";
}

function setPromptsStatus(text) {
  const el = $("promptsStatus");
  if (el) el.textContent = text || "";
}

function showLintBox(title, items, kind) {
  const box = $("rulesLint");
  if (!box) return;
  if (!items || !items.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.classList.remove("error", "warn");
  box.classList.add(kind === "error" ? "error" : "warn");
  const li = items.map((x) => `<li>${escapeHtml(String(x))}</li>`).join("");
  box.innerHTML = `<strong>${escapeHtml(title)}</strong><ul>${li}</ul>`;
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setButtonActive(btn, active) {
  if (active) btn.classList.add("active");
  else btn.classList.remove("active");
}

async function startRun() {
  const reset = $("resetChk").checked;
  const maxIterations = parseInt($("maxIter").value || "20", 10);
  $("startBtn").disabled = true;
  setStatus("Starting...");

  const res = await fetch(`/api/project/${project}/live/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset, max_iterations: maxIterations })
  });
  const data = await res.json();
  if (!res.ok) {
    $("startBtn").disabled = false;
    setStatus(data.error || "Failed to start");
    return;
  }

  runId = data.run_id;
  setStatus(`Live run started: ${runId}`);
  startPolling();
}

async function loadRules() {
  $("loadRulesBtn").disabled = true;
  setRulesStatus("Loading rules...");
  try {
    const res = await fetch(`/api/project/${project}/config/rules`);
    const text = await res.text();
    if (!res.ok) throw new Error(text || "Failed to load rules");
    $("rulesBox").value = text;
    rulesLoadedText = text;
    $("saveRulesBtn").disabled = false;
    showLintBox("", [], "warn");
    setRulesStatus("Rules loaded. Edit and click Save.");
  } catch (e) {
    setRulesStatus(`Error: ${e.message}`);
  } finally {
    $("loadRulesBtn").disabled = false;
  }
}

async function saveRules() {
  const text = $("rulesBox").value || "";
  if (!text.trim()) {
    setRulesStatus("Rules are empty (not saving).");
    return;
  }

  $("saveRulesBtn").disabled = true;
  setRulesStatus("Saving rules...");
  showLintBox("", [], "warn");

  try {
    const res = await fetch(`/api/project/${project}/config/rules`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const contentType = res.headers.get("content-type") || "";
    const data = contentType.includes("application/json") ? await res.json() : { error: await res.text() };

    if (!res.ok) {
      const errs = data.errors || [];
      const warns = data.warnings || [];
      if (errs.length) showLintBox("Errors (fix these before saving):", errs, "error");
      if (warns.length) showLintBox("Warnings:", warns, "warn");
      setRulesStatus(data.error || "Failed to save rules");
      $("saveRulesBtn").disabled = false;
      return;
    }

    const warns = data.warnings || [];
    if (warns.length) showLintBox("Warnings:", warns, "warn");
    else showLintBox("", [], "warn");

    rulesLoadedText = text;
    setRulesStatus("Saved. Changes will apply from the next iteration.");
  } catch (e) {
    setRulesStatus(`Error: ${e.message}`);
    $("saveRulesBtn").disabled = false;
  }
}

async function loadPrompts() {
  $("loadPromptsBtn").disabled = true;
  setPromptsStatus("Loading prompts...");
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load prompts");
    $("positivePromptBox").value = data.positive || "";
    $("negativePromptBox").value = data.negative || "";
    $("savePromptsBtn").disabled = false;
    promptsLoaded = true;
    setPromptsStatus("Prompts loaded. Edit and click Save.");
  } catch (e) {
    setPromptsStatus(`Error: ${e.message}`);
  } finally {
    $("loadPromptsBtn").disabled = false;
  }
}

async function savePrompts() {
  if (!promptsLoaded) {
    setPromptsStatus("Load prompts first.");
    return;
  }
  const positive = $("positivePromptBox").value || "";
  const negative = $("negativePromptBox").value || "";
  $("savePromptsBtn").disabled = true;
  setPromptsStatus("Saving prompts...");
  try {
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ positive, negative })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to save prompts");
    setPromptsStatus("Saved. Next iteration will use these prompts.");
  } catch (e) {
    setPromptsStatus(`Error: ${e.message}`);
  } finally {
    $("savePromptsBtn").disabled = false;
  }
}

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
  const submitBtn = $("submitFeedbackBtn");
  const banner = $("turnBanner");
  const spinner = $("spinnerOverlay");
  const feedbackPanel = $("feedbackPanel");

  if (latest) {
    const n = latest.iteration_number;
    $("latestImg").src = `/api/project/${project}/run/${runId}/image/iteration_${n}.png`;
    const score = (latest.evaluation || {}).overall_score;
    $("latestMeta").textContent = `Iteration ${n} | score: ${score}`;
  }

  if (state.status === "waiting") {
    submitBtn.disabled = false;
    $("feedbackHint").textContent = `Waiting for your feedback for iteration ${state.expected_feedback_for_iteration}.`;
    banner.classList.remove("hidden");
    spinner.classList.add("hidden");
    feedbackPanel.classList.add("attention");
  } else if (state.status === "running" || state.status === "starting") {
    submitBtn.disabled = true;
    $("feedbackHint").textContent = "Generating... (feedback will unlock between iterations)";
    banner.classList.add("hidden");
    spinner.classList.remove("hidden");
    feedbackPanel.classList.remove("attention");
  } else if (state.status === "finished") {
    submitBtn.disabled = true;
    $("feedbackHint").textContent = state.message || "Finished";
    banner.classList.add("hidden");
    spinner.classList.add("hidden");
    feedbackPanel.classList.remove("attention");
  } else if (state.status === "error") {
    submitBtn.disabled = true;
    $("feedbackHint").textContent = `Error: ${state.message || "unknown"}`;
    banner.classList.add("hidden");
    spinner.classList.add("hidden");
    feedbackPanel.classList.remove("attention");
  } else {
    submitBtn.disabled = true;
    $("feedbackHint").textContent = "";
    banner.classList.add("hidden");
    spinner.classList.add("hidden");
    feedbackPanel.classList.remove("attention");
  }

  setStatus(`${state.status} | iter ${state.current_iteration} | best ${state.best_score} (iter ${state.best_iteration || "-"})`);
}

function renderIterations(iterations) {
  const list = $("iterationsList");
  list.innerHTML = "";

  const reversed = [...iterations].reverse();
  reversed.forEach((it) => {
    const n = it.iteration_number;
    const score = (it.evaluation || {}).overall_score;
    const div = document.createElement("div");
    div.className = "iteration-item";
    div.innerHTML = `
      <div class="iteration-header">
        <strong>Iteration ${n}</strong>
        <span class="muted">score: ${score}</span>
      </div>
      <img class="thumb" src="/api/project/${project}/run/${runId}/image/iteration_${n}.png" alt="iteration ${n}">
    `;
    list.appendChild(div);
  });
}

async function pollOnce() {
  try {
    const state = await fetchState();
    const iterations = await fetchIterations();
    if (!state) return;
    renderLatest(state, iterations);
    renderIterations(iterations);
  } catch (e) {
    setStatus(`poll error: ${e.message}`);
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  pollTimer = setInterval(pollOnce, 2000);
}

async function submitFeedback() {
  if (!runId) return;

  const state = await fetchState();
  const iteration = state.expected_feedback_for_iteration || state.current_iteration || 0;
  const comment = $("commentBox").value || "";

  $("submitFeedbackBtn").disabled = true;
  setStatus("Submitting feedback...");

  const res = await fetch(`/api/project/${project}/live/${runId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iteration, comment, nudge: pendingNudge })
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus(data.error || "Failed to submit feedback");
    $("submitFeedbackBtn").disabled = false;
    return;
  }

  // Reset UI nudge state after submit
  pendingNudge = { less_random: false, too_similar: false };
  setButtonActive($("btnLessRandom"), false);
  setButtonActive($("btnTooSimilar"), false);
  $("commentBox").value = "";
  setStatus("Feedback submitted, continuing...");
  pollOnce();
}

function wireUI() {
  $("startBtn").addEventListener("click", startRun);
  $("submitFeedbackBtn").addEventListener("click", submitFeedback);
  $("loadRulesBtn").addEventListener("click", loadRules);
  $("saveRulesBtn").addEventListener("click", saveRules);
  $("loadPromptsBtn").addEventListener("click", loadPrompts);
  $("savePromptsBtn").addEventListener("click", savePrompts);

  $("btnLessRandom").addEventListener("click", () => {
    pendingNudge.less_random = !pendingNudge.less_random;
    setButtonActive($("btnLessRandom"), pendingNudge.less_random);
  });
  $("btnTooSimilar").addEventListener("click", () => {
    pendingNudge.too_similar = !pendingNudge.too_similar;
    setButtonActive($("btnTooSimilar"), pendingNudge.too_similar);
  });
}

wireUI();

