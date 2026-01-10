/* global window, document, fetch */

const project = window.__RULESUI_PROJECT__;

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("rulesuiStatus").textContent = text || "";
}

function setTextStatus(id, text) {
  const el = $(id);
  if (!el) return;
  el.textContent = text || "";
}

function openYamlModal() {
  const m = $("yamlModal");
  m.classList.add("is-active");
  // Copy current preview into modal
  $("yamlModalText").value = $("yamlPreview").value || "";
}

function closeYamlModal() {
  const m = $("yamlModal");
  m.classList.remove("is-active");
}

async function copyYaml() {
  const text = $("yamlModalText").value || $("yamlPreview").value || "";
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied YAML to clipboard.");
  } catch (e) {
    // Fallback
    $("yamlModalText").focus();
    $("yamlModalText").select();
    document.execCommand("copy");
    setStatus("Copied YAML.");
  }
}

function showLint(errors, warnings) {
  const box = $("rulesuiLint");
  const errs = errors || [];
  const warns = warnings || [];
  if ((!errs.length) && (!warns.length)) {
    box.classList.add("is-hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("is-hidden");
  const parts = [];
  if (errs.length) {
    box.classList.remove("is-warning");
    box.classList.add("is-danger");
    parts.push(`<strong>Errors</strong><ul>${errs.map(x => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`);
  } else if (warns.length) {
    box.classList.remove("is-danger");
    box.classList.add("is-warning");
    parts.push(`<strong>Warnings</strong><ul>${warns.map(x => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul>`);
  }
  box.innerHTML = parts.join("");
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

let state = {
  masks: [],
  rules: null,
  scope: "default", // <maskName> | all
  selectedField: null,
  dirty: false
};

function scopeKeyForPrompts() {
  // "Show all" is a view filter; treat it as global prompts.
  if (state.scope === "all") return "global";
  // "default" is still a mask scope; prompts may exist per-mask.
  return state.scope || "global";
}

function getScopePrompts() {
  const rules = state.rules || {};
  const prompts = rules.prompts || {};
  const key = scopeKeyForPrompts();
  if (key !== "global" && prompts.masks && typeof prompts.masks === "object" && prompts.masks[key]) {
    const p = prompts.masks[key] || {};
    return { positive: String(p.positive || ""), negative: String(p.negative || "") };
  }
  const g = prompts.global || {};
  return { positive: String(g.positive || ""), negative: String(g.negative || "") };
}

function renderScopePrompts() {
  const posEl = $("scopePromptPositive");
  const negEl = $("scopePromptNegative");
  if (!posEl || !negEl) return;
  const p = getScopePrompts();
  posEl.value = p.positive || "";
  negEl.value = p.negative || "";
}

function updateScopePreview() {
  const maskImg = $("scopeMaskImg");
  if (!maskImg) return;
  // For "show all", hide overlay.
  if (state.scope === "all") {
    maskImg.src = "";
    maskImg.style.display = "none";
    return;
  }
  // If the mask file doesn't exist yet, keep the preview clean (no 404 noise).
  const exists = (state.masks || []).some(m => (m && m.name) === state.scope);
  if (!exists && state.scope !== "default") {
    maskImg.src = "";
    maskImg.style.display = "none";
    return;
  }
  maskImg.style.display = "block";
  maskImg.src = `/api/project/${project}/input/mask/${encodeURIComponent(state.scope)}`;
}

function markDirty() {
  state.dirty = true;
  $("btnSave").disabled = false;
}

function getCriteria() {
  const rules = state.rules || {};
  const arr = rules.acceptance_criteria || [];
  return Array.isArray(arr) ? arr : [];
}

function setCriteria(arr) {
  state.rules.acceptance_criteria = arr;
}

function getQuestions() {
  const rules = state.rules || {};
  const arr = rules.questions || [];
  return Array.isArray(arr) ? arr : [];
}

function maskNames() {
  const ms = state.masks || [];
  const names = ms.map(m => m.name).filter(Boolean);
  if (!names.includes("default")) names.unshift("default");
  return names;
}

function membership() {
  const rules = state.rules || {};
  const masking = rules.masking || {};
  const masks = masking.masks || [];
  return Array.isArray(masks) ? masks : [];
}

function getActiveSet(maskName) {
  const m = membership().find(x => String(x.name || "") === String(maskName));
  const arr = (m && m.active_criteria) ? m.active_criteria : [];
  const list = Array.isArray(arr) ? arr.map(String) : [];
  return new Set(list);
}

function setActiveSet(maskName, set) {
  const masks = membership();
  let m = masks.find(x => String(x.name || "") === String(maskName));
  if (!m) {
    m = { name: maskName, active_criteria: [] };
    masks.push(m);
  }
  m.active_criteria = Array.from(set);
  state.rules.masking = state.rules.masking || {};
  state.rules.masking.masks = masks;
}

function toggleCriterionInMask(maskName, field) {
  const s = getActiveSet(maskName);
  if (s.has(field)) s.delete(field);
  else s.add(field);
  setActiveSet(maskName, s);
  markDirty();
  renderAll();
}

function scopeLabel() {
  if (state.scope === "all") return "All criteria (view)";
  return `Mask: ${state.scope}`;
}

function criteriaInScope() {
  const all = getCriteria();
  if (state.scope === "all") return all;
  const active = getActiveSet(state.scope);
  return all.filter(c => active.has(String(c.field || "")));
}

function renderScopes() {
  const root = $("scopeList");
  root.innerHTML = "";

  const scopes = [...maskNames(), "all"];
  for (const s of scopes) {
    const btn = document.createElement("button");
    btn.className = "button is-small";
    btn.textContent = (s === "all") ? "Show all" : `Mask: ${s}`;
    btn.addEventListener("click", () => {
      state.scope = s;
      state.selectedField = null;
      renderAll();
    });
    if (state.scope === s) {
      btn.classList.add("is-active");
      btn.classList.add("is-primary");
    }
    root.appendChild(btn);
  }
}

function renderQuestions() {
  const q = getQuestions();
  const el = $("questionsList");
  if (!q.length) {
    el.textContent = "No questions configured.";
    return;
  }
  const items = q.map(x => {
    const f = escapeHtml(String(x.field || ""));
    const qt = escapeHtml(String(x.question || ""));
    const t = escapeHtml(String(x.type || "string"));
    return `<div class="question-item"><span class="mono">${f}</span> <span class="muted">(${t})</span><div class="muted">${qt}</div></div>`;
  });
  el.innerHTML = items.join("");
}

function renderBoard() {
  const changeCol = $("colChange");
  const preserveCol = $("colPreserve");
  changeCol.innerHTML = "";
  preserveCol.innerHTML = "";

  const all = getCriteria();
  const active = (state.scope === "all") ? new Set(all.map(c => String(c.field || ""))) : getActiveSet(state.scope);
  const activeCrits = all.filter(c => active.has(String(c.field || "")));
  const inactiveCrits = all.filter(c => !active.has(String(c.field || "")));
  const crits = (state.scope === "all") ? all : [...activeCrits, ...inactiveCrits];

  for (const c of crits) {
    const intent = (c.intent || "preserve").toLowerCase() === "change" ? "change" : "preserve";
    const card = document.createElement("div");
    card.className = "crit-card";
    card.draggable = true;
    card.dataset.field = String(c.field || "");
    card.dataset.intent = intent;

    const title = document.createElement("div");
    title.className = "crit-title";
    title.textContent = String(c.field || "(missing field)");

    const q = document.createElement("div");
    q.className = "crit-question muted";
    q.textContent = String(c.question || "");

    const meta = document.createElement("div");
    meta.className = "crit-meta muted";
    meta.textContent = `${intent} • strength=${String(c.edit_strength || "medium")}`;

    card.appendChild(title);
    card.appendChild(q);
    card.appendChild(meta);

    card.addEventListener("click", () => {
      const f = String(c.field || "");
      if (state.scope !== "all" && !active.has(f)) {
        toggleCriterionInMask(state.scope, f);
        return;
      }
      state.selectedField = f;
      renderEditor();
    });

    card.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/plain", String(c.field || ""));
      ev.dataTransfer.effectAllowed = "move";
    });

    if (state.selectedField && state.selectedField === String(c.field || "")) {
      card.classList.add("selected");
    }
    if (state.scope !== "all" && !active.has(String(c.field || ""))) {
      card.classList.add("inactive");
    }

    (intent === "change" ? changeCol : preserveCol).appendChild(card);
  }
}

function findCriterion(field) {
  return getCriteria().find(c => String(c.field || "") === String(field));
}

function removeCriterionEverywhere(field) {
  const f = String(field || "");
  // Remove from acceptance_criteria list
  const crits = getCriteria().filter(c => String(c.field || "") !== f);
  setCriteria(crits);
  // Remove from all mask memberships
  const masks = membership();
  for (const m of masks) {
    if (!m || typeof m !== "object") continue;
    const arr = Array.isArray(m.active_criteria) ? m.active_criteria : [];
    m.active_criteria = arr.filter(x => String(x) !== f);
  }
  state.rules.masking = state.rules.masking || {};
  state.rules.masking.masks = masks;
  if (state.selectedField === f) state.selectedField = null;
  markDirty();
  renderAll();
}

function splitLines(s) {
  return String(s || "")
    .split("\n")
    .map(x => x.trim())
    .filter(Boolean);
}

function joinLines(arr) {
  return (arr || []).map(String).join("\n");
}

function renderEditor() {
  const empty = $("criterionEmpty");
  const editor = $("criterionEditor");
  if (!state.selectedField) {
    empty.classList.remove("is-hidden");
    editor.classList.add("is-hidden");
    return;
  }
  const c = findCriterion(state.selectedField);
  if (!c) {
    state.selectedField = null;
    empty.classList.remove("is-hidden");
    editor.classList.add("is-hidden");
    return;
  }

  empty.classList.add("is-hidden");
  editor.classList.remove("is-hidden");

  $("critField").textContent = String(c.field || "");
  $("critQuestion").value = String(c.question || "");
  $("critIntent").value = ((c.intent || "preserve").toLowerCase() === "change") ? "change" : "preserve";
  $("critStrength").value = String(c.edit_strength || "medium").toLowerCase();

  $("critMust").value = joinLines(c.must_include || []);
  $("critBan").value = joinLines(c.ban_terms || []);
  $("critAvoid").value = joinLines(c.avoid_terms || []);

  // Header actions
  const btnDel = $("btnDeleteCriterion");
  if (btnDel) {
    btnDel.onclick = () => {
      const f = String(c.field || "");
      if (!f) return;
      if (!window.confirm(`Delete criterion '${f}'?\n\nThis removes it from rules.yaml and all mask memberships.`)) return;
      removeCriterionEverywhere(f);
    };
  }

  const btnDeact = $("btnDeactivateCurrent");
  if (btnDeact) {
    const can = state.scope !== "all";
    btnDeact.disabled = !can;
    btnDeact.onclick = () => {
      if (state.scope === "all") return;
      const f = String(c.field || "");
      const s = getActiveSet(state.scope);
      if (s.has(f)) {
        s.delete(f);
        setActiveSet(state.scope, s);
        markDirty();
        renderAll();
      }
    };
  }

  // mask membership checkboxes
  const scopesEl = $("critScopes");
  scopesEl.innerHTML = "";
  const field = String(c.field || "");
  const names = maskNames();

  const mk = (name) => {
    const lab = document.createElement("label");
    lab.className = "chip";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = getActiveSet(name).has(field);
    cb.addEventListener("change", () => {
      toggleCriterionInMask(name, field);
    });
    lab.appendChild(cb);
    const t = document.createElement("span");
    t.textContent = name;
    lab.appendChild(t);
    return lab;
  };

  for (const n of names) scopesEl.appendChild(mk(n));

  // Wire updates (simple, overwrite on input)
  $("critQuestion").oninput = () => { c.question = $("critQuestion").value; markDirty(); renderAll(); };
  $("critIntent").onchange = () => { c.intent = $("critIntent").value; markDirty(); renderAll(); };
  $("critStrength").onchange = () => { c.edit_strength = $("critStrength").value; markDirty(); renderAll(); };
  $("critMust").oninput = () => { c.must_include = splitLines($("critMust").value); markDirty(); };
  $("critBan").oninput = () => { c.ban_terms = splitLines($("critBan").value); markDirty(); };
  $("critAvoid").oninput = () => { c.avoid_terms = splitLines($("critAvoid").value); markDirty(); };
}

async function refreshYamlPreview() {
  const res = await fetch(`/api/project/${project}/config/rules_render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rules: state.rules })
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    setStatus(data.error || "Failed to render YAML");
    return;
  }
  $("yamlPreview").value = data.yaml || "";
  showLint(data.errors || [], data.warnings || []);
}

function wireDnD() {
  const zones = document.querySelectorAll(".dropzone");
  for (const z of zones) {
    z.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      z.classList.add("drop-hover");
    });
    z.addEventListener("dragleave", () => z.classList.remove("drop-hover"));
    z.addEventListener("drop", (ev) => {
      ev.preventDefault();
      z.classList.remove("drop-hover");
      const field = ev.dataTransfer.getData("text/plain");
      if (!field) return;
      const c = findCriterion(field);
      if (!c) return;

      // Set intent based on which column we dropped into
      const parentCol = z.closest(".board-col");
      const intent = parentCol && parentCol.dataset.intent === "change" ? "change" : "preserve";
      c.intent = intent;

      // Ensure membership: if we're in a specific mask scope, activate it there
      if (state.scope !== "all") {
        const s = getActiveSet(state.scope);
        s.add(field);
        setActiveSet(state.scope, s);
      }

      markDirty();
      renderAll();
    });
  }
}

function renderProjectSettings() {
  const proj = (state.rules && state.rules.project) ? state.rules.project : {};
  const maxIt = parseInt(proj.max_iterations || "20", 10);
  $("projMaxIterations").value = String(Number.isFinite(maxIt) ? maxIt : 20);

  const ph = proj.preserve_heavy;
  $("projPreserveHeavy").value = (ph === true) ? "true" : (ph === false) ? "false" : "auto";
  $("projLockSeed").value = (proj.lock_seed === false) ? "false" : "true";

  $("projMaxIterations").onchange = () => {
    state.rules.project = state.rules.project || {};
    state.rules.project.max_iterations = parseInt($("projMaxIterations").value || "20", 10);
    markDirty();
    refreshYamlPreview();
  };
  $("projPreserveHeavy").onchange = () => {
    state.rules.project = state.rules.project || {};
    const v = $("projPreserveHeavy").value;
    if (v === "auto") delete state.rules.project.preserve_heavy;
    else state.rules.project.preserve_heavy = (v === "true");
    markDirty();
    refreshYamlPreview();
  };
  $("projLockSeed").onchange = () => {
    state.rules.project = state.rules.project || {};
    state.rules.project.lock_seed = ($("projLockSeed").value === "true");
    markDirty();
    refreshYamlPreview();
  };
}

function renderAll() {
  renderScopes();
  renderProjectSettings();
  renderScopePrompts();
  updateScopePreview();
  updateEditMaskLink();
  renderBoard();
  renderQuestions();
  renderEditor();
  refreshYamlPreview();
  setStatus(`Scope: ${scopeLabel()}${state.dirty ? " • unsaved changes" : ""}`);
}

async function load() {
  $("btnSave").disabled = true;
  state.dirty = false;
  setStatus("Loading rules...");
  const res = await fetch(`/api/project/${project}/config/rules_struct`, { cache: "no-store" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    setStatus(data.error || "Failed to load");
    return;
  }
  state.masks = data.masks || [];
  state.rules = data.rules || {};
  state.scope = "default";
  state.selectedField = null;
  state.dirty = false;
  $("btnSave").disabled = true;
  renderAll();
}

function normaliseFieldName(s) {
  return String(s || "")
    .trim()
    .toLowerCase()
    .replaceAll(/[^a-z0-9_]+/g, "_")
    .replaceAll(/_{2,}/g, "_")
    .replaceAll(/^_+|_+$/g, "");
}

function validateMaskName(name) {
  const n = (name || "").trim();
  if (!n) return null;
  if (n === "default") return "default";
  if (!/^[a-zA-Z0-9_-]+$/.test(n)) return null;
  return n;
}

function updateEditMaskLink() {
  const a = $("btnEditMask");
  if (!a) return;
  const mask = (state.scope === "all") ? "default" : state.scope;
  a.href = `/project/${project}/mask?mask=${encodeURIComponent(mask)}`;
  if (state.scope === "all") a.classList.add("disabled");
  else a.classList.remove("disabled");
}

function createCriterion() {
  const raw = window.prompt("New criterion field name (e.g. left_outfit):");
  if (!raw) return;
  const field = normaliseFieldName(raw);
  if (!field) {
    setStatus("Invalid field name.");
    return;
  }
  const crits = getCriteria();
  if (crits.some(c => String(c.field || "") === field)) {
    setStatus(`Field already exists: ${field}`);
    return;
  }
  const c = {
    field,
    question: "Describe the goal for this criterion.",
    type: "boolean",
    min: 0,
    max: 1,
    intent: "change",
    edit_strength: "medium",
    must_include: [],
    ban_terms: [],
    avoid_terms: []
  };
  // Add to membership for the current mask scope (or default if in Show all view).
  const targetMask = (state.scope === "all") ? "default" : state.scope;
  const s = getActiveSet(targetMask);
  s.add(field);
  setActiveSet(targetMask, s);
  crits.push(c);
  setCriteria(crits);
  state.selectedField = field;
  markDirty();
  renderAll();
}

function createQuestion() {
  const raw = window.prompt("New question field name (e.g. left_outfit_desc):");
  if (!raw) return;
  const field = normaliseFieldName(raw);
  if (!field) {
    setStatus("Invalid question field name.");
    return;
  }
  const q = getQuestions();
  if (q.some(x => String(x.field || "") === field)) {
    setStatus(`Question field already exists: ${field}`);
    return;
  }
  q.push({
    field,
    question: "Write a question the vision model can answer about the image.",
    type: "string"
  });
  state.rules.questions = q;
  markDirty();
  renderAll();
}

async function save() {
  $("btnSave").disabled = true;
  setStatus("Saving rules.yaml...");
  const res = await fetch(`/api/project/${project}/config/rules_struct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rules: state.rules })
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    $("btnSave").disabled = false;
    showLint(data.errors || [], data.warnings || []);
    if (data.yaml) $("yamlPreview").value = data.yaml;
    setStatus(data.error || "Save failed (fix errors).");
    return;
  }
  state.dirty = false;
  $("btnSave").disabled = true;
  setStatus("Saved. Next run will use these rules.");
  // Reload to pick up any normalisation by the writer.
  await load();
}

async function generateBasePrompts() {
  const scope = state.scope || "all";
  setStatus(`Generating base prompts for scope: ${scopeLabel()}...`);
  $("btnGenPrompts").disabled = true;
  try {
    const res = await fetch(`/api/project/${project}/config/rules_generate_base_prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      setStatus(data.error || "Prompt generation failed.");
      return;
    }
    // Update UI immediately (we also reload to ensure state matches disk).
    try {
      state.rules = state.rules || {};
      state.rules.prompts = state.rules.prompts || { global: { positive: "", negative: "" }, masks: {} };
      if (data.scope && data.scope !== "global") {
        state.rules.prompts.masks = state.rules.prompts.masks || {};
        state.rules.prompts.masks[data.scope] = { positive: String(data.positive || ""), negative: String(data.negative || "") };
      } else {
        state.rules.prompts.global = { positive: String(data.positive || ""), negative: String(data.negative || "") };
      }
      renderScopePrompts();
    } catch (e) {
      // ignore UI-only update errors
    }
    setStatus(`Generated base prompts for ${data.scope}. Reloading...`);
    await load();
  } finally {
    $("btnGenPrompts").disabled = false;
  }
}

function wire() {
  $("btnReload").addEventListener("click", load);
  $("btnSave").addEventListener("click", save);
  $("btnAddCriterion").addEventListener("click", createCriterion);
  $("btnAddQuestion").addEventListener("click", createQuestion);
  $("btnShowYaml").addEventListener("click", openYamlModal);
  $("btnGenPrompts").addEventListener("click", generateBasePrompts);
  $("btnCloseYaml")?.addEventListener("click", closeYamlModal);
  $("btnCloseYaml2")?.addEventListener("click", closeYamlModal);
  $("yamlBackdrop")?.addEventListener("click", closeYamlModal);
  $("btnCopyYaml").addEventListener("click", copyYaml);

  // Advanced: working prompts
  const wpPos = $("workingPromptPositive");
  const wpNeg = $("workingPromptNegative");
  const btnLoadWP = $("btnLoadWorkingPrompts");
  const btnSaveWP = $("btnSaveWorkingPrompts");
  async function loadWorkingPrompts() {
    setTextStatus("workingPromptsStatus", "Loading...");
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTextStatus("workingPromptsStatus", data.error || "Failed to load.");
      return;
    }
    wpPos.value = data.positive || "";
    wpNeg.value = data.negative || "";
    btnSaveWP.disabled = true;
    setTextStatus("workingPromptsStatus", "Loaded.");
  }
  async function saveWorkingPrompts() {
    btnSaveWP.disabled = true;
    setTextStatus("workingPromptsStatus", "Saving...");
    const res = await fetch(`/api/project/${project}/working/aigen/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ positive: wpPos.value || "", negative: wpNeg.value || "" })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTextStatus("workingPromptsStatus", data.error || "Save failed.");
      return;
    }
    setTextStatus("workingPromptsStatus", "Saved to working/AIGen.yaml.");
  }
  if (btnLoadWP) btnLoadWP.addEventListener("click", loadWorkingPrompts);
  if (btnSaveWP) btnSaveWP.addEventListener("click", saveWorkingPrompts);
  if (wpPos) wpPos.addEventListener("input", () => { btnSaveWP.disabled = false; });
  if (wpNeg) wpNeg.addEventListener("input", () => { btnSaveWP.disabled = false; });

  // Advanced: config/AIGen.yaml editor
  const aigenBox = $("aigenCfgBox");
  const btnLoadAigen = $("btnLoadAigenCfg");
  const btnSaveAigen = $("btnSaveAigenCfg");
  async function loadAigenCfg() {
    setTextStatus("aigenCfgStatus", "Loading...");
    const res = await fetch(`/api/project/${project}/config/aigen`, { cache: "no-store" });
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      setTextStatus("aigenCfgStatus", text || "Failed to load.");
      return;
    }
    aigenBox.value = text || "";
    btnSaveAigen.disabled = true;
    setTextStatus("aigenCfgStatus", "Loaded.");
  }
  async function saveAigenCfg() {
    btnSaveAigen.disabled = true;
    setTextStatus("aigenCfgStatus", "Saving...");
    const res = await fetch(`/api/project/${project}/config/aigen`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: aigenBox.value || "" })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTextStatus("aigenCfgStatus", data.error || "Save failed.");
      btnSaveAigen.disabled = false;
      return;
    }
    setTextStatus("aigenCfgStatus", "Saved config/AIGen.yaml.");
  }
  if (btnLoadAigen) btnLoadAigen.addEventListener("click", loadAigenCfg);
  if (btnSaveAigen) btnSaveAigen.addEventListener("click", saveAigenCfg);
  if (aigenBox) aigenBox.addEventListener("input", () => { btnSaveAigen.disabled = false; });

  // Advanced: config/AIVis.yaml editor
  const aivisBox = $("aivisCfgBox");
  const btnLoadAiVis = $("btnLoadAiVisCfg");
  const btnSaveAiVis = $("btnSaveAiVisCfg");
  async function loadAiVisCfg() {
    setTextStatus("aivisCfgStatus", "Loading...");
    const res = await fetch(`/api/project/${project}/config/aivis`, { cache: "no-store" });
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      setTextStatus("aivisCfgStatus", text || "Failed to load.");
      return;
    }
    aivisBox.value = text || "";
    btnSaveAiVis.disabled = true;
    setTextStatus("aivisCfgStatus", "Loaded.");
  }
  async function saveAiVisCfg() {
    btnSaveAiVis.disabled = true;
    setTextStatus("aivisCfgStatus", "Saving...");
    const res = await fetch(`/api/project/${project}/config/aivis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: aivisBox.value || "" })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setTextStatus("aivisCfgStatus", data.error || "Save failed.");
      btnSaveAiVis.disabled = false;
      return;
    }
    setTextStatus("aivisCfgStatus", "Saved config/AIVis.yaml.");
  }
  if (btnLoadAiVis) btnLoadAiVis.addEventListener("click", loadAiVisCfg);
  if (btnSaveAiVis) btnSaveAiVis.addEventListener("click", saveAiVisCfg);
  if (aivisBox) aivisBox.addEventListener("input", () => { btnSaveAiVis.disabled = false; });

  const btnCreate = $("btnCreateMaskRulesUI");
  if (btnCreate) {
    btnCreate.addEventListener("click", () => {
      const raw = $("newMaskNameRulesUI").value || "";
      const name = validateMaskName(raw);
      if (!name) {
        setStatus("Invalid mask name. Use letters, numbers, _ or -.");
        return;
      }
      window.location.href = `/project/${project}/mask?mask=${encodeURIComponent(name)}`;
    });
  }
  wireDnD();
}

wire();
load();

