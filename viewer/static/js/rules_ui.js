/* global window, document, fetch */

const project = window.__RULESUI_PROJECT__;

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("rulesuiStatus").textContent = text || "";
}

function showLint(errors, warnings) {
  const box = $("rulesuiLint");
  const errs = errors || [];
  const warns = warnings || [];
  if ((!errs.length) && (!warns.length)) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  const parts = [];
  if (errs.length) {
    parts.push(`<div><strong>Errors</strong><ul>${errs.map(x => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul></div>`);
  }
  if (warns.length) {
    parts.push(`<div><strong>Warnings</strong><ul>${warns.map(x => `<li>${escapeHtml(String(x))}</li>`).join("")}</ul></div>`);
  }
  box.innerHTML = parts.join("");
  box.classList.toggle("error", !!errs.length);
  box.classList.toggle("warn", !errs.length && !!warns.length);
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
  scope: "global", // global | all | <maskName>
  selectedField: null,
  dirty: false
};

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

function scopesForCrit(c) {
  const s = c.applies_to_masks || c.mask_scope;
  if (!s) return null;
  if (Array.isArray(s)) return s.map(String);
  if (typeof s === "string") return [s];
  return [String(s)];
}

function setScopesForCrit(c, scopes) {
  if (!scopes || !scopes.length) {
    delete c.applies_to_masks;
    delete c.mask_scope;
    return;
  }
  c.applies_to_masks = scopes;
  delete c.mask_scope;
}

function scopeLabel() {
  if (state.scope === "global") return "Global";
  if (state.scope === "all") return "All";
  return `Mask: ${state.scope}`;
}

function criteriaInScope() {
  const all = getCriteria();
  if (state.scope === "all") return all;
  if (state.scope === "global") {
    return all.filter(c => !scopesForCrit(c));
  }
  return all.filter(c => {
    const scopes = scopesForCrit(c);
    if (!scopes) return false;
    return scopes.includes(state.scope);
  });
}

function renderScopes() {
  const root = $("scopeList");
  root.innerHTML = "";

  const scopes = ["global", ...maskNames(), "all"];
  for (const s of scopes) {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = (s === "global") ? "Global" : (s === "all") ? "All" : s;
    btn.addEventListener("click", () => {
      state.scope = s;
      state.selectedField = null;
      renderAll();
    });
    if (state.scope === s) btn.classList.add("active");
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

  const crits = criteriaInScope();

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
      state.selectedField = String(c.field || "");
      renderEditor();
    });

    card.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/plain", String(c.field || ""));
      ev.dataTransfer.effectAllowed = "move";
    });

    if (state.selectedField && state.selectedField === String(c.field || "")) {
      card.classList.add("selected");
    }

    (intent === "change" ? changeCol : preserveCol).appendChild(card);
  }
}

function findCriterion(field) {
  return getCriteria().find(c => String(c.field || "") === String(field));
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
    empty.classList.remove("hidden");
    editor.classList.add("hidden");
    return;
  }
  const c = findCriterion(state.selectedField);
  if (!c) {
    state.selectedField = null;
    empty.classList.remove("hidden");
    editor.classList.add("hidden");
    return;
  }

  empty.classList.add("hidden");
  editor.classList.remove("hidden");

  $("critField").textContent = String(c.field || "");
  $("critQuestion").value = String(c.question || "");
  $("critIntent").value = ((c.intent || "preserve").toLowerCase() === "change") ? "change" : "preserve";
  $("critStrength").value = String(c.edit_strength || "medium").toLowerCase();

  $("critMust").value = joinLines(c.must_include || []);
  $("critBan").value = joinLines(c.ban_terms || []);
  $("critAvoid").value = joinLines(c.avoid_terms || []);

  // scopes checkboxes
  const scopesEl = $("critScopes");
  scopesEl.innerHTML = "";
  const currentScopes = new Set(scopesForCrit(c) || []);
  const names = maskNames().filter(n => n !== "default" ? true : true); // include default too

  const mk = (name) => {
    const lab = document.createElement("label");
    lab.className = "chip";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = currentScopes.has(name);
    cb.addEventListener("change", () => {
      if (cb.checked) currentScopes.add(name);
      else currentScopes.delete(name);
      setScopesForCrit(c, Array.from(currentScopes));
      markDirty();
      renderAll();
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

      // Set scope based on selected scope
      if (state.scope === "global") setScopesForCrit(c, []);
      else if (state.scope === "all") {
        // do not change scope in "all" view
      } else {
        setScopesForCrit(c, [state.scope]);
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
  state.scope = "global";
  state.selectedField = null;
  state.dirty = false;
  $("btnSave").disabled = true;
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

function wire() {
  $("btnReload").addEventListener("click", load);
  $("btnSave").addEventListener("click", save);
  wireDnD();
}

wire();
load();

