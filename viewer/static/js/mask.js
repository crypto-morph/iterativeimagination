/* global window, document, fetch */

const project = window.__MASK_PROJECT__;
let maskName = (window.__MASK_NAME__ || "default").trim() || "default";

function $(id) {
  return document.getElementById(id);
}

let tool = "brush"; // brush|eraser
let drawing = false;
let scaleX = 1;
let scaleY = 1;
let anchor = null; // {x,y} in image pixel coords

function setStatus(text) {
  $("maskStatus").textContent = text || "";
}

function setMaskName(name) {
  maskName = (name || "default").trim() || "default";
  const url = new URL(window.location.href);
  url.searchParams.set("mask", maskName);
  window.history.replaceState({}, "", url.toString());
}

async function refreshMaskList() {
  const sel = $("maskSelect");
  if (!sel) return;
  try {
    const res = await fetch(`/api/project/${project}/input/masks`);
    const data = await res.json().catch(() => ({}));
    const masks = (data && data.masks) || [];

    // Always include default in UI (even if it doesn't exist yet).
    const names = new Set(masks.map(m => m.name));
    if (!names.has("default")) {
      masks.unshift({ name: "default", file: "input/mask.png" });
    }

    sel.innerHTML = "";
    for (const m of masks) {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      sel.appendChild(opt);
    }
    sel.value = maskName;
    if (sel.value !== maskName) {
      // If current mask isn't in list yet, add it.
      const opt = document.createElement("option");
      opt.value = maskName;
      opt.textContent = maskName;
      sel.appendChild(opt);
      sel.value = maskName;
    }
  } catch (e) {
    // Non-fatal
  }
}

function validateMaskName(name) {
  const n = (name || "").trim();
  if (!n) return null;
  if (n === "default") return "default";
  if (!/^[a-zA-Z0-9_-]+$/.test(n)) return null;
  return n;
}

function setActiveTool(name) {
  tool = name;
  $("toolBrush").classList.toggle("is-active", tool === "brush");
  $("toolEraser").classList.toggle("is-active", tool === "eraser");
}

function getBrushSize() {
  return parseInt($("brushSize").value || "24", 10);
}

function resizeCanvasToImage() {
  const img = $("baseImg");
  const canvas = $("maskCanvas");

  const w = img.naturalWidth;
  const h = img.naturalHeight;
  canvas.width = w;
  canvas.height = h;

  // Set CSS size to match rendered image
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";

  scaleX = w / img.clientWidth;
  scaleY = h / img.clientHeight;

  // Default: black mask (keep)
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, w, h);

  updateAnchorMarker();
}

function setAnchor(a) {
  if (!a) {
    anchor = null;
  } else {
    anchor = { x: Math.round(a.x), y: Math.round(a.y) };
  }
  updateAnchorMarker();
  const t = $("anchorText");
  if (t) t.textContent = anchor ? `${anchor.x}, ${anchor.y}` : "none";
}

function updateAnchorMarker() {
  const m = $("anchorMarker");
  if (!m) return;
  if (!anchor) {
    m.classList.add("is-hidden");
    return;
  }
    m.classList.remove("is-hidden");
  // Convert image px -> displayed px
  const xCss = anchor.x / scaleX;
  const yCss = anchor.y / scaleY;
  m.style.left = `${xCss}px`;
  m.style.top = `${yCss}px`;
}

async function loadAnchor() {
  try {
    const res = await fetch(`/api/project/${project}/input/mask_anchor/${encodeURIComponent(maskName)}`, { cache: "no-store" });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data && data.anchor) setAnchor(data.anchor);
    else setAnchor(null);
  } catch (e) {
    setAnchor(null);
  }
}

async function saveAnchor() {
  try {
    await fetch(`/api/project/${project}/input/mask_anchor/${encodeURIComponent(maskName)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ anchor })
    });
  } catch (e) {
    // non-fatal
  }
}

async function clearAnchor() {
  setAnchor(null);
  await saveAnchor();
  setStatus("Anchor cleared.");
}

function pointFromEvent(ev) {
  const canvas = $("maskCanvas");
  const rect = canvas.getBoundingClientRect();
  const clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
  const clientY = ev.touches ? ev.touches[0].clientY : ev.clientY;
  const x = (clientX - rect.left) * scaleX;
  const y = (clientY - rect.top) * scaleY;
  return { x, y };
}

function drawAt(x, y) {
  const canvas = $("maskCanvas");
  const ctx = canvas.getContext("2d");
  const r = getBrushSize() / 2;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = getBrushSize();

  if (tool === "eraser") {
    ctx.strokeStyle = "rgb(0,0,0)";
  } else {
    ctx.strokeStyle = "rgb(255,255,255)";
  }

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 0.01, y + 0.01);
  ctx.stroke();

  // Dot (helps on click without move)
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
}

function startDraw(ev) {
  // Alt+Click sets an anchor point instead of drawing.
  if (ev && ev.altKey) {
    const p = pointFromEvent(ev);
    setAnchor(p);
    saveAnchor();
    setStatus(`Anchor set to ${anchor.x}, ${anchor.y}`);
    ev.preventDefault();
    return;
  }
  drawing = true;
  const p = pointFromEvent(ev);
  drawAt(p.x, p.y);
  ev.preventDefault();
}

function moveDraw(ev) {
  if (!drawing) return;
  const p = pointFromEvent(ev);
  drawAt(p.x, p.y);
  ev.preventDefault();
}

function stopDraw() {
  drawing = false;
}

async function loadExistingMask() {
  setStatus(`Loading mask '${maskName}'...`);
  const canvas = $("maskCanvas");
  const ctx = canvas.getContext("2d");

  try {
    const res = await fetch(`/api/project/${project}/input/mask/${encodeURIComponent(maskName)}`);
    if (!res.ok) {
      setStatus("No saved mask found (starting blank).");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      setStatus("Loaded existing mask.");
    };
    img.src = url;
  } catch (e) {
    setStatus(`Failed to load: ${e.message}`);
  }
}

function clearMask() {
  const canvas = $("maskCanvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(0,0,0)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  setStatus("Cleared.");
}

function fillWhite() {
  const canvas = $("maskCanvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgb(255,255,255)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  setStatus("Filled white (editable everywhere).");
}

async function saveMask() {
  setStatus(`Saving '${maskName}'...`);
  const canvas = $("maskCanvas");
  const dataUrl = canvas.toDataURL("image/png");

  const res = await fetch(`/api/project/${project}/input/mask/${encodeURIComponent(maskName)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_url: dataUrl })
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    setStatus(data.error || "Failed to save");
    return;
  }
  await refreshMaskList();
  setStatus(`Saved '${maskName}' (backup created if it existed).`);
}

async function suggestMask() {
  const text = ($("suggestText").value || "").trim();
  if (!text) {
    setStatus("Type what to mask first (e.g. 'left woman').");
    return;
  }
  const focusEl = $("suggestFocus");
  const focus = focusEl ? String(focusEl.value || "auto") : "auto";
  const featherEl = $("suggestFeather");
  const feather = featherEl ? parseFloat(featherEl.value || "0") : 0;
  setStatus(`Suggesting mask '${maskName}' for: ${text} (queueing...)`);
  $("btnSuggestMask").disabled = true;
  try {
    const res = await fetch(`/api/project/${project}/input/mask_suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mask_name: maskName, query: text, threshold: 0.30, focus, anchor, feather })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(data.error || "Mask suggestion failed.");
      return;
    }
    const jobId = data.job_id;
    if (!jobId) {
      setStatus("Mask suggestion started, but no job_id returned.");
      return;
    }
    setStatus(`Suggest job ${jobId} started…`);

    const started = Date.now();
    while (Date.now() - started < 10 * 60 * 1000) { // 10 minutes
      await new Promise(r => setTimeout(r, 1000));
      const stRes = await fetch(`/api/project/${project}/input/mask_suggest/${encodeURIComponent(jobId)}`, { cache: "no-store" });
      const st = await stRes.json().catch(() => ({}));
      if (!stRes.ok) {
        setStatus(st.error || "Mask suggestion status failed.");
        return;
      }
      const status = st.status || "running";
      const msg = st.message || status;
      const pid = st.prompt_id ? ` (prompt ${st.prompt_id})` : "";
      setStatus(`Suggest job ${jobId}${pid}: ${msg}`);
      if (status === "done") {
        await refreshMaskList();
        // Load the newly generated mask (but suppress the "Loaded existing mask" message)
        const canvas = $("maskCanvas");
        const ctx = canvas.getContext("2d");
        try {
          const res = await fetch(`/api/project/${project}/input/mask/${encodeURIComponent(maskName)}`);
          if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const img = new Image();
            await new Promise((resolve, reject) => {
              img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
                resolve();
              };
              img.onerror = reject;
              img.src = url;
            });
          }
        } catch (e) {
          // Non-fatal - mask might still be loading
        }
        setStatus(`Suggested mask saved to '${maskName}'. You can refine it with the brush.`);
        return;
      }
      if (status === "error") {
        setStatus(st.message || st.error || "Mask suggestion failed.");
        return;
      }
    }
    setStatus("Mask suggestion timed out (still running). Try again in a moment.");
  } catch (e) {
    setStatus(`Mask suggestion failed: ${e.message}`);
  } finally {
    $("btnSuggestMask").disabled = false;
  }
}

function createMask() {
  const raw = $("newMaskName").value;
  const n = validateMaskName(raw);
  if (!n) {
    setStatus("Invalid mask name. Use letters, numbers, _ or -.");
    return;
  }
  setMaskName(n);
  $("maskSelect").value = n;
  clearMask();
  setStatus(`Created new mask '${n}' (not saved yet).`);
}

function wire() {
  $("maskSelect").addEventListener("change", async (ev) => {
    const n = ev.target.value;
    setMaskName(n);
    clearMask();
    await loadAnchor();
    await loadExistingMask();
  });
  $("btnCreateMask").addEventListener("click", createMask);
  const suggestBtn = $("btnSuggestMask");
  if (suggestBtn) suggestBtn.addEventListener("click", suggestMask);
  $("toolBrush").addEventListener("click", () => setActiveTool("brush"));
  $("toolEraser").addEventListener("click", () => setActiveTool("eraser"));
  $("btnClear").addEventListener("click", clearMask);
  const fillBtn = $("btnFillWhite");
  if (fillBtn) fillBtn.addEventListener("click", fillWhite);
  const clearAnchorBtn = $("btnClearAnchor");
  if (clearAnchorBtn) clearAnchorBtn.addEventListener("click", clearAnchor);
  $("btnLoad").addEventListener("click", loadExistingMask);
  $("btnSave").addEventListener("click", saveMask);

  const canvas = $("maskCanvas");
  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  window.addEventListener("mouseup", stopDraw);

  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", moveDraw, { passive: false });
  canvas.addEventListener("touchend", stopDraw);
}

function init() {
  const img = $("baseImg");
  img.addEventListener("load", () => {
    resizeCanvasToImage();
    // Try load existing mask automatically (non-fatal).
    refreshMaskList().then(async () => {
      await loadAnchor();
      await loadExistingMask();
    });
  });
  wire();
  setActiveTool("brush");
}

init();

