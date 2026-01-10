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
  $("toolBrush").classList.toggle("active", tool === "brush");
  $("toolEraser").classList.toggle("active", tool === "eraser");
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
    await loadExistingMask();
  });
  $("btnCreateMask").addEventListener("click", createMask);
  $("toolBrush").addEventListener("click", () => setActiveTool("brush"));
  $("toolEraser").addEventListener("click", () => setActiveTool("eraser"));
  $("btnClear").addEventListener("click", clearMask);
  const fillBtn = $("btnFillWhite");
  if (fillBtn) fillBtn.addEventListener("click", fillWhite);
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
    refreshMaskList().then(loadExistingMask);
  });
  wire();
  setActiveTool("brush");
}

init();

