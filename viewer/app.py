#!/usr/bin/env python3
"""
Iteration Viewer - Web app to visualize iteration images and metadata
"""

import json
import threading
import base64
import os
import shutil
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, Response
import yaml
import re

app = Flask(__name__)
PROJECTS_ROOT = Path(__file__).parent.parent / "projects"

# 1x1 transparent PNG (avoids noisy 404s for missing mask previews in the browser UI)
_TRANSPARENT_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/az1eO8AAAAASUVORK5CYII="
)

# Live run registry (in-memory)
_LIVE_RUNS_LOCK = threading.Lock()
_LIVE_RUNS = {}  # (project_name, run_id) -> LiveRunController

# Mask suggestion jobs (in-memory)
_MASK_JOBS_LOCK = threading.Lock()
_MASK_JOBS = {}  # (project_name, job_id) -> dict


@app.route('/')
def index():
    """Main page listing all projects."""
    projects = []
    if PROJECTS_ROOT.exists():
        for project_dir in sorted(PROJECTS_ROOT.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
                continue
            working_dir = project_dir / "working"
            runs = []
            if working_dir.exists():
                for run_dir in sorted(working_dir.iterdir(), reverse=True):
                    if not run_dir.is_dir() or not run_dir.name[0].isdigit():
                        continue
                    metadata_dir = run_dir / "metadata"
                    if metadata_dir.exists():
                        iterations = sorted(
                            [f.stem.replace('iteration_', '').replace('_metadata', '') 
                             for f in metadata_dir.glob('iteration_*_metadata.json')],
                            key=lambda x: int(x) if x.isdigit() else 0
                        )
                        if iterations:
                            runs.append({
                                'id': run_dir.name,
                                'iterations': len(iterations),
                                'first_iteration': int(iterations[0]) if iterations else 0,
                                'last_iteration': int(iterations[-1]) if iterations else 0
                            })
            if runs:
                projects.append({
                    'name': project_dir.name,
                    'runs': runs
                })
    return render_template('index.html', projects=projects)


@app.route('/project/<project_name>/run/<run_id>')
def view_run(project_name: str, run_id: str):
    """View a specific run with all iterations."""
    return render_template('run.html', project_name=project_name, run_id=run_id)


@app.route('/project/<project_name>/live')
def live_run(project_name: str):
    """Live interactive run page (one iteration at a time)."""
    return render_template('live.html', project_name=project_name)


@app.route('/project/<project_name>/mask')
def mask_editor(project_name: str):
    """Mask editor page (supports multiple masks)."""
    mask_name = (request.args.get("mask") or "default").strip() or "default"
    return render_template('mask.html', project_name=project_name, mask_name=mask_name)


@app.route('/project/<project_name>/mask/<mask_name>')
def mask_editor_named(project_name: str, mask_name: str):
    """Mask editor page for a specific mask name."""
    mask_name = (mask_name or "default").strip() or "default"
    return render_template('mask.html', project_name=project_name, mask_name=mask_name)


@app.route('/project/<project_name>/rulesui')
def rules_ui(project_name: str):
    """Structured rules.yaml editor UI."""
    return render_template('rules_ui.html', project_name=project_name)


@app.route('/project/<project_name>/run/<run_id>/rank')
def rank_run(project_name: str, run_id: str):
    """Ranking page for a run (human feedback)."""
    return render_template('rank.html', project_name=project_name, run_id=run_id)


@app.route('/api/project/<project_name>/run/<run_id>/iterations')
def get_iterations(project_name: str, run_id: str):
    """Get all iteration metadata for a run."""
    run_dir = PROJECTS_ROOT / project_name / "working" / run_id
    metadata_dir = run_dir / "metadata"
    
    # For live runs, the browser may poll before any iterations have been written.
    # In that case, treat missing metadata_dir as "no iterations yet" rather than 404.
    if not metadata_dir.exists():
        if run_dir.exists():
            return jsonify({"iterations": []}), 200
        return jsonify({'error': 'Run not found'}), 404
    
    iterations = []
    for meta_file in sorted(metadata_dir.glob('iteration_*_metadata.json')):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract iteration number
                iter_num = int(meta_file.stem.replace('iteration_', '').replace('_metadata', ''))
                data['iteration_number'] = iter_num
                iterations.append(data)
        except Exception as e:
            continue
    
    # Sort by iteration number
    iterations.sort(key=lambda x: x.get('iteration_number', 0))
    
    return jsonify({'iterations': iterations})


@app.route('/api/project/<project_name>/run/<run_id>/image/<filename>')
def get_image(project_name: str, run_id: str, filename: str):
    """Serve iteration images."""
    images_dir = PROJECTS_ROOT / project_name / "working" / run_id / "images"
    if images_dir.exists() and (images_dir / filename).exists():
        return send_from_directory(str(images_dir), filename)
    return jsonify({'error': 'Image not found'}), 404


def _human_dir(project_name: str, run_id: str) -> Path:
    return PROJECTS_ROOT / project_name / "working" / run_id / "human"


@app.route('/api/project/<project_name>/run/<run_id>/human/ranking', methods=['GET'])
def get_human_ranking(project_name: str, run_id: str):
    """Get stored human ranking + notes for a run, if present."""
    path = _human_dir(project_name, run_id) / "ranking.json"
    if not path.exists():
        return jsonify({"ranking": [], "notes": {}, "updated_at": None}), 200
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    except Exception as e:
        return jsonify({"error": "Failed to read ranking.json"}), 500


@app.route('/api/project/<project_name>/run/<run_id>/human/ranking', methods=['POST'])
def save_human_ranking(project_name: str, run_id: str):
    """Save human ranking + notes for a run under working/<run_id>/human/."""
    payload = request.get_json(silent=True) or {}
    ranking = payload.get("ranking") or []
    notes = payload.get("notes") or {}

    if not isinstance(ranking, list):
        return jsonify({"error": "ranking must be a list"}), 400
    if not isinstance(notes, dict):
        return jsonify({"error": "notes must be a dict"}), 400

    # Normalise keys to strings for JSON stability
    ranking_norm = [str(x) for x in ranking]
    notes_norm = {str(k): str(v) for k, v in notes.items()}

    out = {
        "ranking": ranking_norm,
        "notes": notes_norm,
        "updated_at": __import__("time").time(),
    }

    human_dir = _human_dir(project_name, run_id)
    human_dir.mkdir(parents=True, exist_ok=True)
    path = human_dir / "ranking.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return jsonify({"ok": True}), 200


@app.route('/api/project/<project_name>/input/image')
def get_input_image(project_name: str):
    """Serve the original input image."""
    input_dir = PROJECTS_ROOT / project_name / "input"
    if (input_dir / "input.png").exists():
        return send_from_directory(str(input_dir), "input.png")
    return jsonify({'error': 'Input image not found'}), 404


@app.route('/api/project/<project_name>/input/mask')
def get_input_mask(project_name: str):
    """Serve the current mask.png (if present)."""
    input_dir = PROJECTS_ROOT / project_name / "input"
    if (input_dir / "mask.png").exists():
        return send_from_directory(str(input_dir), "mask.png")
    return Response(_TRANSPARENT_PNG_1X1, mimetype="image/png")


@app.route('/api/project/<project_name>/input/mask', methods=['POST'])
def save_input_mask(project_name: str):
    """Save mask.png from a data URL (PNG). Creates a timestamped backup if mask exists."""
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("data_url")
    if not isinstance(data_url, str) or "base64," not in data_url:
        return jsonify({"error": "Expected JSON {data_url: 'data:image/png;base64,...'}"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_path = input_dir / "mask.png"

    # Backup existing mask
    if mask_path.exists():
        import time

        backup = input_dir / f"mask.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        try:
            backup.write_bytes(mask_path.read_bytes())
        except Exception:
            pass

    try:
        b64 = data_url.split("base64,", 1)[1]
        raw = base64.b64decode(b64)
        mask_path.write_bytes(raw)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save mask: {e}"}), 500


@app.route('/api/project/<project_name>/input/masks')
def list_input_masks(project_name: str):
    """List available masks for a project.

    - Legacy: input/mask.png (name=default)
    - Multi-mask: input/masks/<name>.png (name=<name>)
    """
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    return jsonify({"masks": _list_project_masks(project_dir)}), 200


@app.route('/api/project/<project_name>/input/mask/<mask_name>')
def get_named_mask(project_name: str, mask_name: str):
    """Serve a named mask.

    - mask_name=default -> input/mask.png
    - otherwise -> input/masks/<mask_name>.png
    """
    try:
        project_dir = _safe_project_dir(project_name)
        mname = _safe_mask_name(mask_name)
    except Exception:
        return jsonify({"error": "Invalid project or mask"}), 400

    input_dir = project_dir / "input"
    if mname == "default":
        p = input_dir / "mask.png"
        if p.exists():
            return send_from_directory(str(input_dir), "mask.png")
        return Response(_TRANSPARENT_PNG_1X1, mimetype="image/png")

    masks_dir = input_dir / "masks"
    p = masks_dir / f"{mname}.png"
    if p.exists():
        return send_from_directory(str(masks_dir), p.name)
    return Response(_TRANSPARENT_PNG_1X1, mimetype="image/png")


@app.route('/api/project/<project_name>/input/mask/<mask_name>', methods=['POST'])
def save_named_mask(project_name: str, mask_name: str):
    """Save a named mask from a data URL (PNG).

    - mask_name=default -> input/mask.png
    - otherwise -> input/masks/<mask_name>.png
    Creates a timestamped backup if the target exists.
    """
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("data_url")
    if not isinstance(data_url, str) or "base64," not in data_url:
        return jsonify({"error": "Expected JSON {data_url: 'data:image/png;base64,...'}"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
        mname = _safe_mask_name(mask_name)
    except Exception:
        return jsonify({"error": "Invalid project or mask"}), 400

    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if mname == "default":
        target_dir = input_dir
        mask_path = input_dir / "mask.png"
    else:
        target_dir = input_dir / "masks"
        target_dir.mkdir(parents=True, exist_ok=True)
        mask_path = target_dir / f"{mname}.png"

    # Backup existing mask
    if mask_path.exists():
        import time
        backup = target_dir / f"{mask_path.stem}.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        try:
            backup.write_bytes(mask_path.read_bytes())
        except Exception:
            pass

    try:
        b64 = data_url.split("base64,", 1)[1]
        raw = base64.b64decode(b64)
        mask_path.write_bytes(raw)
        return jsonify({"ok": True, "name": mname}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save mask: {e}"}), 500


@app.route('/api/project/<project_name>/input/mask_suggest', methods=['POST'])
def suggest_mask(project_name: str):
    """Start a background mask-suggestion job via ComfyUI (GroundingDINO + SAM2).

    Payload:
      { "mask_name": "left", "query": "left woman", "threshold": 0.30 }

    Returns immediately:
      { ok: true, job_id: "...", status: "running" }
    """
    payload = request.get_json(silent=True) or {}
    mask_name = payload.get("mask_name") or payload.get("name") or "default"
    query = payload.get("query") or payload.get("text") or payload.get("prompt") or ""
    threshold = payload.get("threshold", 0.30)
    focus = payload.get("focus")  # auto|left|middle|right|none

    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Missing query text"}), 400
    try:
        threshold_f = float(threshold)
    except Exception:
        threshold_f = 0.30
    threshold_f = max(0.0, min(1.0, threshold_f))

    try:
        project_dir = _safe_project_dir(project_name)
        mname = _safe_mask_name(str(mask_name))
    except Exception:
        return jsonify({"error": "Invalid project or mask name"}), 400

    job_id = __import__("uuid").uuid4().hex
    key = (project_name, job_id)

    with _MASK_JOBS_LOCK:
        _MASK_JOBS[key] = {
            "job_id": job_id,
            "project": project_name,
            "mask_name": mname,
            "query": query.strip(),
            "threshold": threshold_f,
            "focus": str(focus) if focus is not None else "auto",
            "status": "queued",
            "message": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result_file": None,
            "error": None,
        }

    def _set(status: str, message: str | None = None, **extra):
        with _MASK_JOBS_LOCK:
            j = _MASK_JOBS.get(key)
            if not j:
                return
            j["status"] = status
            if message is not None:
                j["message"] = message
            j["updated_at"] = time.time()
            j.update(extra)

    def _work():
        _set("running", "starting")
        try:
            # Locate input image
            input_path = project_dir / "input" / "input.png"
            if not input_path.exists():
                _set("error", "input/input.png not found", error="missing input.png")
                return

            # Load ComfyUI host/port from project config
            aigen = _load_project_aigen(project_dir)
            comfy = aigen.get("comfyui") if isinstance(aigen, dict) else {}
            host = (comfy.get("host") if isinstance(comfy, dict) else None) or "localhost"
            port = int((comfy.get("port") if isinstance(comfy, dict) else None) or 8188)

            # Import ComfyUI client (in src/)
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
            from comfyui_client import ComfyUIClient  # noqa: E402

            client = ComfyUIClient(host=host, port=port)
            if not client.test_connection():
                _set("error", f"ComfyUI not reachable at {host}:{port}", error="comfyui unreachable")
                return

            comfyui_input_dir = _find_comfyui_input_dir()
            _set("running", "copying input image to ComfyUI...")
            comfy_input_filename = _copy_to_comfyui_input(project_name, input_path, comfyui_input_dir)

            prefix = f"iterative_imagination_{project_name}_mask_suggest_{mname}"
            workflow = {
                "1": {"class_type": "LoadImage", "inputs": {"image": comfy_input_filename}},
                "2": {"class_type": "GroundingDinoModelLoader (segment anything2)", "inputs": {"model_name": "GroundingDINO_SwinT_OGC (694MB)"}},
                "3": {"class_type": "SAM2ModelLoader (segment anything2)", "inputs": {"model_name": "sam2_hiera_small.pt"}},
                "4": {"class_type": "GroundingDinoSAM2Segment (segment anything2)", "inputs": {
                    "sam_model": ["3", 0],
                    "grounding_dino_model": ["2", 0],
                    "image": ["1", 0],
                    "prompt": query.strip(),
                    "threshold": threshold_f,
                }},
                "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 1]}},
                "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": prefix}},
            }

            _set("running", "queueing workflow to ComfyUI...")
            prompt_id = client.queue_prompt(workflow)
            _set("running", "running (this can take a while on first use)...", prompt_id=prompt_id)
            # NOTE: ComfyUI websocket progress isn't always reliable in some environments.
            # For mask suggestions, prefer polling history so the UI doesn't appear "stuck"
            # even if the websocket never reports completion.
            deadline = time.time() + 600
            while time.time() < deadline:
                try:
                    hist = client.get_history(prompt_id)
                    rec = hist.get(prompt_id) or {}
                    outputs = rec.get("outputs") or {}
                    status = (rec.get("status") or {}).get("status_str")
                    completed = (rec.get("status") or {}).get("completed")
                    if outputs or completed or status == "success":
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                _set("error", "Timed out waiting for ComfyUI mask generation", error="timeout")
                return

            _set("running", "downloading mask result...")
            hist = client.get_history(prompt_id)
            rec = hist.get(prompt_id) or {}
            outputs = rec.get("outputs") or {}
            out6 = outputs.get("6") or {}
            images = out6.get("images") or []
            if not images:
                _set("error", "ComfyUI returned no output image for the mask", error="no output")
                return
            img0 = images[0]
            filename = img0.get("filename")
            subfolder = img0.get("subfolder", "")
            ftype = img0.get("type", "output")
            if not filename:
                _set("error", "ComfyUI output missing filename", error="bad output")
                return
            raw = client.download_image(filename, subfolder=subfolder, folder_type=ftype)

            # Reduce accidental multi-person selection by keeping only a region if requested.
            focus_eff = str(focus).strip().lower() if isinstance(focus, str) else "auto"
            if focus_eff in ("auto", "", "none"):
                # Infer from mask name for common cases.
                mn = str(mname).lower()
                if mn in ("left", "middle", "right"):
                    focus_eff = mn
            raw = _apply_focus_to_mask_png(raw, focus_eff)

            # Save result into project
            input_dir = project_dir / "input"
            if mname == "default":
                out_path = input_dir / "mask.png"
            else:
                (input_dir / "masks").mkdir(parents=True, exist_ok=True)
                out_path = input_dir / "masks" / f"{mname}.png"
            out_path.write_bytes(raw)
            _set("done", "done", result_file=str(out_path.relative_to(project_dir)))
        except Exception as e:
            _set("error", f"Mask generation failed: {e}", error=str(e))

    threading.Thread(target=_work, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 202


@app.route('/api/project/<project_name>/input/mask_suggest/<job_id>')
def suggest_mask_status(project_name: str, job_id: str):
    """Get status of a mask suggestion job."""
    key = (project_name, str(job_id))
    with _MASK_JOBS_LOCK:
        j = _MASK_JOBS.get(key)
        if not j:
            return jsonify({"error": "Job not found"}), 404
        # shallow copy for safety
        out = dict(j)
    return jsonify({"ok": True, **out}), 200


@app.route('/api/project/<project_name>/input/mask_suggest')
def list_suggest_mask_jobs(project_name: str):
    """List recent mask suggestion jobs for a project (in-memory)."""
    try:
        _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    with _MASK_JOBS_LOCK:
        jobs = [dict(v) for (p, _jid), v in _MASK_JOBS.items() if p == project_name]
    jobs.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    # keep payload small
    slim = []
    for j in jobs[:25]:
        slim.append({
            "job_id": j.get("job_id"),
            "mask_name": j.get("mask_name"),
            "query": j.get("query"),
            "threshold": j.get("threshold"),
            "status": j.get("status"),
            "message": j.get("message"),
            "prompt_id": j.get("prompt_id"),
            "result_file": j.get("result_file"),
            "error": j.get("error"),
            "created_at": j.get("created_at"),
            "updated_at": j.get("updated_at"),
        })
    return jsonify({"ok": True, "jobs": slim}), 200


@app.route('/api/project/<project_name>/config')
def get_project_config(project_name: str):
    """Get project configuration (rules.yaml)."""
    rules_path = PROJECTS_ROOT / project_name / "config" / "rules.yaml"
    if rules_path.exists():
        with open(rules_path, 'r', encoding='utf-8') as f:
            return jsonify(yaml.safe_load(f))
    return jsonify({'error': 'Config not found'}), 404


def _safe_project_dir(project_name: str) -> Path:
    root = PROJECTS_ROOT.resolve()
    p = (PROJECTS_ROOT / project_name).resolve()
    if not str(p).startswith(str(root) + "/") and p != root:
        raise ValueError("Invalid project")
    return p


def _safe_mask_name(mask_name: str) -> str:
    """Sanitise mask names to avoid path traversal and weird filenames."""
    name = (mask_name or "").strip()
    if not name:
        return "default"
    if name == "default":
        return "default"
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        raise ValueError("Invalid mask name")
    return name


def _apply_focus_to_mask_png(raw_png: bytes, focus: str | None) -> bytes:
    """Post-process a mask PNG to keep only pixels within a horizontal focus region.

    focus:
      - left: keep x in [0, 0.45w]
      - middle: keep x in [0.28w, 0.72w]
      - right: keep x in [0.55w, w]
      - none/auto/unknown: no change
    """
    if not raw_png:
        return raw_png
    f = (focus or "").strip().lower()
    if f in ("", "auto", "none"):
        return raw_png
    try:
        from PIL import Image  # type: ignore
        import io

        img = Image.open(io.BytesIO(raw_png))
        g = img.convert("L")
        w, h = g.size
        if f == "left":
            x0, x1 = 0, int(w * 0.45)
        elif f == "middle":
            x0, x1 = int(w * 0.28), int(w * 0.72)
        elif f == "right":
            x0, x1 = int(w * 0.55), w
        else:
            return raw_png

        # Threshold to binary mask
        bw = g.point(lambda p: 255 if p >= 128 else 0, mode="L")
        out = Image.new("L", (w, h), 0)
        region = bw.crop((x0, 0, x1, h))
        out.paste(region, (x0, 0))

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raw_png


def _list_project_masks(project_dir: Path) -> list[dict]:
    """Return masks available for a project (default + named)."""
    input_dir = project_dir / "input"
    masks_dir = input_dir / "masks"
    masks: list[dict] = []
    if (input_dir / "mask.png").exists():
        masks.append({"name": "default", "file": "input/mask.png"})
    if masks_dir.exists():
        for p in sorted(masks_dir.glob("*.png")):
            try:
                safe = _safe_mask_name(p.stem)
            except Exception:
                continue
            masks.append({"name": safe, "file": f"input/masks/{p.name}"})
    return masks


def _find_comfyui_input_dir() -> Path:
    """Best-effort ComfyUI input directory discovery (mirrors ProjectManager.find_comfyui_input_dir)."""
    env_path = os.environ.get("COMFYUI_DIR")
    if env_path:
        comfyui_dir = Path(env_path).expanduser().resolve()
        if (comfyui_dir / "main.py").exists():
            return comfyui_dir / "input"

    # Common location for this setup
    p = Path.home() / "ComfyUI" / "input"
    if p.exists():
        return p

    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy_to_comfyui_input(project_name: str, input_image_path: Path, comfyui_input_dir: Path) -> str:
    """Copy a file into ComfyUI input dir with a unique name; return filename."""
    stamp = time.time_ns()
    stem = (input_image_path.stem or "input").replace(" ", "_")
    filename = f"iterative_imagination_{project_name}_{stem}_{stamp}.png"
    comfyui_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_image_path, comfyui_input_dir / filename)
    return filename


def _load_project_aigen(project_dir: Path) -> dict:
    """Load project config/working AIGen.yaml for ComfyUI host/port."""
    for p in [project_dir / "working" / "AIGen.yaml", project_dir / "config" / "AIGen.yaml"]:
        if p.exists():
            try:
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
    return {}


def _working_aigen_path(project_dir: Path) -> Path:
    return project_dir / "working" / "AIGen.yaml"


@app.route('/api/project/<project_name>/config/rules', methods=['GET'])
def get_rules_yaml(project_name: str):
    """Get raw rules.yaml text for editing."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404
    return Response(rules_path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route('/api/project/<project_name>/config/rules', methods=['POST'])
def save_rules_yaml(project_name: str):
    """Save raw rules.yaml text (validates YAML and lints rules)."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing rules text"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    # Parse YAML for validation
    try:
        rules_obj = yaml.safe_load(text) or {}
    except Exception as e:
        return jsonify({"error": f"YAML parse error: {e}"}), 400

    # Lint using existing rules_checker if available
    errors = []
    warnings = []
    try:
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from rules_checker import lint_rules  # noqa: E402

        errors, warnings = lint_rules(rules_obj)
    except Exception:
        # If linting cannot run, treat as no lint results (YAML parse already succeeded).
        errors, warnings = [], []

    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings}), 400

    # Backup existing file then write raw text (preserves comments/formatting from the editor).
    import time

    backup = rules_path.with_name(f"rules.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    rules_path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True, "warnings": warnings}), 200


def _lint_rules_obj(rules_obj: dict) -> tuple[list, list]:
    """Run rules_checker lint if available, else return no lints."""
    errors: list = []
    warnings: list = []
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from rules_checker import lint_rules  # noqa: E402
        errors, warnings = lint_rules(rules_obj)
    except Exception:
        errors, warnings = [], []
    return errors, warnings


def _render_rules_yaml(rules_obj: dict) -> str:
    """Render rules dict to YAML (stable-ish for UI; comments are not preserved)."""
    return yaml.safe_dump(rules_obj, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _ensure_membership_model(rules_obj: dict, masks: list[dict]) -> dict:
    """Ensure rules.yaml contains a mask->active_criteria membership model.

    Backwards compatible migration:
    - If rules_obj has no top-level `masking.masks`, derive membership from acceptance_criteria.applies_to_masks.
    - Criteria without applies_to_masks are treated as active in the `default` mask.

    The membership model is:
      masking:
        masks:
          - name: default
            active_criteria: [field1, field2, ...]
          - name: left
            active_criteria: [...]
    """
    if not isinstance(rules_obj, dict):
        return rules_obj

    masking = rules_obj.get("masking")
    if isinstance(masking, dict) and isinstance(masking.get("masks"), list) and masking.get("masks"):
        return rules_obj

    criteria = rules_obj.get("acceptance_criteria") or []
    if not isinstance(criteria, list):
        criteria = []

    # Available mask names from filesystem (always include default)
    mask_names = [m.get("name") for m in masks if isinstance(m, dict)]
    mask_names = [str(x) for x in mask_names if x]
    if "default" not in mask_names:
        mask_names.insert(0, "default")

    # Seed membership sets
    active_by_mask: dict[str, list[str]] = {n: [] for n in mask_names}

    def _add(mask_name: str, field: str):
        if mask_name not in active_by_mask:
            active_by_mask[mask_name] = []
        if field not in active_by_mask[mask_name]:
            active_by_mask[mask_name].append(field)

    for c in criteria:
        if not isinstance(c, dict):
            continue
        field = str(c.get("field") or "").strip()
        if not field:
            continue

        scopes = c.get("applies_to_masks") or c.get("mask_scope")
        if not scopes:
            _add("default", field)
            continue
        if isinstance(scopes, str):
            scopes_list = [scopes]
        elif isinstance(scopes, list):
            scopes_list = scopes
        else:
            scopes_list = [str(scopes)]
        scopes_norm = [str(x).strip() for x in scopes_list if str(x).strip()]
        for s in scopes_norm:
            _add(s, field)

    rules_obj["masking"] = {
        "masks": [{"name": name, "active_criteria": active_by_mask.get(name, [])} for name in mask_names]
    }
    return rules_obj


def _normalise_membership_model(rules_obj: dict) -> dict:
    """Normalise membership model and strip legacy per-criterion scoping fields."""
    if not isinstance(rules_obj, dict):
        return rules_obj
    masking = rules_obj.get("masking") or {}
    if not isinstance(masking, dict):
        return rules_obj
    masks = masking.get("masks")
    if not isinstance(masks, list):
        return rules_obj

    # Normalise mask entries
    seen = set()
    out = []
    for m in masks:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ac = m.get("active_criteria") or []
        if not isinstance(ac, list):
            ac = []
        ac_norm = []
        seen_f = set()
        for f in ac:
            ff = str(f).strip()
            if not ff or ff in seen_f:
                continue
            seen_f.add(ff)
            ac_norm.append(ff)
        out.append({"name": name, "active_criteria": ac_norm})
    rules_obj["masking"] = {"masks": out}

    # Strip legacy fields so there is only one source of truth.
    crits = rules_obj.get("acceptance_criteria")
    if isinstance(crits, list):
        for c in crits:
            if isinstance(c, dict):
                c.pop("applies_to_masks", None)
                c.pop("mask_scope", None)
    return rules_obj


@app.route('/api/project/<project_name>/config/rules_struct', methods=['GET'])
def get_rules_struct(project_name: str):
    """Get structured rules.yaml as JSON + mask list for UI."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    try:
        rules_obj = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        if not isinstance(rules_obj, dict):
            return jsonify({"error": "rules.yaml must contain a mapping at top level"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse rules.yaml: {e}"}), 400

    # Include masks so the UI can show valid scopes and build membership.
    masks = _list_project_masks(project_dir)
    rules_obj = _ensure_membership_model(rules_obj, masks)
    errors, warnings = _lint_rules_obj(rules_obj)
    yaml_text = _render_rules_yaml(rules_obj)
    return jsonify({"ok": True, "rules": rules_obj, "masks": masks, "yaml": yaml_text, "errors": errors, "warnings": warnings}), 200


@app.route('/api/project/<project_name>/config/rules_render', methods=['POST'])
def render_rules_struct(project_name: str):
    """Render and lint rules dict without saving (for YAML preview)."""
    payload = request.get_json(silent=True) or {}
    rules_obj = payload.get("rules")
    if not isinstance(rules_obj, dict):
        return jsonify({"error": "Expected JSON {rules: {...}}"}), 400

    rules_obj = _normalise_membership_model(rules_obj)
    errors, warnings = _lint_rules_obj(rules_obj)
    yaml_text = _render_rules_yaml(rules_obj)
    return jsonify({"ok": True, "yaml": yaml_text, "errors": errors, "warnings": warnings}), 200


@app.route('/api/project/<project_name>/config/rules_struct', methods=['POST'])
def save_rules_struct(project_name: str):
    """Save rules.yaml from a structured dict (creates backup)."""
    payload = request.get_json(silent=True) or {}
    rules_obj = payload.get("rules")
    if not isinstance(rules_obj, dict):
        return jsonify({"error": "Expected JSON {rules: {...}}"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    rules_obj = _normalise_membership_model(rules_obj)
    errors, warnings = _lint_rules_obj(rules_obj)
    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings, "yaml": _render_rules_yaml(rules_obj)}), 400

    import time
    backup = rules_path.with_name(f"rules.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    rules_path.write_text(_render_rules_yaml(rules_obj), encoding="utf-8")
    return jsonify({"ok": True, "warnings": warnings}), 200


@app.route('/api/project/<project_name>/working/aigen/prompts', methods=['GET'])
def get_working_aigen_prompts(project_name: str):
    """Get current prompts from working/AIGen.yaml (positive/negative)."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    path = _working_aigen_path(project_dir)
    if not path.exists():
        # If not present, fall back to config/AIGen.yaml so UI still works.
        cfg = project_dir / "config" / "AIGen.yaml"
        if cfg.exists():
            path = cfg
        else:
            return jsonify({"error": "AIGen.yaml not found"}), 404

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        prompts = data.get("prompts") or {}
        return jsonify({
            "positive": str(prompts.get("positive") or ""),
            "negative": str(prompts.get("negative") or ""),
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read AIGen.yaml: {e}"}), 500


@app.route('/api/project/<project_name>/working/aigen/prompts', methods=['POST'])
def save_working_aigen_prompts(project_name: str):
    """Save prompts into working/AIGen.yaml (creates timestamped backup)."""
    payload = request.get_json(silent=True) or {}
    positive = payload.get("positive", "")
    negative = payload.get("negative", "")
    if not isinstance(positive, str) or not isinstance(negative, str):
        return jsonify({"error": "positive and negative must be strings"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    working_dir = project_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    path = _working_aigen_path(project_dir)

    # Ensure working/AIGen.yaml exists by copying config if needed.
    if not path.exists():
        cfg = project_dir / "config" / "AIGen.yaml"
        if cfg.exists():
            path.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Minimal fallback
            path.write_text(yaml.safe_dump({"prompts": {"positive": "", "negative": ""}}, sort_keys=False), encoding="utf-8")

    # Backup existing working file
    import time

    backup = working_dir / f"AIGen.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("prompts", {})
        if not isinstance(data["prompts"], dict):
            data["prompts"] = {}
        data["prompts"]["positive"] = positive
        data["prompts"]["negative"] = negative
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save AIGen.yaml: {e}"}), 500


def _make_run_id() -> str:
    import time

    return time.strftime("%Y-%m-%d_%H-%M-%S")


@app.route('/api/project/<project_name>/live/start', methods=['POST'])
def live_start(project_name: str):
    """Start a live run for a project, returning a run_id."""
    payload = request.get_json(silent=True) or {}
    max_iterations = int(payload.get("max_iterations") or 20)
    reset = bool(payload.get("reset"))

    # Import here to avoid slowing initial page load.
    from live_runner import LiveRunController

    # Optionally reset project working state before creating the new run folder.
    if reset:
        try:
            import sys
            from pathlib import Path as _Path

            sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
            from project_manager import ProjectManager  # noqa: E402
            # Best-effort reset: remove checkpoint so a fresh loop is forced.
            ckpt = ProjectManager(project_name).get_checkpoint_path()
            if ckpt.exists():
                ckpt.unlink()
        except Exception:
            pass

    run_id = _make_run_id()

    # Create run directories immediately so the UI can poll without 404s.
    run_root = PROJECTS_ROOT / project_name / "working" / run_id
    for sub in ("images", "questions", "evaluation", "comparison", "metadata", "human"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    if reset:
        # Best-effort reset: remove checkpoint so a fresh loop is forced.
        ckpt = PROJECTS_ROOT / project_name / "working" / "checkpoint.json"
        if ckpt.exists():
            try:
                ckpt.unlink()
            except Exception:
                pass

    ctl = LiveRunController(project=project_name, run_id=run_id, max_iterations=max_iterations)
    with _LIVE_RUNS_LOCK:
        _LIVE_RUNS[(project_name, run_id)] = ctl
    ctl.start()
    return jsonify({"ok": True, "run_id": run_id}), 200


@app.route('/api/project/<project_name>/live/<run_id>/state')
def live_state(project_name: str, run_id: str):
    """Get live run controller state."""
    with _LIVE_RUNS_LOCK:
        ctl = _LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404
    return jsonify({"state": ctl.state()}), 200


@app.route('/api/project/<project_name>/live/<run_id>/feedback', methods=['POST'])
def live_feedback(project_name: str, run_id: str):
    """Submit feedback (comment + nudges) for the current iteration and resume."""
    payload = request.get_json(silent=True) or {}
    iteration = int(payload.get("iteration") or 0)
    comment = str(payload.get("comment") or "")
    nudge = payload.get("nudge") or {}
    if not isinstance(nudge, dict):
        nudge = {}

    with _LIVE_RUNS_LOCK:
        ctl = _LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404

    ctl.submit_feedback(iteration=iteration, comment=comment, nudge=nudge)
    return jsonify({"ok": True}), 200


if __name__ == '__main__':
    # Default to stable behavior (no auto-reloader) so long-running requests don't get dropped.
    debug = os.environ.get("VIEWER_DEBUG", "").strip() == "1"
    app.run(debug=debug, use_reloader=debug, threaded=True, host='0.0.0.0', port=5000)
