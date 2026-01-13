#!/usr/bin/env python3
"""Iteration Viewer entrypoint.

The implementation lives under viewer/ii_viewer/. This file is intentionally thin.
"""

import os
import json
import threading
import base64
import shutil
import time
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, jsonify, send_from_directory, request, Response
import yaml
import re

try:
    from ii_viewer import create_app
    from ii_viewer.settings import PROJECTS_ROOT, _TRANSPARENT_PNG_1X1
    from ii_viewer.registry import LIVE_RUNS_LOCK as _LIVE_RUNS_LOCK, LIVE_RUNS as _LIVE_RUNS
    from ii_viewer.registry import MASK_JOBS_LOCK as _MASK_JOBS_LOCK, MASK_JOBS as _MASK_JOBS
except ModuleNotFoundError:
    from viewer.ii_viewer import create_app
    from viewer.ii_viewer.settings import PROJECTS_ROOT, _TRANSPARENT_PNG_1X1
    from viewer.ii_viewer.registry import LIVE_RUNS_LOCK as _LIVE_RUNS_LOCK, LIVE_RUNS as _LIVE_RUNS
    from viewer.ii_viewer.registry import MASK_JOBS_LOCK as _MASK_JOBS_LOCK, MASK_JOBS as _MASK_JOBS


app = create_app()


if __name__ == '__main__':
    # Default to stable behavior (no auto-reloader) so long-running requests don't get dropped.
    debug = os.environ.get("VIEWER_DEBUG", "").strip() == "1"
    app.run(debug=debug, use_reloader=debug, threaded=True, host='0.0.0.0', port=5000)


def _apply_feather_to_mask_png(raw_png: bytes, feather_pixels: float) -> bytes:
    """Apply a feathered (soft) border to a mask PNG using Gaussian blur.
    
    This helps the AI generation blend better at mask edges by creating a gradual
    transition from editable (white) to preserved (black) regions.
    
    Args:
        raw_png: Raw PNG bytes of the mask
        feather_pixels: Radius of feathering in pixels (typically 5-20)
    
    Returns:
        Feathered mask PNG bytes
    """
    if not raw_png or feather_pixels <= 0:
        return raw_png
    try:
        from PIL import Image, ImageFilter  # type: ignore
        import io

        img = Image.open(io.BytesIO(raw_png)).convert("L")
        
        # Apply Gaussian blur to create soft edges
        # The blur radius should be proportional to feather_pixels
        # PIL's GaussianBlur uses a radius parameter
        blurred = img.filter(ImageFilter.GaussianBlur(radius=feather_pixels))
        
        buf = io.BytesIO()
        blurred.save(buf, format="PNG")
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


@app.route('/api/project/<project_name>/config/aigen', methods=['GET'])
def get_aigen_yaml(project_name: str):
    """Get raw config/AIGen.yaml text."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIGen.yaml"
    if not path.exists():
        return jsonify({"error": "AIGen.yaml not found"}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route('/api/project/<project_name>/config/aigen', methods=['POST'])
def save_aigen_yaml(project_name: str):
    """Save raw config/AIGen.yaml text (validates YAML)."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing AIGen.yaml text"}), 400
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIGen.yaml"
    if not path.exists():
        return jsonify({"error": "AIGen.yaml not found"}), 404
    try:
        yaml.safe_load(text)
    except Exception as e:
        return jsonify({"error": f"YAML parse error: {e}"}), 400
    import time
    backup = path.with_name(f"AIGen.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True}), 200


@app.route('/api/project/<project_name>/config/aivis', methods=['GET'])
def get_aivis_yaml(project_name: str):
    """Get raw config/AIVis.yaml text."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIVis.yaml"
    if not path.exists():
        return jsonify({"error": "AIVis.yaml not found"}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route('/api/project/<project_name>/config/aivis', methods=['POST'])
def save_aivis_yaml(project_name: str):
    """Save raw config/AIVis.yaml text (validates YAML)."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing AIVis.yaml text"}), 400
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIVis.yaml"
    if not path.exists():
        return jsonify({"error": "AIVis.yaml not found"}), 404
    try:
        yaml.safe_load(text)
    except Exception as e:
        return jsonify({"error": f"YAML parse error: {e}"}), 400
    import time
    backup = path.with_name(f"AIVis.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True}), 200


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


def _ensure_prompts_model(rules_obj: dict) -> dict:
    """Ensure rules.yaml has a prompts section for storing base prompts.

    Schema:
      prompts:
        global:
          positive: ""
          negative: ""
        masks:
          left:
            positive: ""
            negative: ""
    """
    if not isinstance(rules_obj, dict):
        return rules_obj
    prompts = rules_obj.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {}
    g = prompts.get("global")
    if not isinstance(g, dict):
        g = {}
    g.setdefault("positive", "")
    g.setdefault("negative", "")
    m = prompts.get("masks")
    if not isinstance(m, dict):
        m = {}
    prompts["global"] = g
    prompts["masks"] = m
    rules_obj["prompts"] = prompts
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
    rules_obj = _ensure_prompts_model(rules_obj)
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
    rules_obj = _ensure_prompts_model(rules_obj)
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
    rules_obj = _ensure_prompts_model(rules_obj)
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


@app.route('/api/project/<project_name>/config/rules_generate_base_prompts', methods=['POST'])
def generate_rules_base_prompts(project_name: str):
    """Generate and persist base prompts into config/rules.yaml for a scope (global or mask name)."""
    payload = request.get_json(silent=True) or {}
    scope = str(payload.get("scope") or "all").strip() or "all"
    if scope == "all":
        scope = "global"

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

    # Normalise models
    masks = _list_project_masks(project_dir)
    rules_obj = _ensure_membership_model(rules_obj, masks)
    rules_obj = _ensure_prompts_model(rules_obj)

    # Determine active criteria for this scope.
    masking = rules_obj.get("masking") or {}
    mm = masking.get("masks") if isinstance(masking, dict) else []
    if not isinstance(mm, list):
        mm = []
    active_fields: list[str] = []
    known_masks: set[str] = set()
    for m in mm:
        if isinstance(m, dict) and m.get("name"):
            known_masks.add(str(m.get("name")))
    if scope == "global":
        # Global base prompts are for "no-mask / whole image" projects.
        # Use all criteria fields (in order), since there is no single mask scope to consult.
        crits_all = rules_obj.get("acceptance_criteria") or []
        if isinstance(crits_all, list):
            for c in crits_all:
                if isinstance(c, dict) and c.get("field"):
                    active_fields.append(str(c.get("field")))
    else:
        for m in mm:
            if isinstance(m, dict) and str(m.get("name") or "") == scope:
                ac = m.get("active_criteria") or []
                if isinstance(ac, list):
                    active_fields = [str(x) for x in ac if str(x).strip()]
                break
        # A mask can legitimately have zero active criteria; that is still a valid scope.
        # Validate scope by membership presence (or allow legacy default).
        if scope != "default" and scope not in known_masks:
            return jsonify({"error": f"Unknown mask scope: {scope}"}), 400

    crits = rules_obj.get("acceptance_criteria") or []
    if not isinstance(crits, list):
        crits = []
    by_field = {}
    for c in crits:
        if isinstance(c, dict) and c.get("field"):
            by_field[str(c.get("field"))] = c

    def _lines_for_crit(c: dict) -> list[str]:
        field = str(c.get("field") or "")
        intent = str(c.get("intent") or "")
        must = c.get("must_include") or []
        ban = c.get("ban_terms") or []
        avoid = c.get("avoid_terms") or []
        if not isinstance(must, list):
            must = []
        if not isinstance(ban, list):
            ban = []
        if not isinstance(avoid, list):
            avoid = []
        must_s = ", ".join([str(x).strip() for x in must if str(x).strip()])
        ban_s = ", ".join([str(x).strip() for x in ban if str(x).strip()])
        avoid_s = ", ".join([str(x).strip() for x in avoid if str(x).strip()])
        out = [f"- {field} (intent: {intent})"]
        if must_s:
            out.append(f"  must_include: {must_s}")
        if ban_s:
            out.append(f"  ban_terms: {ban_s}")
        if avoid_s:
            out.append(f"  avoid_terms: {avoid_s}")
        return out

    active_criteria_text = "\n".join(
        sum([_lines_for_crit(by_field[f]) for f in active_fields if f in by_field], [])
    ).strip()

    # Load / cache the original description based on the current input image.
    input_dir = project_dir / "input"
    img_path = input_dir / "progress.png"
    if not img_path.exists():
        img_path = input_dir / "input.png"
    if not img_path.exists():
        return jsonify({"error": "Input image not found (input/progress.png or input/input.png)"}), 404

    working_dir = project_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    desc_cache = working_dir / "original_description.txt"
    original_description = None
    try:
        if desc_cache.exists():
            original_description = desc_cache.read_text(encoding="utf-8").strip()
    except Exception:
        original_description = None

    # Build AIVis client (project override, else defaults)
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
    except Exception as e:
        return jsonify({"error": f"Failed to import AIVis client: {e}"}), 500

    aivis_cfg_path = project_dir / "config" / "AIVis.yaml"
    if not aivis_cfg_path.exists():
        aivis_cfg_path = _Path(__file__).parent.parent / "defaults" / "config" / "AIVis.yaml"
    try:
        aivis_cfg = yaml.safe_load(_Path(aivis_cfg_path).read_text(encoding="utf-8")) or {}
    except Exception:
        aivis_cfg = {}

    provider = str(aivis_cfg.get("provider") or "openrouter")
    model = str(aivis_cfg.get("model") or "qwen/qwen-2.5-vl-7b-instruct:free")
    fallback_provider = aivis_cfg.get("fallback_provider")
    fallback_model = aivis_cfg.get("fallback_model")
    api_key = aivis_cfg.get("api_key")

    prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
    try:
        client = AIVisClient(
            model=model,
            provider=provider,
            api_key=api_key,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            prompts_path=prompts_path,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to initialise AIVis client: {e}"}), 500

    if not original_description:
        original_description = client.describe_image(str(img_path))
        try:
            desc_cache.write_text(original_description, encoding="utf-8")
        except Exception:
            pass

    try:
        parsed, meta = client.generate_base_prompts(
            original_description=original_description,
            scope=scope,
            active_criteria_text=active_criteria_text,
        )
    except Exception as e:
        return jsonify({"error": f"Prompt generation failed: {e}"}), 500

    positive = str(parsed.get("positive") or "").strip()
    negative = str(parsed.get("negative") or "").strip()
    if not positive and not negative:
        return jsonify({"error": "AIVis returned empty prompts"}), 500

    # Filter contradictory terms: must_include should not be in negative, ban_terms should not be in positive
    def _filter_csv_by_terms(text: str, forbidden_terms: list) -> str:
        """Remove terms (and variants) from comma-separated text."""
        if not text or not forbidden_terms:
            return text
        parts = [p.strip() for p in text.split(",") if p.strip()]
        out = []
        for p in parts:
            pl = p.lower()
            # Check if any forbidden term (or variant) appears in this part
            should_skip = False
            for t in forbidden_terms:
                if not t:
                    continue
                tl = str(t).strip().lower()
                if tl in pl or pl in tl:  # Substring match to catch plurals/variants
                    should_skip = True
                    break
            if not should_skip:
                out.append(p)
        return ", ".join(out).strip(" ,\n")

    # Extract must_include and ban_terms from active criteria
    must_include_terms = []
    ban_terms = []
    avoid_terms = []
    for crit in active_criteria:
        if not isinstance(crit, dict):
            continue
        must_include_terms.extend(crit.get("must_include") or [])
        ban_terms.extend(crit.get("ban_terms") or [])
        avoid_terms.extend(crit.get("avoid_terms") or [])

    # Normalize terms (strip, lowercase for comparison)
    must_include_terms = [str(t).strip().lower() for t in must_include_terms if t]
    ban_terms = [str(t).strip().lower() for t in ban_terms if t]
    avoid_terms = [str(t).strip().lower() for t in avoid_terms if t]

    # Apply filters
    negative = _filter_csv_by_terms(negative, must_include_terms)
    positive = _filter_csv_by_terms(positive, ban_terms)
    positive = _filter_csv_by_terms(positive, avoid_terms)

    # Reorder positive prompt: put must_include terms from "change" intent criteria at the front
    # (Stable Diffusion gives more weight to terms earlier in the prompt)
    change_must_include = []
    preserve_must_include = []
    for crit in active_criteria:
        if not isinstance(crit, dict):
            continue
        intent = str(crit.get("intent") or "preserve").strip().lower()
        terms = crit.get("must_include") or []
        if intent == "change":
            change_must_include.extend(terms)
        else:
            preserve_must_include.extend(terms)
    
    if change_must_include:
        # Extract change terms from positive prompt and move to front
        pos_parts = [p.strip() for p in positive.split(",") if p.strip()]
        change_terms_found = []
        remaining_parts = []
        
        for part in pos_parts:
            part_lower = part.lower()
            is_change_term = False
            for ct in change_must_include:
                ct_lower = str(ct).strip().lower()
                if ct_lower in part_lower or part_lower in ct_lower:
                    change_terms_found.append(part)
                    is_change_term = True
                    break
            if not is_change_term:
                remaining_parts.append(part)
        
        # Reconstruct: change terms first, then rest
        if change_terms_found:
            positive = ", ".join(change_terms_found + remaining_parts)

    prompts = rules_obj.get("prompts") or {}
    if scope == "global":
        prompts["global"] = {"positive": positive, "negative": negative}
    else:
        masks_map = prompts.get("masks") if isinstance(prompts.get("masks"), dict) else {}
        masks_map[scope] = {"positive": positive, "negative": negative}
        prompts["masks"] = masks_map
    rules_obj["prompts"] = prompts

    rules_obj = _normalise_membership_model(rules_obj)
    rules_obj = _ensure_prompts_model(rules_obj)
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

    return jsonify({
        "ok": True,
        "scope": scope,
        "positive": positive,
        "negative": negative,
        "warnings": warnings,
        "aivis_metadata": meta,
    }), 200


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


if __name__ == '__main__':
    # Default to stable behavior (no auto-reloader) so long-running requests don't get dropped.
    debug = os.environ.get("VIEWER_DEBUG", "").strip() == "1"
    app.run(debug=debug, use_reloader=debug, threaded=True, host='0.0.0.0', port=5000)
