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
from typing import List, Dict
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


@app.route('/api/project/<project_name>/input/describe')
def describe_input(project_name: str):
    """Get AIVis description of the input image (legacy - returns text)."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    input_path = project_dir / "input" / "input.png"
    if not input_path.exists():
        return jsonify({"error": "Input image not found"}), 404
    
    # Load AIVis config and initialize client (same as describe_iteration)
    try:
        aivis_config_path = project_dir / "config" / "AIVis.yaml"
        if not aivis_config_path.exists():
            defaults_dir = Path(__file__).parent.parent / "defaults" / "config"
            aivis_config_path = defaults_dir / "AIVis.yaml"
        
        with open(aivis_config_path, 'r', encoding='utf-8') as f:
            aivis_config = yaml.safe_load(f) or {}
    except Exception as e:
        return jsonify({"error": f"Failed to load AIVis config: {e}"}), 500
    
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
        
        prompts_path = project_dir / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
        
        provider = str(aivis_config.get("provider") or "ollama")
        model = str(aivis_config.get("model") or "qwen3-vl:4b")
        api_key = aivis_config.get("api_key")
        fallback_provider = aivis_config.get("fallback_provider")
        fallback_model = aivis_config.get("fallback_model")
        max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
        base_url = aivis_config.get("base_url")
        
        aivis = AIVisClient(
            model=model, provider=provider, api_key=api_key,
            fallback_provider=fallback_provider, fallback_model=fallback_model,
            prompts_path=prompts_path, max_concurrent=max_concurrent, base_url=base_url,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500
    
    try:
        description = aivis.describe_image(str(input_path))
        return jsonify({"ok": True, "description": description}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500


@app.route('/api/project/<project_name>/input/describe_terms')
def describe_input_terms(project_name: str):
    """Get AIVis description of the input image as structured term lists."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    input_path = project_dir / "input" / "input.png"
    if not input_path.exists():
        return jsonify({"error": "Input image not found"}), 404
    
    # Load AIVis config and initialize client
    try:
        aivis_config_path = project_dir / "config" / "AIVis.yaml"
        if not aivis_config_path.exists():
            defaults_dir = Path(__file__).parent.parent / "defaults" / "config"
            aivis_config_path = defaults_dir / "AIVis.yaml"
        
        with open(aivis_config_path, 'r', encoding='utf-8') as f:
            aivis_config = yaml.safe_load(f) or {}
    except Exception as e:
        return jsonify({"error": f"Failed to load AIVis config: {e}"}), 500
    
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
        
        prompts_path = project_dir / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
        
        provider = str(aivis_config.get("provider") or "ollama")
        model = str(aivis_config.get("model") or "qwen3-vl:4b")
        api_key = aivis_config.get("api_key")
        fallback_provider = aivis_config.get("fallback_provider")
        fallback_model = aivis_config.get("fallback_model")
        max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
        base_url = aivis_config.get("base_url")
        
        aivis = AIVisClient(
            model=model, provider=provider, api_key=api_key,
            fallback_provider=fallback_provider, fallback_model=fallback_model,
            prompts_path=prompts_path, max_concurrent=max_concurrent, base_url=base_url,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500
    
    try:
        terms = aivis.describe_image_as_terms(str(input_path))
        return jsonify({"ok": True, **terms}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image as terms: {e}"}), 500


@app.route('/api/project/<project_name>/mask/<mask_name>/prompts')
def get_mask_prompts(project_name: str, mask_name: str):
    """Get prompts for a specific mask (from rules.yaml)."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404
    
    try:
        rules_obj = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        prompts = rules_obj.get("prompts") or {}
        masks_prompts = prompts.get("masks") or {}
        global_prompts = prompts.get("global") or {}
        
        # Get mask-specific prompts, fallback to global
        if mask_name and mask_name != "global" and isinstance(masks_prompts, dict) and mask_name in masks_prompts:
            mask_p = masks_prompts[mask_name] or {}
            return jsonify({
                "positive": str(mask_p.get("positive") or ""),
                "negative": str(mask_p.get("negative") or "")
            }), 200
        else:
            return jsonify({
                "positive": str(global_prompts.get("positive") or ""),
                "negative": str(global_prompts.get("negative") or "")
            }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load prompts: {e}"}), 500


@app.route('/api/project/<project_name>/mask/<mask_name>/terms')
def get_mask_terms(project_name: str, mask_name: str):
    """Get terms (must_include, ban_terms, avoid_terms) for a specific mask."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404
    
    try:
        rules_obj = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        masking = rules_obj.get("masking") or {}
        masks = masking.get("masks") or []
        
        # Find active criteria for this mask
        active_fields = []
        if mask_name == "global":
            # For global, use all criteria
            criteria_all = rules_obj.get("acceptance_criteria") or []
            if isinstance(criteria_all, list):
                for c in criteria_all:
                    if isinstance(c, dict) and c.get("field"):
                        active_fields.append(str(c.get("field")))
        else:
            # For specific mask, use its active_criteria
            for m in masks:
                if isinstance(m, dict) and str(m.get("name") or "") == mask_name:
                    ac = m.get("active_criteria") or []
                    if isinstance(ac, list):
                        active_fields = [str(x).strip() for x in ac if str(x).strip()]
                    break
        
        # Get criteria and extract terms
        criteria = rules_obj.get("acceptance_criteria") or []
        must_include = []
        ban_terms = []
        avoid_terms = []
        
        for crit in criteria:
            if not isinstance(crit, dict):
                continue
            field = str(crit.get("field") or "")
            if field not in active_fields:
                continue
            
            must_include.extend(crit.get("must_include") or [])
            ban_terms.extend(crit.get("ban_terms") or [])
            avoid_terms.extend(crit.get("avoid_terms") or [])
        
        # Remove duplicates while preserving order
        def dedupe(lst):
            seen = set()
            result = []
            for item in lst:
                item_str = str(item).strip()
                if item_str and item_str.lower() not in seen:
                    seen.add(item_str.lower())
                    result.append(item_str)
            return result
        
        return jsonify({
            "must_include": dedupe(must_include),
            "ban_terms": dedupe(ban_terms),
            "avoid_terms": dedupe(avoid_terms)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load terms: {e}"}), 500


@app.route('/api/project/<project_name>/working/aigen/settings', methods=['GET'])
def get_aigen_settings(project_name: str):
    """Get current AIGen settings (denoise, cfg, scheduler, model, etc.)."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    aigen_path = project_dir / "working" / "AIGen.yaml"
    if not aigen_path.exists():
        aigen_path = project_dir / "config" / "AIGen.yaml"
    
    if not aigen_path.exists():
        return jsonify({"error": "AIGen.yaml not found"}), 404
    
    try:
        aigen = yaml.safe_load(aigen_path.read_text(encoding="utf-8")) or {}
        params = aigen.get("parameters") or {}
        model_info = aigen.get("model") or {}
        
        return jsonify({
            "denoise": params.get("denoise", 0.5),
            "cfg": params.get("cfg", 7.0),
            "steps": params.get("steps", 25),
            "sampler_name": params.get("sampler_name", "dpmpp_2m_sde_gpu"),
            "scheduler": params.get("scheduler", "karras"),
            "checkpoint": model_info.get("ckpt_name", "unknown"),
            "workflow": aigen.get("workflow_file", "unknown")
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to load settings: {e}"}), 500


@app.route('/api/project/<project_name>/working/aigen/settings', methods=['POST'])
def save_aigen_settings(project_name: str):
    """Save AIGen settings to working/AIGen.yaml (creates timestamped backup)."""
    payload = request.get_json(silent=True) or {}
    
    denoise = payload.get("denoise")
    cfg = payload.get("cfg")
    steps = payload.get("steps")
    sampler_name = payload.get("sampler_name")
    scheduler = payload.get("scheduler")
    
    if denoise is None or cfg is None or steps is None or not sampler_name or not scheduler:
        return jsonify({"error": "Missing required fields: denoise, cfg, steps, sampler_name, scheduler"}), 400
    
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    working_dir = project_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    path = _working_aigen_path(project_dir)
    
    # Ensure working/AIGen.yaml exists by copying config if needed
    if not path.exists():
        cfg = project_dir / "config" / "AIGen.yaml"
        if cfg.exists():
            path.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Minimal fallback
            path.write_text(yaml.safe_dump({
                "parameters": {
                    "denoise": 0.5,
                    "cfg": 7.0,
                    "steps": 25,
                    "sampler_name": "dpmpp_2m_sde_gpu",
                    "scheduler": "karras"
                }
            }, sort_keys=False), encoding="utf-8")
    
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
        
        # Ensure parameters section exists
        data.setdefault("parameters", {})
        if not isinstance(data["parameters"], dict):
            data["parameters"] = {}
        
        # Update parameters
        data["parameters"]["denoise"] = float(denoise)
        data["parameters"]["cfg"] = float(cfg)
        data["parameters"]["steps"] = int(steps)
        data["parameters"]["sampler_name"] = str(sampler_name)
        data["parameters"]["scheduler"] = str(scheduler)
        
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save AIGen.yaml: {e}"}), 500


def _parse_prompt_to_terms(prompt_str: str) -> List[str]:
    """Parse a comma-separated prompt string into a list of terms."""
    if not prompt_str:
        return []
    # Split by comma, strip whitespace, filter empty
    terms = [t.strip() for t in prompt_str.split(",") if t.strip()]
    return terms

def _compare_term_lists(current: List[str], suggested: List[str]) -> Dict:
    """Compare two term lists and return suggestions (add/remove).
    
    Returns: {
        "to_add": [terms in suggested but not in current],
        "to_remove": [terms in current but not in suggested],
        "unchanged": [terms in both]
    }
    """
    current_lower = {t.lower(): t for t in current}  # Preserve original case
    suggested_lower = {t.lower(): t for t in suggested}
    
    to_add = [suggested_lower[t] for t in suggested_lower if t not in current_lower]
    to_remove = [current_lower[t] for t in current_lower if t not in suggested_lower]
    unchanged = [current_lower[t] for t in current_lower if t in suggested_lower]
    
    return {
        "to_add": to_add,
        "to_remove": to_remove,
        "unchanged": unchanged
    }


@app.route('/api/project/<project_name>/run/<run_id>/suggest_prompts')
def suggest_prompts(project_name: str, run_id: str):
    """Get AIVis suggestions for improving prompts based on latest iteration.
    
    Query parameters:
    - mask_name: Optional mask name to get mask-specific suggestions
    
    Returns structured term lists with suggestions (to_add, to_remove).
    """
    # Get mask name from query parameter
    mask_name = request.args.get("mask_name", "global").strip() or "global"
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    # Get latest iteration
    run_dir = project_dir / "working" / run_id
    metadata_dir = run_dir / "metadata"
    if not metadata_dir.exists():
        return jsonify({"error": "No iterations found"}), 404
    
    # Find latest iteration
    meta_files = sorted(metadata_dir.glob('iteration_*_metadata.json'))
    if not meta_files:
        return jsonify({"error": "No iterations found"}), 404
    
    latest_meta = meta_files[-1]
    with open(latest_meta, 'r', encoding='utf-8') as f:
        iteration_data = json.load(f)
    
    iteration_num = iteration_data.get('iteration', 1)
    image_path = run_dir / "images" / f"iteration_{iteration_num}.png"
    if not image_path.exists():
        return jsonify({"error": "Iteration image not found"}), 404
    
    # Get current prompts as term lists
    current_positive_str = iteration_data.get('prompts_used', {}).get('positive', '')
    current_negative_str = iteration_data.get('prompts_used', {}).get('negative', '')
    current_positive_terms = _parse_prompt_to_terms(current_positive_str)
    current_negative_terms = _parse_prompt_to_terms(current_negative_str)
    
    evaluation = iteration_data.get('evaluation', {})
    comparison = iteration_data.get('comparison', {})
    
    # Initialize AIVis
    try:
        aivis_config_path = project_dir / "config" / "AIVis.yaml"
        if not aivis_config_path.exists():
            defaults_dir = Path(__file__).parent.parent / "defaults" / "config"
            aivis_config_path = defaults_dir / "AIVis.yaml"
        
        with open(aivis_config_path, 'r', encoding='utf-8') as f:
            aivis_config = yaml.safe_load(f) or {}
    except Exception as e:
        return jsonify({"error": f"Failed to load AIVis config: {e}"}), 500
    
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
        from prompt_improver import PromptImprover  # noqa: E402
        
        prompts_path = project_dir / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
        
        provider = str(aivis_config.get("provider") or "ollama")
        model = str(aivis_config.get("model") or "qwen3-vl:4b")
        api_key = aivis_config.get("api_key")
        fallback_provider = aivis_config.get("fallback_provider")
        fallback_model = aivis_config.get("fallback_model")
        max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
        base_url = aivis_config.get("base_url")
        
        aivis = AIVisClient(
            model=model, provider=provider, api_key=api_key,
            fallback_provider=fallback_provider, fallback_model=fallback_model,
            prompts_path=prompts_path, max_concurrent=max_concurrent, base_url=base_url,
        )
        
        # Get original description (for grounding/context)
        input_path = project_dir / "input" / "input.png"
        original_desc = aivis.describe_image(str(input_path)) if input_path.exists() else None
        
        # Get latest iteration description (this is what we're actually analyzing)
        latest_iteration_desc = None
        try:
            latest_iteration_desc = aivis.describe_image(str(image_path))
        except Exception as e:
            # If describing latest iteration fails, continue without it
            pass
        
        # Use prompt improver
        import logging
        logger = logging.getLogger(__name__)
        improver = PromptImprover(logger, aivis, lambda: original_desc, "")
        
        # Get criteria for context
        rules_path = project_dir / "config" / "rules.yaml"
        rules_obj = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {} if rules_path.exists() else {}
        criteria_defs = rules_obj.get("acceptance_criteria", []) or []
        criteria_by_field = {c.get('field'): c for c in criteria_defs if isinstance(c, dict) and c.get('field')}
        
        failed_criteria = []
        criteria_results = evaluation.get('criteria_results', {}) or {}
        for field, result in criteria_results.items():
            crit = criteria_by_field.get(field)
            if not crit:
                continue
            ctype = (crit.get('type') or 'boolean').lower()
            min_v = crit.get('min', 0)
            if ctype == 'boolean':
                if result is not True:
                    failed_criteria.append(field)
            else:
                if isinstance(result, (int, float)) and result < min_v:
                    failed_criteria.append(field)
        
        # Use latest iteration description if available (more accurate than original)
        description_for_improvement = latest_iteration_desc or original_desc
        
        improved_positive, improved_negative, diff_info = improver.improve_prompts(
            current_positive=current_positive_str,
            current_negative=current_negative_str,
            evaluation=evaluation,
            comparison=comparison,
            failed_criteria=failed_criteria,
            criteria_defs=criteria_defs,
            criteria_by_field=criteria_by_field,
            original_description=description_for_improvement,
        )
        
        # Parse improved prompts into term lists
        suggested_positive_terms = _parse_prompt_to_terms(improved_positive)
        suggested_negative_terms = _parse_prompt_to_terms(improved_negative)
        
        # Compare to get suggestions
        positive_suggestions = _compare_term_lists(current_positive_terms, suggested_positive_terms)
        negative_suggestions = _compare_term_lists(current_negative_terms, suggested_negative_terms)
        
        return jsonify({
            "ok": True,
            "current_positive_terms": current_positive_terms,
            "current_negative_terms": current_negative_terms,
            "suggested_positive_terms": suggested_positive_terms,
            "suggested_negative_terms": suggested_negative_terms,
            "positive_suggestions": positive_suggestions,
            "negative_suggestions": negative_suggestions,
            "diff_info": diff_info,
            "latest_iteration_description": latest_iteration_desc,  # Add the description
            "iteration_number": iteration_num
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate suggestions: {e}"}), 500


@app.route('/api/project/<project_name>/run/<run_id>/describe_terms/<int:iteration_number>')
def describe_iteration_terms(project_name: str, run_id: str, iteration_number: int):
    """Get AIVis description of an iteration image as structured term lists."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    run_dir = project_dir / "working" / run_id
    image_path = run_dir / "images" / f"iteration_{iteration_number}.png"
    
    if not image_path.exists():
        return jsonify({"error": f"Iteration {iteration_number} image not found"}), 404
    
    # Load AIVis config
    try:
        aivis_config_path = project_dir / "config" / "AIVis.yaml"
        if not aivis_config_path.exists():
            defaults_dir = Path(__file__).parent.parent / "defaults" / "config"
            aivis_config_path = defaults_dir / "AIVis.yaml"
        
        with open(aivis_config_path, 'r', encoding='utf-8') as f:
            aivis_config = yaml.safe_load(f) or {}
    except Exception as e:
        return jsonify({"error": f"Failed to load AIVis config: {e}"}), 500
    
    # Initialize AIVis client
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
        
        prompts_path = project_dir / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
        
        provider = str(aivis_config.get("provider") or "ollama")
        model = str(aivis_config.get("model") or "qwen3-vl:4b")
        api_key = aivis_config.get("api_key")
        fallback_provider = aivis_config.get("fallback_provider")
        fallback_model = aivis_config.get("fallback_model")
        max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
        base_url = aivis_config.get("base_url")
        
        aivis = AIVisClient(
            model=model, provider=provider, api_key=api_key,
            fallback_provider=fallback_provider, fallback_model=fallback_model,
            prompts_path=prompts_path, max_concurrent=max_concurrent, base_url=base_url,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500
    
    # Get description as terms
    try:
        terms = aivis.describe_image_as_terms(str(image_path))
        return jsonify({
            "ok": True,
            "iteration": iteration_number,
            **terms,
            "metadata": getattr(aivis, '_last_request_metadata', {})
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500


@app.route('/api/project/<project_name>/run/<run_id>/describe/<int:iteration_number>')
def describe_iteration(project_name: str, run_id: str, iteration_number: int):
    """Get AIVis description of an iteration image."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    
    run_dir = project_dir / "working" / run_id
    image_path = run_dir / "images" / f"iteration_{iteration_number}.png"
    
    if not image_path.exists():
        return jsonify({"error": f"Iteration {iteration_number} image not found"}), 404
    
    # Load AIVis config
    try:
        aivis_config_path = project_dir / "config" / "AIVis.yaml"
        if not aivis_config_path.exists():
            # Fallback to defaults
            defaults_dir = Path(__file__).parent.parent / "defaults" / "config"
            aivis_config_path = defaults_dir / "AIVis.yaml"
        
        with open(aivis_config_path, 'r', encoding='utf-8') as f:
            aivis_config = yaml.safe_load(f) or {}
    except Exception as e:
        return jsonify({"error": f"Failed to load AIVis config: {e}"}), 500
    
    # Initialize AIVis client
    try:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from aivis_client import AIVisClient  # noqa: E402
        
        # Load prompts path (project-specific or defaults)
        prompts_path = project_dir / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = _Path(__file__).parent.parent / "defaults" / "prompts.yaml"
        
        # Extract config values
        provider = str(aivis_config.get("provider") or "ollama")
        model = str(aivis_config.get("model") or "qwen3-vl:4b")
        api_key = aivis_config.get("api_key")
        fallback_provider = aivis_config.get("fallback_provider")
        fallback_model = aivis_config.get("fallback_model")
        max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
        base_url = aivis_config.get("base_url")  # For Ollama
        
        # Initialize client
        aivis = AIVisClient(
            model=model,
            provider=provider,
            api_key=api_key,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            prompts_path=prompts_path,
            max_concurrent=max_concurrent,
            base_url=base_url,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500
    
    # Get description
    try:
        description = aivis.describe_image(str(image_path))
        return jsonify({
            "ok": True,
            "iteration": iteration_number,
            "description": description,
            "metadata": getattr(aivis, '_last_request_metadata', {})
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500


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
    anchor = payload.get("anchor")
    feather = payload.get("feather", 0)  # pixels to feather (0 = no feathering)

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
            "anchor": anchor if isinstance(anchor, dict) else None,
            "feather": float(feather) if feather is not None else 0.0,
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

            # If an anchor is provided (or saved), keep only the component that matches it.
            anchor_eff = anchor if isinstance(anchor, dict) else None
            if anchor_eff is None:
                try:
                    ap = _anchor_path(project_dir, mname)
                    if ap.exists():
                        anchor_eff = (json.loads(ap.read_text(encoding="utf-8")) or {}).get("anchor")
                except Exception:
                    anchor_eff = None
            if isinstance(anchor_eff, dict):
                raw = _apply_anchor_component_to_mask_png(raw, anchor_eff)

            # Apply feathering if requested (softens mask edges for better blending)
            feather_px = float(feather) if feather is not None else 0.0
            if feather_px > 0:
                raw = _apply_feather_to_mask_png(raw, feather_px)

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


def _anchor_path(project_dir: Path, mask_name: str) -> Path:
    """Path to store anchor metadata for a mask."""
    input_dir = project_dir / "input"
    if mask_name == "default":
        return input_dir / "mask.anchor.json"
    return input_dir / "masks" / f"{mask_name}.anchor.json"


@app.route('/api/project/<project_name>/input/mask_anchor/<mask_name>', methods=['GET'])
def get_mask_anchor(project_name: str, mask_name: str):
    """Get the saved anchor point for a mask (if any)."""
    try:
        project_dir = _safe_project_dir(project_name)
        mname = _safe_mask_name(mask_name)
    except Exception:
        return jsonify({"error": "Invalid project or mask"}), 400

    p = _anchor_path(project_dir, mname)
    if not p.exists():
        return jsonify({"ok": True, "mask_name": mname, "anchor": None}), 200
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        a = data.get("anchor") if isinstance(data, dict) else None
        if isinstance(a, dict) and "x" in a and "y" in a:
            return jsonify({"ok": True, "mask_name": mname, "anchor": {"x": int(a["x"]), "y": int(a["y"])}}), 200
        return jsonify({"ok": True, "mask_name": mname, "anchor": None}), 200
    except Exception:
        return jsonify({"ok": True, "mask_name": mname, "anchor": None}), 200


@app.route('/api/project/<project_name>/input/mask_anchor/<mask_name>', methods=['POST'])
def set_mask_anchor(project_name: str, mask_name: str):
    """Set or clear the anchor point for a mask.

    Payload:
      { "anchor": { "x": 123, "y": 456 } }  OR  { "anchor": null }
    """
    payload = request.get_json(silent=True) or {}
    anchor = payload.get("anchor")
    try:
        project_dir = _safe_project_dir(project_name)
        mname = _safe_mask_name(mask_name)
    except Exception:
        return jsonify({"error": "Invalid project or mask"}), 400

    p = _anchor_path(project_dir, mname)
    p.parent.mkdir(parents=True, exist_ok=True)

    if anchor is None:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"ok": True, "mask_name": mname, "anchor": None}), 200

    if not isinstance(anchor, dict) or "x" not in anchor or "y" not in anchor:
        return jsonify({"error": "Expected {anchor: {x, y}} or {anchor: null}"}), 400
    try:
        x = int(anchor.get("x"))
        y = int(anchor.get("y"))
    except Exception:
        return jsonify({"error": "anchor.x and anchor.y must be integers"}), 400

    try:
        p.write_text(json.dumps({"anchor": {"x": x, "y": y}}, indent=2), encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Failed to save anchor: {e}"}), 500
    return jsonify({"ok": True, "mask_name": mname, "anchor": {"x": x, "y": y}}), 200


def _apply_anchor_component_to_mask_png(raw_png: bytes, anchor: dict | None) -> bytes:
    """Keep only the connected component that contains (or is nearest to) the anchor point."""
    if not raw_png or not isinstance(anchor, dict):
        return raw_png
    try:
        ax = int(anchor.get("x"))
        ay = int(anchor.get("y"))
    except Exception:
        return raw_png

    try:
        from PIL import Image  # type: ignore
        import io
        import numpy as np  # type: ignore
        from collections import deque

        img = Image.open(io.BytesIO(raw_png)).convert("L")
        arr = np.array(img)
        h, w = arr.shape[:2]
        if ax < 0 or ay < 0 or ax >= w or ay >= h:
            return raw_png

        mask = arr >= 128
        if not mask.any():
            return raw_png

        sy, sx = ay, ax
        if not mask[sy, sx]:
            found = False
            # Search for nearest white pixel in expanding squares
            for r in range(1, 90):
                y0 = max(0, sy - r)
                y1 = min(h, sy + r + 1)
                x0 = max(0, sx - r)
                x1 = min(w, sx + r + 1)
                window = mask[y0:y1, x0:x1]
                idx = np.argwhere(window)
                if idx.size:
                    coords = idx + np.array([y0, x0])
                    dy = coords[:, 0] - sy
                    dx = coords[:, 1] - sx
                    i = int(np.argmin(dx * dx + dy * dy))
                    sy, sx = int(coords[i, 0]), int(coords[i, 1])
                    found = True
                    break
            if not found:
                return raw_png

        out = np.zeros((h, w), dtype=np.uint8)
        q = deque([(sy, sx)])
        mask[sy, sx] = False
        out[sy, sx] = 255
        while q:
            y, x = q.popleft()
            if y > 0 and mask[y - 1, x]:
                mask[y - 1, x] = False
                out[y - 1, x] = 255
                q.append((y - 1, x))
            if y + 1 < h and mask[y + 1, x]:
                mask[y + 1, x] = False
                out[y + 1, x] = 255
                q.append((y + 1, x))
            if x > 0 and mask[y, x - 1]:
                mask[y, x - 1] = False
                out[y, x - 1] = 255
                q.append((y, x - 1))
            if x + 1 < w and mask[y, x + 1]:
                mask[y, x + 1] = False
                out[y, x + 1] = 255
                q.append((y, x + 1))

        out_img = Image.fromarray(out, mode="L")
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raw_png


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
