"""Prompt suggestion API endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from ..services.aivis import create_aivis_client
from ..services.projects import safe_project_dir
from ..services.prompt_terms import compare_term_lists, parse_prompt_to_terms
from ..services.rules import load_rules


prompt_suggestions_api_bp = Blueprint("prompt_suggestions_api", __name__)


@prompt_suggestions_api_bp.route("/api/project/<project_name>/run/<run_id>/suggest_prompts")
def suggest_prompts(project_name: str, run_id: str):
    """Get AIVis suggestions for improving prompts based on latest iteration."""
    mask_name = request.args.get("mask_name", "global").strip() or "global"

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    run_dir = project_dir / "working" / run_id
    metadata_dir = run_dir / "metadata"
    if not metadata_dir.exists():
        return jsonify({"error": "No iterations found"}), 404

    meta_files = sorted(metadata_dir.glob("iteration_*_metadata.json"))
    if not meta_files:
        return jsonify({"error": "No iterations found"}), 404

    latest_meta = meta_files[-1]
    with open(latest_meta, "r", encoding="utf-8") as f:
        iteration_data = json.load(f)

    iteration_num = iteration_data.get("iteration", 1)
    image_path = run_dir / "images" / f"iteration_{iteration_num}.png"
    if not image_path.exists():
        return jsonify({"error": "Iteration image not found"}), 404

    current_positive_str = (iteration_data.get("prompts_used") or {}).get("positive", "")
    current_negative_str = (iteration_data.get("prompts_used") or {}).get("negative", "")
    current_positive_terms = parse_prompt_to_terms(current_positive_str)
    current_negative_terms = parse_prompt_to_terms(current_negative_str)

    evaluation = iteration_data.get("evaluation") or {}
    comparison = iteration_data.get("comparison") or {}

    try:
        aivis = create_aivis_client(project_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500

    # Get original description (grounding/context)
    input_path = project_dir / "input" / "input.png"
    original_desc = None
    try:
        if input_path.exists():
            original_desc = aivis.describe_image(str(input_path))
    except Exception:
        original_desc = None

    # Get latest iteration description
    latest_iteration_desc = None
    try:
        latest_iteration_desc = aivis.describe_image(str(image_path))
    except Exception:
        latest_iteration_desc = None

    # Prepare prompt improver
    try:
        from prompt_improver import PromptImprover  # imported from repo src/ via create_aivis_client
    except Exception as e:
        return jsonify({"error": f"Failed to import PromptImprover: {e}"}), 500

    logger = logging.getLogger(__name__)
    improver = PromptImprover(logger, aivis, lambda: original_desc or "", "")

    rules_obj = load_rules(project_dir)
    criteria_defs = rules_obj.get("acceptance_criteria") or []
    criteria_by_field = {
        c.get("field"): c
        for c in criteria_defs
        if isinstance(c, dict) and c.get("field")
    }

    failed_criteria = []
    criteria_results = evaluation.get("criteria_results", {}) or {}
    for field, result in criteria_results.items():
        crit = criteria_by_field.get(field)
        if not crit:
            continue
        ctype = (crit.get("type") or "boolean").lower()
        min_v = crit.get("min", 0)
        if ctype == "boolean":
            if result is not True:
                failed_criteria.append(field)
        else:
            if isinstance(result, (int, float)) and result < min_v:
                failed_criteria.append(field)

    description_for_improvement = latest_iteration_desc or original_desc

    try:
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
    except Exception as e:
        return jsonify({"error": f"Failed to generate suggestions: {e}"}), 500

    suggested_positive_terms = parse_prompt_to_terms(improved_positive)
    suggested_negative_terms = parse_prompt_to_terms(improved_negative)

    positive_suggestions = compare_term_lists(current_positive_terms, suggested_positive_terms)
    negative_suggestions = compare_term_lists(current_negative_terms, suggested_negative_terms)

    mask_suggestions = None
    if mask_name and mask_name != "global":
        # We can re-introduce mask term suggestions later as a dedicated endpoint.
        mask_suggestions = None

    return (
        jsonify(
            {
                "ok": True,
                "current_positive_terms": current_positive_terms,
                "current_negative_terms": current_negative_terms,
                "suggested_positive_terms": suggested_positive_terms,
                "suggested_negative_terms": suggested_negative_terms,
                "positive_suggestions": positive_suggestions,
                "negative_suggestions": negative_suggestions,
                "diff_info": diff_info,
                "latest_iteration_description": latest_iteration_desc,
                "iteration_number": iteration_num,
                "mask_suggestions": mask_suggestions,
                "mask_name": mask_name,
            }
        ),
        200,
    )
