"""Mask-related JSON API endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify
import yaml

from ..services.projects import safe_project_dir


mask_api_bp = Blueprint("mask_api", __name__)


@mask_api_bp.route("/api/project/<project_name>/mask/<mask_name>/prompts")
def get_mask_prompts(project_name: str, mask_name: str):
    """Get prompts for a specific mask (from rules.yaml)."""
    try:
        project_dir = safe_project_dir(project_name)
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

        if mask_name and mask_name != "global" and isinstance(masks_prompts, dict) and mask_name in masks_prompts:
            mask_p = masks_prompts[mask_name] or {}
            return (
                jsonify(
                    {
                        "positive": str(mask_p.get("positive") or ""),
                        "negative": str(mask_p.get("negative") or ""),
                    }
                ),
                200,
            )

        return (
            jsonify(
                {
                    "positive": str(global_prompts.get("positive") or ""),
                    "negative": str(global_prompts.get("negative") or ""),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to load prompts: {e}"}), 500


@mask_api_bp.route("/api/project/<project_name>/mask/<mask_name>/terms")
def get_mask_terms(project_name: str, mask_name: str):
    """Get terms (must_include, ban_terms, avoid_terms) for a specific mask."""
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    try:
        rules_obj = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
        masking = rules_obj.get("masking") or {}
        masks = masking.get("masks") or []

        active_fields = []
        if mask_name == "global":
            criteria_all = rules_obj.get("acceptance_criteria") or []
            if isinstance(criteria_all, list):
                for c in criteria_all:
                    if isinstance(c, dict) and c.get("field"):
                        active_fields.append(str(c.get("field")))
        else:
            for m in masks:
                if isinstance(m, dict) and str(m.get("name") or "") == mask_name:
                    ac = m.get("active_criteria") or []
                    if isinstance(ac, list):
                        active_fields = [str(x).strip() for x in ac if str(x).strip()]
                    break

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

        def dedupe(lst):
            seen = set()
            result = []
            for item in lst:
                item_str = str(item).strip()
                if item_str and item_str.lower() not in seen:
                    seen.add(item_str.lower())
                    result.append(item_str)
            return result

        return (
            jsonify(
                {
                    "must_include": dedupe(must_include),
                    "ban_terms": dedupe(ban_terms),
                    "avoid_terms": dedupe(avoid_terms),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to load terms: {e}"}), 500
