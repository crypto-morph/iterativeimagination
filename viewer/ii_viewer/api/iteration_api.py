"""Iteration description API endpoints."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify

from ..services.aivis import create_aivis_client
from ..services.projects import safe_project_dir


iteration_api_bp = Blueprint("iteration_api", __name__)


def _load_iteration_metadata(project_dir, run_id: str, iteration_number: int):
    run_dir = project_dir / "working" / run_id
    metadata_path = run_dir / "metadata" / f"iteration_{iteration_number}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata not found")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


@iteration_api_bp.route("/api/project/<project_name>/run/<run_id>/describe_terms/<int:iteration_number>")
def describe_iteration_terms(project_name: str, run_id: str, iteration_number: int):
    """Get AIVis description of an iteration image as structured term lists."""
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    run_dir = project_dir / "working" / run_id
    image_path = run_dir / "images" / f"iteration_{iteration_number}.png"
    if not image_path.exists():
        return jsonify({"error": "Iteration image not found"}), 404

    try:
        aivis = create_aivis_client(project_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500

    try:
        terms = aivis.describe_image_as_terms(str(image_path))
        return jsonify({"ok": True, **terms}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500


@iteration_api_bp.route("/api/project/<project_name>/run/<run_id>/describe/<int:iteration_number>")
def describe_iteration(project_name: str, run_id: str, iteration_number: int):
    """Get AIVis description of an iteration image."""
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    run_dir = project_dir / "working" / run_id
    image_path = run_dir / "images" / f"iteration_{iteration_number}.png"
    if not image_path.exists():
        return jsonify({"error": "Iteration image not found"}), 404

    try:
        aivis = create_aivis_client(project_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500

    try:
        description = aivis.describe_image(str(image_path))
        return jsonify({"ok": True, "description": description}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500
