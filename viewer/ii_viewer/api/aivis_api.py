"""AIVis-backed API endpoints for the viewer."""

from __future__ import annotations

from flask import Blueprint, jsonify

from ..services.aivis import create_aivis_client
from ..services.projects import safe_project_dir


aivis_api_bp = Blueprint("aivis_api", __name__)


@aivis_api_bp.route("/api/project/<project_name>/input/describe")
def describe_input(project_name: str):
    """Get AIVis description of the input image (text)."""
    try:
        project_dir = safe_project_dir(project_name)
    except ValueError:
        return jsonify({"error": "Invalid project"}), 400
    except FileNotFoundError:
        return jsonify({"error": "Invalid project"}), 400

    input_path = project_dir / "input" / "input.png"
    if not input_path.exists():
        return jsonify({"error": "Input image not found"}), 404

    try:
        aivis = create_aivis_client(project_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500

    try:
        description = aivis.describe_image(str(input_path))
        return jsonify({"ok": True, "description": description}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image: {e}"}), 500


@aivis_api_bp.route("/api/project/<project_name>/input/describe_terms")
def describe_input_terms(project_name: str):
    """Get AIVis description of the input image as structured term lists."""
    try:
        project_dir = safe_project_dir(project_name)
    except ValueError:
        return jsonify({"error": "Invalid project"}), 400
    except FileNotFoundError:
        return jsonify({"error": "Invalid project"}), 400

    input_path = project_dir / "input" / "input.png"
    if not input_path.exists():
        return jsonify({"error": "Input image not found"}), 404

    try:
        aivis = create_aivis_client(project_dir)
    except Exception as e:
        return jsonify({"error": f"Failed to initialize AIVis: {e}"}), 500

    try:
        terms = aivis.describe_image_as_terms(str(input_path))
        return jsonify({"ok": True, **terms}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to describe image as terms: {e}"}), 500
