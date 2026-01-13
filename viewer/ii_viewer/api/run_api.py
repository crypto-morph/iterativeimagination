"""Run-level API endpoints (images, human ranking, input assets)."""

from __future__ import annotations

import json
import yaml
from pathlib import Path

from flask import Blueprint, jsonify, request, Response, send_from_directory

from ..settings import PROJECTS_ROOT, _TRANSPARENT_PNG_1X1
from ..services.projects import safe_project_dir


run_api_bp = Blueprint("run_api", __name__)


def _human_dir(project_name: str, run_id: str) -> Path:
    return PROJECTS_ROOT / project_name / "working" / run_id / "human"


@run_api_bp.route("/api/project/<project_name>/run/<run_id>/image/<filename>")
def get_image(project_name: str, run_id: str, filename: str):
    """Serve iteration images."""
    images_dir = PROJECTS_ROOT / project_name / "working" / run_id / "images"
    if images_dir.exists() and (images_dir / filename).exists():
        return send_from_directory(str(images_dir), filename)
    return jsonify({"error": "Image not found"}), 404


@run_api_bp.route("/api/project/<project_name>/run/<run_id>/human/ranking", methods=["GET"])
def get_human_ranking(project_name: str, run_id: str):
    """Get stored human ranking + notes for a run, if present."""
    path = _human_dir(project_name, run_id) / "ranking.json"
    if not path.exists():
        return jsonify({"error": "No ranking saved"}), 404
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    except Exception:
        return jsonify({"error": "Failed to read ranking.json"}), 500


@run_api_bp.route("/api/project/<project_name>/run/<run_id>/human/ranking", methods=["POST"])
def save_human_ranking(project_name: str, run_id: str):
    """Save human ranking + notes for a run under working/<run_id>/human/."""
    payload = request.get_json(silent=True) or {}
    ranking = payload.get("ranking")
    notes = payload.get("notes", "")
    if not isinstance(ranking, list):
        return jsonify({"error": "ranking must be a list"}), 400

    human_dir = _human_dir(project_name, run_id)
    human_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(human_dir / "ranking.json", "w", encoding="utf-8") as f:
            json.dump({"ranking": ranking, "notes": notes}, f, indent=2)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save ranking: {e}"}), 500


@run_api_bp.route("/api/project/<project_name>/input/image")
def get_input_image(project_name: str):
    """Serve the original input image."""
    input_dir = PROJECTS_ROOT / project_name / "input"
    if input_dir.exists() and (input_dir / "input.png").exists():
        return send_from_directory(str(input_dir), "input.png")
    return jsonify({"error": "Input image not found"}), 404


@run_api_bp.route("/api/project/<project_name>/input/mask")
def get_input_mask(project_name: str):
    """Serve the current mask.png (if present)."""
    input_dir = PROJECTS_ROOT / project_name / "input"
    if input_dir.exists() and (input_dir / "mask.png").exists():
        return send_from_directory(str(input_dir), "mask.png")
    return Response(_TRANSPARENT_PNG_1X1, mimetype="image/png")


@run_api_bp.route("/api/project/<project_name>/input/mask", methods=["POST"])
def save_input_mask(project_name: str):
    """Save mask.png from a data URL (PNG). Creates a timestamped backup if mask exists."""
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("data_url", "")
    if not data_url.startswith("data:image/png;base64,"):
        return jsonify({"error": "Invalid data URL"}), 400

    try:
        import base64
        import time
        png_bytes = base64.b64decode(data_url.split(",", 1)[1])
        input_dir = PROJECTS_ROOT / project_name / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        mask_path = input_dir / "mask.png"
        if mask_path.exists():
            backup = input_dir / f"mask.{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            mask_path.rename(backup)
        mask_path.write_bytes(png_bytes)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save mask: {e}"}), 500
