"""AIGen settings JSON API endpoints."""

from __future__ import annotations

import time

from flask import Blueprint, jsonify, request
import yaml

from ..services.projects import safe_project_dir


aigen_api_bp = Blueprint("aigen_api", __name__)


def _working_aigen_path(project_dir):
    return project_dir / "working" / "AIGen.yaml"


@aigen_api_bp.route("/api/project/<project_name>/working/aigen/settings", methods=["GET"])
def get_aigen_settings(project_name: str):
    """Get current AIGen settings (denoise, cfg, scheduler, model, etc.)."""
    try:
        project_dir = safe_project_dir(project_name)
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

        return (
            jsonify(
                {
                    "denoise": params.get("denoise", 0.5),
                    "cfg": params.get("cfg", 7.0),
                    "steps": params.get("steps", 25),
                    "sampler_name": params.get("sampler_name", "dpmpp_2m_sde_gpu"),
                    "scheduler": params.get("scheduler", "karras"),
                    "checkpoint": model_info.get("ckpt_name", "unknown"),
                    "workflow": aigen.get("workflow_file", "unknown"),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to load settings: {e}"}), 500


@aigen_api_bp.route("/api/project/<project_name>/working/aigen/settings", methods=["POST"])
def save_aigen_settings(project_name: str):
    """Save AIGen settings to working/AIGen.yaml (creates timestamped backup)."""
    payload = request.get_json(silent=True) or {}

    denoise = payload.get("denoise")
    cfg = payload.get("cfg")
    steps = payload.get("steps")
    sampler_name = payload.get("sampler_name")
    scheduler = payload.get("scheduler")

    if denoise is None or cfg is None or steps is None or not sampler_name or not scheduler:
        return (
            jsonify({"error": "Missing required fields: denoise, cfg, steps, sampler_name, scheduler"}),
            400,
        )

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    working_dir = project_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    path = _working_aigen_path(project_dir)

    if not path.exists():
        cfg_path = project_dir / "config" / "AIGen.yaml"
        if cfg_path.exists():
            path.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text(
                yaml.safe_dump(
                    {
                        "parameters": {
                            "denoise": 0.5,
                            "cfg": 7.0,
                            "steps": 25,
                            "sampler_name": "dpmpp_2m_sde_gpu",
                            "scheduler": "karras",
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

    backup = working_dir / f"AIGen.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            data = {}

        data.setdefault("parameters", {})
        if not isinstance(data["parameters"], dict):
            data["parameters"] = {}

        data["parameters"]["denoise"] = float(denoise)
        data["parameters"]["cfg"] = float(cfg)
        data["parameters"]["steps"] = int(steps)
        data["parameters"]["sampler_name"] = str(sampler_name)
        data["parameters"]["scheduler"] = str(scheduler)

        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to save AIGen.yaml: {e}"}), 500
