"""Project config editor endpoints."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, request
import yaml

from ..services.projects import safe_project_dir


config_api_bp = Blueprint("config_api", __name__)


def _project_file(project_dir: Path, relative: str) -> Path:
    return project_dir / relative


def _working_aigen_path(project_dir: Path) -> Path:
    return project_dir / "working" / "AIGen.yaml"


@config_api_bp.route("/api/project/<project_name>/config/rules", methods=["GET"])
def get_rules_text(project_name: str):
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = _project_file(project_dir, "config/rules.yaml")
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404
    return Response(rules_path.read_text(encoding="utf-8"), mimetype="text/plain")


@config_api_bp.route("/api/project/<project_name>/config/rules", methods=["POST"])
def save_rules_text(project_name: str):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing rules text"}), 400

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    try:
        rules_obj = yaml.safe_load(text) or {}
    except Exception as exc:
        return jsonify({"error": f"YAML parse error: {exc}"}), 400

    errors, warnings = [], []
    try:
        from ..services.rules_lint import lint_rules_obj  # late import

        errors, warnings = lint_rules_obj(rules_obj)
    except Exception:
        errors, warnings = [], []

    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings}), 400

    import time

    backup = rules_path.with_name(f"rules.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    rules_path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True, "warnings": warnings}), 200


def _read_yaml_file(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml_file(path: Path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


@config_api_bp.route("/api/project/<project_name>/config/aigen", methods=["GET"])
def get_aigen_yaml(project_name: str):
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIGen.yaml"
    if not path.exists():
        return jsonify({"error": "AIGen.yaml not found"}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")


@config_api_bp.route("/api/project/<project_name>/config/aigen", methods=["POST"])
def save_aigen_yaml(project_name: str):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing AIGen.yaml text"}), 400

    try:
        yaml.safe_load(text)
    except Exception as exc:
        return jsonify({"error": f"YAML parse error: {exc}"}), 400

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIGen.yaml"
    if not path.exists():
        return jsonify({"error": "AIGen.yaml not found"}), 404

    import time

    backup = path.with_name(f"AIGen.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True}), 200


@config_api_bp.route("/api/project/<project_name>/config/aivis", methods=["GET"])
def get_aivis_yaml(project_name: str):
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIVis.yaml"
    if not path.exists():
        return jsonify({"error": "AIVis.yaml not found"}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/plain")


@config_api_bp.route("/api/project/<project_name>/config/aivis", methods=["POST"])
def save_aivis_yaml(project_name: str):
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing AIVis.yaml text"}), 400

    try:
        yaml.safe_load(text)
    except Exception as exc:
        return jsonify({"error": f"YAML parse error: {exc}"}), 400

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400
    path = project_dir / "config" / "AIVis.yaml"
    if not path.exists():
        return jsonify({"error": "AIVis.yaml not found"}), 404

    import time

    backup = path.with_name(f"AIVis.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True}), 200


@config_api_bp.route("/api/project/<project_name>/working/aigen/prompts", methods=["GET"])
def get_working_prompts(project_name: str):
    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    path = _working_aigen_path(project_dir)
    if not path.exists():
        cfg = project_dir / "config" / "AIGen.yaml"
        if cfg.exists():
            path = cfg
        else:
            return jsonify({"error": "AIGen.yaml not found"}), 404

    try:
        data = _read_yaml_file(path)
        prompts = data.get("prompts") or {}
        return jsonify(
            {
                "positive": str(prompts.get("positive") or ""),
                "negative": str(prompts.get("negative") or ""),
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to read AIGen.yaml: {exc}"}), 500


@config_api_bp.route("/api/project/<project_name>/working/aigen/prompts", methods=["POST"])
def save_working_prompts(project_name: str):
    payload = request.get_json(silent=True) or {}
    positive = payload.get("positive", "")
    negative = payload.get("negative", "")
    if not isinstance(positive, str) or not isinstance(negative, str):
        return jsonify({"error": "positive and negative must be strings"}), 400

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    working_dir = project_dir / "working"
    working_dir.mkdir(parents=True, exist_ok=True)
    path = _working_aigen_path(project_dir)

    if not path.exists():
        cfg = project_dir / "config" / "AIGen.yaml"
        if cfg.exists():
            path.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump({"prompts": {"positive": "", "negative": ""}}, sort_keys=False), encoding="utf-8")

    import time

    backup = working_dir / f"AIGen.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    try:
        data = _read_yaml_file(path)
        if not isinstance(data, dict):
            data = {}
        data.setdefault("prompts", {})
        if not isinstance(data["prompts"], dict):
            data["prompts"] = {}
        data["prompts"]["positive"] = positive
        data["prompts"]["negative"] = negative
        _write_yaml_file(path, data)
        return jsonify({"ok": True}), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to save AIGen.yaml: {exc}"}), 500
