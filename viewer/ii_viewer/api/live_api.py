"""Live-run API endpoints."""

from __future__ import annotations

import time

from flask import Blueprint, jsonify, request

from ..registry import LIVE_RUNS, LIVE_RUNS_LOCK
from ..services.projects import safe_project_dir


live_api_bp = Blueprint("live_api", __name__)


def _make_run_id() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _import_live_controller():
    try:
        from live_runner import LiveRunController  # type: ignore
        return LiveRunController
    except Exception:
        from viewer.live_runner import LiveRunController  # type: ignore
        return LiveRunController


@live_api_bp.route("/api/project/<project_name>/live/start", methods=["POST"])
def live_start(project_name: str):
    """Start a live run for a project, returning a run_id."""
    payload = request.get_json(silent=True) or {}
    max_iterations = int(payload.get("max_iterations") or 20)
    reset = bool(payload.get("reset"))
    mask_name = payload.get("mask_name") or "global"

    try:
        project_dir = safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    LiveRunController = _import_live_controller()

    if reset:
        try:
            ckpt = project_dir / "working" / "checkpoint.json"
            if ckpt.exists():
                ckpt.unlink()
        except Exception:
            pass

    # Ensure mask is set in working/AIGen.yaml before starting
    if mask_name and mask_name != "global":
        try:
            import yaml

            working_aigen = project_dir / "working" / "AIGen.yaml"
            aigen_config = {}
            if working_aigen.exists():
                with open(working_aigen, "r", encoding="utf-8") as f:
                    aigen_config = yaml.safe_load(f) or {}

            aigen_config.setdefault("masking", {})
            aigen_config["masking"]["enabled"] = True
            aigen_config["masking"]["active_mask"] = mask_name

            with open(working_aigen, "w", encoding="utf-8") as f:
                yaml.dump(aigen_config, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass

    run_id = _make_run_id()

    # Create run directories immediately so the UI can poll without 404s.
    run_root = project_dir / "working" / run_id
    for sub in ("images", "questions", "evaluation", "comparison", "metadata", "human"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    ctl = LiveRunController(project=project_name, run_id=run_id, max_iterations=max_iterations)
    with LIVE_RUNS_LOCK:
        LIVE_RUNS[(project_name, run_id)] = ctl
    ctl.start()

    return jsonify({"ok": True, "run_id": run_id}), 200


@live_api_bp.route("/api/project/<project_name>/live/<run_id>/state")
def live_state(project_name: str, run_id: str):
    """Get live run controller state."""
    with LIVE_RUNS_LOCK:
        ctl = LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404
    return jsonify({"state": ctl.state()}), 200


@live_api_bp.route("/api/project/<project_name>/live/<run_id>/feedback", methods=["POST"])
def live_feedback(project_name: str, run_id: str):
    """Submit feedback (comment + nudges) for the current iteration and resume."""
    payload = request.get_json(silent=True) or {}
    iteration = int(payload.get("iteration") or 0)
    comment = str(payload.get("comment") or "")
    nudge = payload.get("nudge") or {}
    if not isinstance(nudge, dict):
        nudge = {}

    with LIVE_RUNS_LOCK:
        ctl = LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404

    ctl.submit_feedback(iteration=iteration, comment=comment, nudge=nudge)
    return jsonify({"ok": True}), 200
