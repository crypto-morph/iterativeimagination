"""Page (HTML) routes for the viewer."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, render_template, request

from ..settings import PROJECTS_ROOT


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Main page listing all projects."""
    projects = []
    if PROJECTS_ROOT.exists():
        for project_dir in sorted(PROJECTS_ROOT.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith("."):
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
                            [
                                f.stem.replace("iteration_", "").replace("_metadata", "")
                                for f in metadata_dir.glob("iteration_*_metadata.json")
                            ],
                            key=lambda x: int(x) if x.isdigit() else 0,
                        )
                        if iterations:
                            runs.append(
                                {
                                    "id": run_dir.name,
                                    "iterations": len(iterations),
                                    "first_iteration": int(iterations[0]) if iterations else 0,
                                    "last_iteration": int(iterations[-1]) if iterations else 0,
                                }
                            )
            if runs:
                projects.append({"name": project_dir.name, "runs": runs})
    return render_template("index.html", projects=projects)


@pages_bp.route("/project/<project_name>/run/<run_id>")
def view_run(project_name: str, run_id: str):
    """View a specific run with all iterations."""
    return render_template("run.html", project_name=project_name, run_id=run_id)


@pages_bp.route("/project/<project_name>/live")
def live_run(project_name: str):
    """Live interactive run page (one iteration at a time)."""
    return render_template("live.html", project_name=project_name)


@pages_bp.route("/project/<project_name>/mask")
def mask_editor(project_name: str):
    """Mask editor page (supports multiple masks)."""
    mask_name = (request.args.get("mask") or "default").strip() or "default"
    return render_template("mask.html", project_name=project_name, mask_name=mask_name)


@pages_bp.route("/project/<project_name>/mask/<mask_name>")
def mask_editor_named(project_name: str, mask_name: str):
    """Mask editor page for a specific mask name."""
    mask_name = (mask_name or "default").strip() or "default"
    return render_template("mask.html", project_name=project_name, mask_name=mask_name)


@pages_bp.route("/project/<project_name>/rulesui")
def rules_ui(project_name: str):
    """Structured rules.yaml editor UI."""
    return render_template("rules_ui.html", project_name=project_name)


@pages_bp.route("/project/<project_name>/run/<run_id>/rank")
def rank_run(project_name: str, run_id: str):
    """Ranking page for a run (human feedback)."""
    return render_template("rank.html", project_name=project_name, run_id=run_id)
