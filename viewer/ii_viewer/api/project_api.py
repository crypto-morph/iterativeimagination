"""Project-related JSON API endpoints."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify

from ..settings import PROJECTS_ROOT


project_api_bp = Blueprint("project_api", __name__)


@project_api_bp.route("/api/project/<project_name>/run/<run_id>/iterations")
def get_iterations(project_name: str, run_id: str):
    """Get all iteration metadata for a run."""
    run_dir = PROJECTS_ROOT / project_name / "working" / run_id
    metadata_dir = run_dir / "metadata"

    # For live runs, the browser may poll before any iterations have been written.
    if not metadata_dir.exists():
        if run_dir.exists():
            return jsonify({"iterations": []}), 200
        return jsonify({"error": "Run not found"}), 404

    iterations = []
    for meta_file in sorted(metadata_dir.glob("iteration_*_metadata.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            iter_num = int(meta_file.stem.replace("iteration_", "").replace("_metadata", ""))
            data["iteration_number"] = iter_num
            iterations.append(data)
        except Exception:
            continue

    iterations.sort(key=lambda x: x.get("iteration_number", 0))
    return jsonify({"iterations": iterations}), 200
