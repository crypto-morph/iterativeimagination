#!/usr/bin/env python3
"""
Iteration Viewer - Web app to visualize iteration images and metadata
"""

import json
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, Response
import yaml

app = Flask(__name__)
PROJECTS_ROOT = Path(__file__).parent.parent / "projects"

# Live run registry (in-memory)
_LIVE_RUNS_LOCK = threading.Lock()
_LIVE_RUNS = {}  # (project_name, run_id) -> LiveRunController


@app.route('/')
def index():
    """Main page listing all projects."""
    projects = []
    if PROJECTS_ROOT.exists():
        for project_dir in sorted(PROJECTS_ROOT.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
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
                            [f.stem.replace('iteration_', '').replace('_metadata', '') 
                             for f in metadata_dir.glob('iteration_*_metadata.json')],
                            key=lambda x: int(x) if x.isdigit() else 0
                        )
                        if iterations:
                            runs.append({
                                'id': run_dir.name,
                                'iterations': len(iterations),
                                'first_iteration': int(iterations[0]) if iterations else 0,
                                'last_iteration': int(iterations[-1]) if iterations else 0
                            })
            if runs:
                projects.append({
                    'name': project_dir.name,
                    'runs': runs
                })
    return render_template('index.html', projects=projects)


@app.route('/project/<project_name>/run/<run_id>')
def view_run(project_name: str, run_id: str):
    """View a specific run with all iterations."""
    return render_template('run.html', project_name=project_name, run_id=run_id)


@app.route('/project/<project_name>/live')
def live_run(project_name: str):
    """Live interactive run page (one iteration at a time)."""
    return render_template('live.html', project_name=project_name)


@app.route('/project/<project_name>/run/<run_id>/rank')
def rank_run(project_name: str, run_id: str):
    """Ranking page for a run (human feedback)."""
    return render_template('rank.html', project_name=project_name, run_id=run_id)


@app.route('/api/project/<project_name>/run/<run_id>/iterations')
def get_iterations(project_name: str, run_id: str):
    """Get all iteration metadata for a run."""
    run_dir = PROJECTS_ROOT / project_name / "working" / run_id
    metadata_dir = run_dir / "metadata"
    
    # For live runs, the browser may poll before any iterations have been written.
    # In that case, treat missing metadata_dir as "no iterations yet" rather than 404.
    if not metadata_dir.exists():
        if run_dir.exists():
            return jsonify({"iterations": []}), 200
        return jsonify({'error': 'Run not found'}), 404
    
    iterations = []
    for meta_file in sorted(metadata_dir.glob('iteration_*_metadata.json')):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract iteration number
                iter_num = int(meta_file.stem.replace('iteration_', '').replace('_metadata', ''))
                data['iteration_number'] = iter_num
                iterations.append(data)
        except Exception as e:
            continue
    
    # Sort by iteration number
    iterations.sort(key=lambda x: x.get('iteration_number', 0))
    
    return jsonify({'iterations': iterations})


@app.route('/api/project/<project_name>/run/<run_id>/image/<filename>')
def get_image(project_name: str, run_id: str, filename: str):
    """Serve iteration images."""
    images_dir = PROJECTS_ROOT / project_name / "working" / run_id / "images"
    if images_dir.exists() and (images_dir / filename).exists():
        return send_from_directory(str(images_dir), filename)
    return jsonify({'error': 'Image not found'}), 404


def _human_dir(project_name: str, run_id: str) -> Path:
    return PROJECTS_ROOT / project_name / "working" / run_id / "human"


@app.route('/api/project/<project_name>/run/<run_id>/human/ranking', methods=['GET'])
def get_human_ranking(project_name: str, run_id: str):
    """Get stored human ranking + notes for a run, if present."""
    path = _human_dir(project_name, run_id) / "ranking.json"
    if not path.exists():
        return jsonify({"ranking": [], "notes": {}, "updated_at": None}), 200
    try:
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200
    except Exception as e:
        return jsonify({"error": "Failed to read ranking.json"}), 500


@app.route('/api/project/<project_name>/run/<run_id>/human/ranking', methods=['POST'])
def save_human_ranking(project_name: str, run_id: str):
    """Save human ranking + notes for a run under working/<run_id>/human/."""
    payload = request.get_json(silent=True) or {}
    ranking = payload.get("ranking") or []
    notes = payload.get("notes") or {}

    if not isinstance(ranking, list):
        return jsonify({"error": "ranking must be a list"}), 400
    if not isinstance(notes, dict):
        return jsonify({"error": "notes must be a dict"}), 400

    # Normalise keys to strings for JSON stability
    ranking_norm = [str(x) for x in ranking]
    notes_norm = {str(k): str(v) for k, v in notes.items()}

    out = {
        "ranking": ranking_norm,
        "notes": notes_norm,
        "updated_at": __import__("time").time(),
    }

    human_dir = _human_dir(project_name, run_id)
    human_dir.mkdir(parents=True, exist_ok=True)
    path = human_dir / "ranking.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return jsonify({"ok": True}), 200


@app.route('/api/project/<project_name>/input/image')
def get_input_image(project_name: str):
    """Serve the original input image."""
    input_dir = PROJECTS_ROOT / project_name / "input"
    if (input_dir / "input.png").exists():
        return send_from_directory(str(input_dir), "input.png")
    return jsonify({'error': 'Input image not found'}), 404


@app.route('/api/project/<project_name>/config')
def get_project_config(project_name: str):
    """Get project configuration (rules.yaml)."""
    rules_path = PROJECTS_ROOT / project_name / "config" / "rules.yaml"
    if rules_path.exists():
        with open(rules_path, 'r', encoding='utf-8') as f:
            return jsonify(yaml.safe_load(f))
    return jsonify({'error': 'Config not found'}), 404


def _safe_project_dir(project_name: str) -> Path:
    root = PROJECTS_ROOT.resolve()
    p = (PROJECTS_ROOT / project_name).resolve()
    if not str(p).startswith(str(root) + "/") and p != root:
        raise ValueError("Invalid project")
    return p


@app.route('/api/project/<project_name>/config/rules', methods=['GET'])
def get_rules_yaml(project_name: str):
    """Get raw rules.yaml text for editing."""
    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404
    return Response(rules_path.read_text(encoding="utf-8"), mimetype="text/plain")


@app.route('/api/project/<project_name>/config/rules', methods=['POST'])
def save_rules_yaml(project_name: str):
    """Save raw rules.yaml text (validates YAML and lints rules)."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing rules text"}), 400

    try:
        project_dir = _safe_project_dir(project_name)
    except Exception:
        return jsonify({"error": "Invalid project"}), 400

    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return jsonify({"error": "rules.yaml not found"}), 404

    # Parse YAML for validation
    try:
        rules_obj = yaml.safe_load(text) or {}
    except Exception as e:
        return jsonify({"error": f"YAML parse error: {e}"}), 400

    # Lint using existing rules_checker if available
    errors = []
    warnings = []
    try:
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
        from rules_checker import lint_rules  # noqa: E402

        errors, warnings = lint_rules(rules_obj)
    except Exception:
        # If linting cannot run, treat as no lint results (YAML parse already succeeded).
        errors, warnings = [], []

    if errors:
        return jsonify({"ok": False, "errors": errors, "warnings": warnings}), 400

    # Backup existing file then write raw text (preserves comments/formatting from the editor).
    import time

    backup = rules_path.with_name(f"rules.yaml.bak.{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    try:
        backup.write_text(rules_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    rules_path.write_text(text, encoding="utf-8")
    return jsonify({"ok": True, "warnings": warnings}), 200


def _make_run_id() -> str:
    import time

    return time.strftime("%Y-%m-%d_%H-%M-%S")


@app.route('/api/project/<project_name>/live/start', methods=['POST'])
def live_start(project_name: str):
    """Start a live run for a project, returning a run_id."""
    payload = request.get_json(silent=True) or {}
    max_iterations = int(payload.get("max_iterations") or 20)
    reset = bool(payload.get("reset"))

    # Import here to avoid slowing initial page load.
    from live_runner import LiveRunController

    # Optionally reset project working state before creating the new run folder.
    if reset:
        try:
            import sys
            from pathlib import Path as _Path

            sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
            from project_manager import ProjectManager  # noqa: E402
            # Best-effort reset: remove checkpoint so a fresh loop is forced.
            ckpt = ProjectManager(project_name).get_checkpoint_path()
            if ckpt.exists():
                ckpt.unlink()
        except Exception:
            pass

    run_id = _make_run_id()

    # Create run directories immediately so the UI can poll without 404s.
    run_root = PROJECTS_ROOT / project_name / "working" / run_id
    for sub in ("images", "questions", "evaluation", "comparison", "metadata", "human"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)

    if reset:
        # Best-effort reset: remove checkpoint so a fresh loop is forced.
        ckpt = PROJECTS_ROOT / project_name / "working" / "checkpoint.json"
        if ckpt.exists():
            try:
                ckpt.unlink()
            except Exception:
                pass

    ctl = LiveRunController(project=project_name, run_id=run_id, max_iterations=max_iterations)
    with _LIVE_RUNS_LOCK:
        _LIVE_RUNS[(project_name, run_id)] = ctl
    ctl.start()
    return jsonify({"ok": True, "run_id": run_id}), 200


@app.route('/api/project/<project_name>/live/<run_id>/state')
def live_state(project_name: str, run_id: str):
    """Get live run controller state."""
    with _LIVE_RUNS_LOCK:
        ctl = _LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404
    return jsonify({"state": ctl.state()}), 200


@app.route('/api/project/<project_name>/live/<run_id>/feedback', methods=['POST'])
def live_feedback(project_name: str, run_id: str):
    """Submit feedback (comment + nudges) for the current iteration and resume."""
    payload = request.get_json(silent=True) or {}
    iteration = int(payload.get("iteration") or 0)
    comment = str(payload.get("comment") or "")
    nudge = payload.get("nudge") or {}
    if not isinstance(nudge, dict):
        nudge = {}

    with _LIVE_RUNS_LOCK:
        ctl = _LIVE_RUNS.get((project_name, run_id))
    if not ctl:
        return jsonify({"error": "Live run not found"}), 404

    ctl.submit_feedback(iteration=iteration, comment=comment, nudge=nudge)
    return jsonify({"ok": True}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
