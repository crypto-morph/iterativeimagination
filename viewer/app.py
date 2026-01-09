#!/usr/bin/env python3
"""
Iteration Viewer - Web app to visualize iteration images and metadata
"""

import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory
import yaml

app = Flask(__name__)
PROJECTS_ROOT = Path(__file__).parent.parent / "projects"


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


@app.route('/api/project/<project_name>/run/<run_id>/iterations')
def get_iterations(project_name: str, run_id: str):
    """Get all iteration metadata for a run."""
    run_dir = PROJECTS_ROOT / project_name / "working" / run_id
    metadata_dir = run_dir / "metadata"
    
    if not metadata_dir.exists():
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
