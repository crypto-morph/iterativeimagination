"""Project path helpers for the viewer."""

from __future__ import annotations

from pathlib import Path

from ..settings import PROJECTS_ROOT


def safe_project_dir(project_name: str) -> Path:
    root = PROJECTS_ROOT.resolve()
    p = (PROJECTS_ROOT / project_name).resolve()
    if not str(p).startswith(str(root) + "/") and p != root:
        raise ValueError("Invalid project")
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError("Project not found")
    return p
