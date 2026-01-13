import pytest

from core.config.project_paths import ProjectPaths


def test_iteration_paths_run_mode(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()

    itr_paths = paths.iteration_paths(3, run_id="2026-01-12_08-50-00")

    assert itr_paths["image"].parent.name == "images"
    assert itr_paths["image"].name == "iteration_3.png"
    assert itr_paths["metadata"].parent.name == "metadata"


def test_iteration_paths_legacy_mode(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()

    itr_paths = paths.iteration_paths(5, run_id=None)

    assert itr_paths["image"].parent.name == "working"
    assert itr_paths["image"].name == "iteration_5.png"


def test_checkpoint_path(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()

    assert paths.checkpoint_path.name == "checkpoint.json"
