import textwrap

import pytest

from core.config.project_paths import ProjectPaths
from core.config.config_store import ConfigStore


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_load_rules(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()
    rules_path = paths.config_file("rules.yaml")
    write_file(
        rules_path,
        """
        project:
          name: demo
        """,
    )

    store = ConfigStore(paths, defaults_root=tmp_path / "defaults")
    rules = store.load_rules()
    assert rules["project"]["name"] == "demo"


def test_load_aigen_prefers_working(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()
    store = ConfigStore(paths, defaults_root=tmp_path / "defaults")

    write_file(paths.working_dir / "AIGen.yaml", "parameters:\n  denoise: 0.4\n")
    write_file(paths.config_dir / "AIGen.yaml", "aivis:\n  provider: ollama\n")

    config = store.load_aigen_config()
    assert config["parameters"]["denoise"] == 0.4
    assert config["aivis"]["provider"] == "ollama"
    assert (paths.working_dir / "AIGen.yaml").exists()


def test_load_aivis_defaults(tmp_path):
    paths = ProjectPaths("demo", projects_root=tmp_path)
    paths.ensure_project_directories()
    defaults_root = tmp_path / "defaults"
    write_file(defaults_root / "config" / "AIVis.yaml", "provider: openrouter\nmodel: test-model\n")

    store = ConfigStore(paths, defaults_root=defaults_root)
    cfg = store.load_aivis_config()
    assert cfg["provider"] == "openrouter"
    assert cfg["model"] == "test-model"
