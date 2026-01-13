import textwrap
from pathlib import Path

from core.config.prompt_library import load_prompt_templates


def write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_load_prompt_templates_layers(tmp_path, monkeypatch):
    defaults = tmp_path / "defaults" / "prompts.yaml"
    repo_prompts = tmp_path / "prompts.yaml"
    project_prompts = tmp_path / "projects" / "demo" / "config" / "prompts.yaml"

    write_yaml(defaults, "ask_question: default-version\n")
    write_yaml(repo_prompts, "ask_question: repo-version\ncompare_images: repo-cmp\n")
    write_yaml(project_prompts, "improve_prompts: project-version\n")

    # Monkeypatch Path() resolution inside prompt library to use tmp_root
    monkeypatch.chdir(tmp_path)

    result = load_prompt_templates(prompts_path=project_prompts, defaults_root=tmp_path / "defaults", repo_root=tmp_path)

    assert result["ask_question"] == "repo-version"
    assert result["compare_images"] == "repo-cmp"
    assert result["improve_prompts"] == "project-version"
