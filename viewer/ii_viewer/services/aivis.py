"""AIVis wiring for the viewer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from ..settings import REPO_ROOT


def load_aivis_config(project_dir: Path) -> Dict[str, Any]:
    aivis_config_path = project_dir / "config" / "AIVis.yaml"
    if not aivis_config_path.exists():
        aivis_config_path = REPO_ROOT / "defaults" / "config" / "AIVis.yaml"
    with open(aivis_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_aivis_client(project_dir: Path):
    # Ensure repo src/ is importable
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from aivis_client import AIVisClient  # noqa: E402

    aivis_config = load_aivis_config(project_dir)

    prompts_path = project_dir / "config" / "prompts.yaml"
    if not prompts_path.exists():
        prompts_path = REPO_ROOT / "defaults" / "prompts.yaml"

    provider = str(aivis_config.get("provider") or "ollama")
    model = str(aivis_config.get("model") or "qwen3-vl:4b")
    api_key = aivis_config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    fallback_provider = aivis_config.get("fallback_provider")
    fallback_model = aivis_config.get("fallback_model")
    max_concurrent = int(aivis_config.get("max_concurrent", 1) or 1)
    base_url = aivis_config.get("base_url")

    return AIVisClient(
        model=model,
        provider=provider,
        api_key=api_key,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        prompts_path=prompts_path,
        max_concurrent=max_concurrent,
        base_url=base_url,
    )
