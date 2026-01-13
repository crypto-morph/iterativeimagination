"""Configuration loading and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from .project_paths import ProjectPaths


class ConfigStore:
    """Handles project configuration files (rules, AIGen, AIVis)."""

    def __init__(self, paths: ProjectPaths, defaults_root: Path | str = "defaults"):
        self.paths = paths
        self.defaults_root = Path(defaults_root)

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------
    def load_rules(self) -> Dict:
        rules_path = self.paths.config_file("rules.yaml")
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_path}")
        with open(rules_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ------------------------------------------------------------------
    # AIGen
    # ------------------------------------------------------------------
    def load_aigen_config(self) -> Dict:
        """Load working/config/default AIGen.yaml and persist merged result to working."""
        working_aigen = self.paths.working_dir / "AIGen.yaml"
        project_aigen = self.paths.config_file("AIGen.yaml")
        defaults_aigen = self.defaults_root / "config" / "AIGen.yaml"

        config: Dict = {}
        if working_aigen.exists():
            with open(working_aigen, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}

        if project_aigen.exists():
            with open(project_aigen, "r", encoding="utf-8") as fh:
                project_config = yaml.safe_load(fh) or {}
            if "aivis" in project_config:
                config["aivis"] = project_config["aivis"]

        if not working_aigen.exists() and not project_aigen.exists():
            if defaults_aigen.exists():
                with open(defaults_aigen, "r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh) or {}
            else:
                raise FileNotFoundError("No AIGen.yaml found in project, config, or defaults")

        with open(working_aigen, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)
        return config

    def save_aigen_config(self, config: Dict) -> None:
        working_aigen = self.paths.working_dir / "AIGen.yaml"
        with open(working_aigen, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False)

    # ------------------------------------------------------------------
    # AIVis
    # ------------------------------------------------------------------
    def load_aivis_config(self) -> Dict:
        project_aivis = self.paths.config_file("AIVis.yaml")
        defaults_aivis = self.defaults_root / "config" / "AIVis.yaml"

        if project_aivis.exists():
            with open(project_aivis, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

        if defaults_aivis.exists():
            with open(defaults_aivis, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}

        return {
            "provider": "ollama",
            "model": "qwen3-vl:4b",
            "fallback_provider": "ollama",
            "fallback_model": "llava-phi3:latest",
        }
