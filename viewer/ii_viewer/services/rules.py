"""Rules.yaml loading helpers for viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_rules(project_dir: Path) -> Dict[str, Any]:
    rules_path = project_dir / "config" / "rules.yaml"
    if not rules_path.exists():
        return {}
    try:
        return yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
