"""Prompt term parsing helpers used by the viewer."""

from __future__ import annotations

from typing import Dict, List


def parse_prompt_to_terms(prompt_str: str) -> List[str]:
    if not prompt_str:
        return []
    return [t.strip() for t in prompt_str.split(",") if t.strip()]


def compare_term_lists(current: List[str], suggested: List[str]) -> Dict:
    current_lower = {t.lower(): t for t in current}
    suggested_lower = {t.lower(): t for t in suggested}

    to_add = [suggested_lower[t] for t in suggested_lower if t not in current_lower]
    to_remove = [current_lower[t] for t in current_lower if t not in suggested_lower]
    unchanged = [current_lower[t] for t in current_lower if t in suggested_lower]

    return {"to_add": to_add, "to_remove": to_remove, "unchanged": unchanged}
