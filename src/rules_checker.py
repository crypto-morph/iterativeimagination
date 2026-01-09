#!/usr/bin/env python3
"""
rules_checker.py

Lint + AI-assisted tag suggestions for `rules.yaml` acceptance_criteria.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure repo src/ is importable when executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aivis_client import AIVisClient  # noqa: E402
from project_manager import ProjectManager  # noqa: E402


VALID_INTENTS = {"change", "preserve"}
VALID_STRENGTH = {"low", "medium", "high"}
VALID_TYPES = {"boolean", "number", "integer", "string", "array"}

ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RESET = "\x1b[0m"


def _listify(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return [str(v)]


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def lint_rules(rules: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    proj = rules.get("project") or {}
    if not isinstance(proj, dict):
        errors.append("`project` must be a mapping.")
    else:
        if not proj.get("name"):
            warnings.append("`project.name` is missing.")
        if "max_iterations" in proj and not isinstance(proj.get("max_iterations"), int):
            warnings.append("`project.max_iterations` should be an integer.")

    criteria = rules.get("acceptance_criteria")
    if not isinstance(criteria, list) or not criteria:
        errors.append("`acceptance_criteria` must be a non-empty list.")
        return errors, warnings

    seen_fields = set()
    for i, c in enumerate(criteria, 1):
        if not isinstance(c, dict):
            errors.append(f"acceptance_criteria[{i}] must be a mapping.")
            continue
        field = c.get("field")
        if not field:
            errors.append(f"acceptance_criteria[{i}].field is missing.")
            continue
        if field in seen_fields:
            errors.append(f"Duplicate acceptance_criteria field: {field}")
        seen_fields.add(field)

        if not c.get("question"):
            warnings.append(f"{field}: missing `question`.")

        ctype = (c.get("type") or "").strip().lower()
        if ctype and ctype not in VALID_TYPES:
            warnings.append(f"{field}: unknown type '{c.get('type')}'. Expected one of: {sorted(VALID_TYPES)}")

        intent = (c.get("intent") or "").strip().lower()
        if intent and intent not in VALID_INTENTS:
            warnings.append(f"{field}: invalid intent '{c.get('intent')}'. Expected: {sorted(VALID_INTENTS)}")

        strength = (c.get("edit_strength") or "").strip().lower()
        if strength and strength not in VALID_STRENGTH:
            warnings.append(f"{field}: invalid edit_strength '{c.get('edit_strength')}'. Expected: {sorted(VALID_STRENGTH)}")

        for k in ("must_include", "ban_terms", "avoid_terms"):
            v = c.get(k)
            if v is not None and not isinstance(v, list):
                warnings.append(f"{field}: `{k}` should be a list (got {type(v).__name__}).")

    return errors, warnings


def load_guidelines_text(
    repo_root: Path,
    project_name: Optional[str],
    guidelines_path: Optional[str],
    no_guidelines: bool,
) -> str:
    """Load guidelines text that will be sent to AIVis.

    Precedence:
    - --no-guidelines: disables all guidelines (empty string)
    - --guidelines <path>: explicit file
    - projects/<project>/config/rules_guidelines.md: per-project override
    - docs/rules_guidelines.md: repo default
    """
    if no_guidelines:
        return ""

    if guidelines_path:
        p = Path(guidelines_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        print(f"⚠ Guidelines file not found: {p}", file=sys.stderr)

    if project_name:
        p = repo_root / "projects" / project_name / "config" / "rules_guidelines.md"
        if p.exists():
            return p.read_text(encoding="utf-8")

    p = repo_root / "docs" / "rules_guidelines.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def build_aivis_client(
    project_name: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
) -> Tuple[AIVisClient, Path]:
    prompts_path: Optional[Path] = None
    aivis_config: Dict[str, Any] = {}

    if project_name:
        pm = ProjectManager(project_name)
        aivis_config = pm.load_aivis_config()
        pp = pm.project_root / "config" / "prompts.yaml"
        prompts_path = pp if pp.exists() else (Path("defaults") / "prompts.yaml")
    else:
        aivis_config = load_yaml(Path("defaults") / "config" / "AIVis.yaml")
        prompts_path = Path("defaults") / "prompts.yaml"

    prov = (provider or aivis_config.get("provider") or "ollama").strip().lower()
    mdl = (model or aivis_config.get("model") or "qwen3-vl:4b").strip()

    # OpenRouter needs an API key. If absent, fall back to the configured fallback provider/model.
    effective_api_key = api_key or aivis_config.get("api_key")
    if prov == "openrouter":
        import os

        if not (effective_api_key or os.environ.get("OPENROUTER_API_KEY")):
            fb_provider = (aivis_config.get("fallback_provider") or "ollama").strip().lower()
            fb_model = (aivis_config.get("fallback_model") or "llava-phi3:latest").strip()
            print(
                "⚠ OpenRouter selected but no API key found (OPENROUTER_API_KEY or AIVis.yaml api_key). "
                f"Falling back to {fb_provider}:{fb_model}.",
                file=sys.stderr,
            )
            prov = fb_provider
            mdl = fb_model

    client = AIVisClient(
        provider=prov,
        model=mdl,
        prompts_path=prompts_path,
        max_concurrent=int(aivis_config.get("max_concurrent", 1) or 1),
        api_key=effective_api_key,
        fallback_provider=aivis_config.get("fallback_provider"),
        fallback_model=aivis_config.get("fallback_model"),
    )
    return client, prompts_path


def apply_tag_suggestions(rules: Dict[str, Any], suggestions: Dict[str, Any], overwrite: bool) -> int:
    criteria = rules.get("acceptance_criteria") or []
    by_field = {c.get("field"): c for c in criteria if isinstance(c, dict) and c.get("field")}

    changed = 0
    for s in (suggestions.get("acceptance_criteria") or []):
        if not isinstance(s, dict):
            continue
        field = s.get("field")
        if not field or field not in by_field:
            continue
        c = by_field[field]

        for k in ("intent", "edit_strength"):
            if overwrite or not c.get(k):
                if s.get(k):
                    c[k] = str(s.get(k)).strip().lower()
                    changed += 1

        for k in ("must_include", "ban_terms", "avoid_terms"):
            incoming = _listify(s.get(k))
            if overwrite or c.get(k) is None:
                if incoming:
                    c[k] = incoming
                    changed += 1
            else:
                # Merge + de-dup
                existing = _listify(c.get(k))
                merged = []
                seen = set()
                for x in existing + incoming:
                    xx = x.strip()
                    if not xx or xx.lower() in seen:
                        continue
                    seen.add(xx.lower())
                    merged.append(xx)
                if merged != existing:
                    c[k] = merged
                    changed += 1

    return changed


def _normalise_list(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        xx = str(x).strip()
        if not xx:
            continue
        k = xx.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(xx)
    return out


def _colour(s: str, colour: str, enabled: bool) -> str:
    if not enabled:
        return s
    return f"{colour}{s}{ANSI_RESET}"


def _diff_lists(old: List[str], new: List[str]) -> Tuple[List[str], List[str]]:
    """Return (removed, added) items comparing case-insensitively."""
    old_n = _normalise_list(old)
    new_n = _normalise_list(new)
    old_set = {x.lower() for x in old_n}
    new_set = {x.lower() for x in new_n}
    removed = [x for x in old_n if x.lower() not in new_set]
    added = [x for x in new_n if x.lower() not in old_set]
    return removed, added


def print_suggestions_diff(
    rules: Dict[str, Any],
    suggestions: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
    colour: bool,
    show_general_notes: bool,
) -> None:
    criteria = rules.get("acceptance_criteria") or []
    by_field = {c.get("field"): c for c in criteria if isinstance(c, dict) and c.get("field")}

    provider = (metadata or {}).get("provider")
    model = (metadata or {}).get("model")
    using_fallback = (metadata or {}).get("using_fallback")
    if provider or model:
        print(f"AIVis: provider={provider} model={model} fallback={using_fallback}")

    any_changes = False
    for s in suggestions.get("acceptance_criteria", []) or []:
        if not isinstance(s, dict):
            continue
        field = s.get("field")
        if not field:
            continue
        current = by_field.get(field) or {}

        changes: List[str] = []

        # intent/edit_strength are scalars
        cur_intent = (current.get("intent") or "").strip().lower()
        sug_intent = (s.get("intent") or "").strip().lower()
        if sug_intent and sug_intent != cur_intent:
            changes.append(
                f"intent: {_colour(cur_intent or '∅', ANSI_RED, colour)} → {_colour(sug_intent, ANSI_GREEN, colour)}"
            )

        cur_strength = (current.get("edit_strength") or "").strip().lower()
        sug_strength = (s.get("edit_strength") or "").strip().lower()
        if sug_strength and sug_strength != cur_strength:
            changes.append(
                f"edit_strength: {_colour(cur_strength or '∅', ANSI_RED, colour)} → {_colour(sug_strength, ANSI_GREEN, colour)}"
            )

        # list tags
        for k in ("must_include", "ban_terms", "avoid_terms"):
            cur_list = _listify(current.get(k))
            sug_list = _listify(s.get(k))
            removed, added = _diff_lists(cur_list, sug_list)
            if removed or added:
                parts: List[str] = []
                if removed:
                    parts.append(_colour(f"-{removed}", ANSI_RED, colour))
                if added:
                    parts.append(_colour(f"+{added}", ANSI_GREEN, colour))
                changes.append(f"{k}: " + " ".join(parts))

        if changes:
            any_changes = True
            print(f"- {field}:")
            for line in changes:
                print(f"    {line}")
            notes = s.get("notes")
            if notes:
                print(f"    notes: {notes}")
        else:
            print(f"- {field}: {_colour('no changes suggested', ANSI_YELLOW, colour)}")

    if show_general_notes:
        for n in suggestions.get("general_notes", []) or []:
            print(f"NOTE: {n}")

    if not any_changes:
        print("No tag changes suggested.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint and AI-suggest acceptance_criteria tags for rules.yaml")
    parser.add_argument("--project", help="Project name (uses projects/<name>/config/rules.yaml)")
    parser.add_argument("--rules", help="Path to rules.yaml (overrides --project)")
    parser.add_argument("--suggest", action="store_true", help="Use AIVis to suggest better tags for acceptance_criteria")
    parser.add_argument("--apply", action="store_true", help="Apply suggested tags back into rules.yaml")
    parser.add_argument("--overwrite-tags", action="store_true", help="Overwrite existing tags instead of only filling/merging")
    parser.add_argument("--provider", help="Override AIVis provider (ollama/openrouter)")
    parser.add_argument("--model", help="Override AIVis model")
    parser.add_argument("--api-key", dest="api_key", help="OpenRouter API key override (otherwise uses OPENROUTER_API_KEY)")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Emit AI suggestions JSON to stdout")
    parser.add_argument("--no-colour", action="store_true", help="Disable ANSI colours in suggestion output")
    parser.add_argument("--no-general-notes", action="store_true", help="Do not print AI general_notes (useful for restricted projects)")
    parser.add_argument("--guidelines", help="Path to a guidelines markdown file to send to AIVis")
    parser.add_argument("--no-guidelines", action="store_true", help="Send no guidelines to AIVis (project will rely on prompt alone)")
    args = parser.parse_args()

    repo_root = REPO_ROOT

    if args.rules:
        rules_path = Path(args.rules)
        project_name = args.project
    elif args.project:
        rules_path = repo_root / "projects" / args.project / "config" / "rules.yaml"
        project_name = args.project
    else:
        parser.error("Provide either --project or --rules")
        return 2

    rules = load_yaml(rules_path)
    errors, warnings = lint_rules(rules)
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"- {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")

    suggestions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    if args.suggest:
        # For restricted projects, default to suppressing general notes (they often include generic content-policy reminders)
        restricted_default = bool(project_name and project_name.lower().startswith("restricted"))
        show_general_notes = not (args.no_general_notes or restricted_default)

        guidelines = load_guidelines_text(
            repo_root=repo_root,
            project_name=project_name,
            guidelines_path=args.guidelines,
            no_guidelines=args.no_guidelines,
        )
        aivis, _prompts_path = build_aivis_client(project_name, args.provider, args.model, args.api_key)
        rules_yaml_text = rules_path.read_text(encoding="utf-8")
        suggestions, metadata = aivis.suggest_rules_tags(rules_yaml_text, guidelines)
        if args.json_out:
            print(json.dumps({"suggestions": suggestions, "metadata": metadata}, indent=2))
        else:
            colour = (sys.stdout.isatty() and not args.no_colour)
            print_suggestions_diff(rules, suggestions, metadata, colour=colour, show_general_notes=show_general_notes)

    if args.apply:
        if not suggestions:
            print("Nothing to apply (run with --suggest first).")
            return 1
        changed = apply_tag_suggestions(rules, suggestions, overwrite=args.overwrite_tags)
        save_yaml(rules_path, rules)
        print(f"Applied {changed} tag updates to {rules_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

