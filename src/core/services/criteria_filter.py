"""Criteria filtering utilities for mask-scoped evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional


def filter_criteria_for_mask(
    criteria_defs: List[Dict],
    mask_name: Optional[str],
    rules: Dict,
) -> List[Dict]:
    """Filter acceptance criteria based on active mask.
    
    Supports two models:
    1. Membership model (preferred): rules.masking.masks[].active_criteria
    2. Legacy per-criterion scoping: acceptance_criteria[].applies_to_masks
    
    Args:
        criteria_defs: List of criterion definitions
        mask_name: Name of the active mask (None for global/unmasked)
        rules: Full rules.yaml dict
        
    Returns:
        Filtered list of criteria that apply to the active mask
    """
    # Membership model (preferred)
    masking_rules = rules.get("masking") if isinstance(rules, dict) else None
    masks = (
        (masking_rules.get("masks") if isinstance(masking_rules, dict) else None)
        if masking_rules
        else None
    )

    if isinstance(masks, list) and masks:
        if not mask_name:
            return criteria_defs

        active_fields: set[str] = set()
        for m in masks:
            if not isinstance(m, dict):
                continue
            if str(m.get("name") or "").strip() != str(mask_name).strip():
                continue
            ac = m.get("active_criteria") or []
            if isinstance(ac, list):
                active_fields = {str(x).strip() for x in ac if str(x).strip()}
            break

        if not active_fields:
            return []

        return [
            c
            for c in criteria_defs
            if isinstance(c, dict)
            and str(c.get("field") or "").strip() in active_fields
        ]

    # Legacy per-criterion scoping fallback
    if not mask_name:
        return criteria_defs

    out = []
    for c in criteria_defs:
        if not isinstance(c, dict):
            continue

        scopes = c.get("applies_to_masks") or c.get("mask_scope")
        if not scopes:
            out.append(c)
            continue

        if isinstance(scopes, str):
            scopes_list = [scopes]
        elif isinstance(scopes, list):
            scopes_list = scopes
        else:
            scopes_list = [str(scopes)]

        scopes_norm = [str(x).strip() for x in scopes_list if str(x).strip()]
        if mask_name in scopes_norm:
            out.append(c)

    return out
