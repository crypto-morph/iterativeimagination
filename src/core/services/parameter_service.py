"""Parameter tuning helpers for iterative improvements."""

from __future__ import annotations

from typing import Dict, List, Optional


class ParameterUpdateService:
    """Encapsulates heuristics for adjusting denoise/CFG/etc."""

    def __init__(self, logger, rules: Dict):
        self.logger = logger
        self.rules = rules or {}

    def update_parameters(
        self,
        aigen_config: Dict,
        iteration_result: Dict,
        evaluation: Dict,
        comparison: Dict,
        criteria_defs: List[Dict],
        criteria_by_field: Dict[str, Dict],
        failed_criteria: List[str],
    ) -> None:
        """Mutate `aigen_config` parameters in-place based on evaluation results."""
        if not aigen_config:
            return

        params = aigen_config.setdefault("parameters", {})
        project_cfg = self.rules.get("project") if isinstance(self.rules, dict) else {}
        similarity = comparison.get("similarity_score", 0.5)
        mask_used = bool(iteration_result.get("mask_used"))

        cur_d = float(params.get("denoise", 0.5) or 0.5)
        cur_cfg = float(params.get("cfg", 7.0) or 7.0)

        denoise_min = _float(project_cfg.get("denoise_min"), 0.20)
        denoise_max = _float(project_cfg.get("denoise_max"), 0.80)
        cfg_min = _float(project_cfg.get("cfg_min"), 4.0)
        cfg_max = _float(project_cfg.get("cfg_max"), 12.0)

        if not mask_used:
            denoise_max = min(denoise_max, _float(project_cfg.get("denoise_max_no_mask"), 0.55))
            cfg_max = min(cfg_max, _float(project_cfg.get("cfg_max_no_mask"), 9.0))

        failed_change, failed_preserve = _partition_failed(criteria_by_field, failed_criteria)
        preserve_heavy = _is_preserve_heavy(criteria_defs, project_cfg)
        if preserve_heavy and failed_preserve and not mask_used:
            denoise_max = min(denoise_max, 0.62)
            cfg_max = min(cfg_max, 8.5)

        if failed_change:
            self._handle_change_failures(
                params=params,
                current_denoise=cur_d,
                current_cfg=cur_cfg,
                denoise_min=denoise_min,
                denoise_max=denoise_max,
                cfg_min=cfg_min,
                cfg_max=cfg_max,
                failed_fields=failed_change,
                criteria_by_field=criteria_by_field,
                mask_used=mask_used,
                iteration_result=iteration_result,
            )
        elif failed_preserve:
            self._handle_preserve_failures(
                params=params,
                current_denoise=cur_d,
                current_cfg=cur_cfg,
                denoise_min=denoise_min,
                cfg_min=cfg_min,
                failed_fields=failed_preserve,
                criteria_by_field=criteria_by_field,
            )

        if similarity <= 0.30:
            params["denoise"] = max(denoise_min, float(params.get("denoise", cur_d)) - 0.05)
            self.logger.info(
                "Images too different (similarity=%.2f) - reducing denoise to %.2f",
                similarity,
                params["denoise"],
            )

    def _handle_change_failures(
        self,
        params: Dict,
        current_denoise: float,
        current_cfg: float,
        denoise_min: float,
        denoise_max: float,
        cfg_min: float,
        cfg_max: float,
        failed_fields: List[str],
        criteria_by_field: Dict[str, Dict],
        mask_used: bool,
        iteration_result: Dict,
    ) -> None:
        if not mask_used:
            self.logger.warning(
                "Change goals are failing but no mask is in use. "
                "Add projects/<name>/input/mask.png for precise edits; "
                "keeping conservative no-mask caps (denoise<=%.2f, cfg<=%.1f).",
                denoise_max,
                cfg_max,
            )

        max_strength = max(_strength_value(criteria_by_field[f].get("edit_strength")) for f in failed_fields)
        iteration_num = iteration_result.get("iteration", 1)
        consecutive_failures = max(0, int(iteration_num) - 1)

        base_delta = 0.06 + 0.02 * max_strength
        base_cfg_delta = 0.5 + 0.5 * max_strength
        failure_multiplier = min(2.0, 1.0 + (consecutive_failures * 0.3))
        delta = base_delta * failure_multiplier
        cfg_delta = base_cfg_delta * failure_multiplier

        cur_d = float(params.get("denoise", current_denoise))
        cur_cfg = float(params.get("cfg", current_cfg))
        if cur_d >= 0.85 and cur_cfg >= 12.0:
            delta = max(delta, 0.05)
            cfg_delta = max(cfg_delta, 1.5)
            self.logger.warning(
                "High denoise/cfg (%.2f/%.1f) but change still failing. "
                "Trying larger increments (Δdenoise=%.3f, Δcfg=%.1f).",
                cur_d,
                cur_cfg,
                delta,
                cfg_delta,
            )

        params["denoise"] = min(denoise_max, max(denoise_min, cur_d + delta))
        params["cfg"] = min(cfg_max, max(cfg_min, cur_cfg + cfg_delta))
        self.logger.info(
            "Change goals failing (%s, %d consecutive) - increasing denoise to %.2f (Δ%.3f), cfg to %.1f (Δ%.1f).",
            failed_fields,
            consecutive_failures,
            params["denoise"],
            delta,
            params["cfg"],
            cfg_delta,
        )

    def _handle_preserve_failures(
        self,
        params: Dict,
        current_denoise: float,
        current_cfg: float,
        denoise_min: float,
        cfg_min: float,
        failed_fields: List[str],
        criteria_by_field: Dict[str, Dict],
    ) -> None:
        max_strength = max(_strength_value(criteria_by_field[f].get("edit_strength")) for f in failed_fields)
        delta = 0.05 + 0.02 * max_strength
        params["denoise"] = max(denoise_min, float(params.get("denoise", current_denoise)) - delta)
        params["cfg"] = max(cfg_min, float(params.get("cfg", current_cfg)) - (0.5 + 0.5 * max_strength))
        self.logger.info(
            "Preserve goals failing (%s) - decreasing denoise to %.2f, cfg to %.1f",
            failed_fields,
            params["denoise"],
            params["cfg"],
        )


def _float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def _strength_value(raw: Optional[str]) -> float:
    val = (raw or "medium").strip().lower()
    if val in {"low", "l"}:
        return 0.5
    if val in {"high", "h"}:
        return 1.5
    return 1.0


def _partition_failed(
    criteria_by_field: Dict[str, Dict],
    failed_fields: List[str],
) -> (List[str], List[str]):
    failed_change: List[str] = []
    failed_preserve: List[str] = []
    for field in failed_fields:
        crit = criteria_by_field.get(field, {})
        intent = (crit.get("intent") or "preserve").strip().lower()
        if intent == "change":
            failed_change.append(field)
        else:
            failed_preserve.append(field)
    return failed_change, failed_preserve


def _is_preserve_heavy(criteria_defs: List[Dict], project_cfg: Dict) -> bool:
    if bool(project_cfg.get("preserve_heavy")):
        return True
    intents = [
        (c.get("intent") or "preserve").strip().lower()
        for c in criteria_defs
        if isinstance(c, dict)
    ]
    n_preserve = sum(1 for x in intents if x != "change")
    n_change = sum(1 for x in intents if x == "change")
    return n_preserve >= 3 and n_change <= 2
