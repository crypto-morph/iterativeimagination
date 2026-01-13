"""Prompt update helpers."""

from __future__ import annotations

from typing import Dict, List, Optional


class PromptUpdateService:
    """Orchestrates prompt improvement decisions."""

    def __init__(
        self,
        logger,
        rules: Dict,
        prompt_improver,
        describe_original_image_fn,
    ) -> None:
        self.logger = logger
        self.rules = rules
        self.prompt_improver = prompt_improver
        self._describe_original_image = describe_original_image_fn

    def maybe_improve_prompts(
        self,
        aigen_config: Dict,
        evaluation: Dict,
        comparison: Dict,
        failed_criteria: List[str],
        criteria_defs: List[Dict],
        criteria_by_field: Dict[str, Dict],
    ) -> bool:
        """Improve prompts when allowed by project config."""
        if not failed_criteria:
            return False

        project_cfg = self.rules.get("project") if isinstance(self.rules, dict) else {}
        if not project_cfg or not project_cfg.get("improve_prompts", False):
            self.logger.info(
                "Prompt improvement disabled (project.improve_prompts=false). Keeping prompts unchanged."
            )
            return False

        self.logger.info("Improving prompts based on failed criteria: %s", failed_criteria)
        try:
            original_desc = self._describe_original_image()
        except Exception:  # pragma: no cover - defensive
            original_desc = None

        prompts_cfg = aigen_config.setdefault("prompts", {})
        current_positive = prompts_cfg.get("positive", "")
        current_negative = prompts_cfg.get("negative", "")

        improved_positive, improved_negative, diff_info = self.prompt_improver.improve_prompts(
            current_positive=current_positive,
            current_negative=current_negative,
            evaluation=evaluation,
            comparison=comparison,
            failed_criteria=failed_criteria,
            criteria_defs=criteria_defs,
            criteria_by_field=criteria_by_field,
            original_description=original_desc,
        )

        self._log_diff_info(diff_info)

        prompts_cfg["positive"] = improved_positive
        prompts_cfg["negative"] = improved_negative
        aigen_config["prompts"] = prompts_cfg
        return True

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_diff_info(self, diff_info: Dict) -> None:
        must_include_terms = diff_info.get("must_include_terms")
        if must_include_terms:
            self.logger.info("  Tag must_include terms: %s", must_include_terms)
        ban_terms = diff_info.get("ban_terms")
        if ban_terms:
            self.logger.info("  Tag ban_terms: %s", ban_terms)
        avoid_terms = diff_info.get("avoid_terms")
        if avoid_terms:
            self.logger.info("  Tag avoid_terms: %s", avoid_terms)

        pos_diff = diff_info.get("pos_diff", {})
        neg_diff = diff_info.get("neg_diff", {})
        if pos_diff.get("added") or pos_diff.get("removed"):
            self.logger.info(
                "  Positive prompt changes: +%s -%s",
                pos_diff.get("added"),
                pos_diff.get("removed"),
            )
        if neg_diff.get("added") or neg_diff.get("removed"):
            self.logger.info(
                "  Negative prompt changes: +%s -%s",
                neg_diff.get("added"),
                neg_diff.get("removed"),
            )
