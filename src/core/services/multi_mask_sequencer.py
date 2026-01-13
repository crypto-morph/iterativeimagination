"""Multi-mask sequencing logic for sequential mask processing."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple


class MaskSequenceItem(NamedTuple):
    """A mask in the sequence with its associated criterion."""
    name: str
    criterion: str


class MultiMaskSequencer:
    """Handles sequential processing of multiple masks."""

    def __init__(self, logger, project, rules: Dict):
        self.logger = logger
        self.project = project
        self.rules = rules

    def detect_mask_sequence(self) -> List[MaskSequenceItem]:
        """Detect masks and their associated criteria for sequential mask cycling.
        
        Returns:
            List of MaskSequenceItem tuples, empty if no multi-mask sequence detected
        """
        mask_sequence = []
        masking_rules = self.rules.get("masking") if isinstance(self.rules, dict) else None
        masks = (
            (masking_rules.get("masks") if isinstance(masking_rules, dict) else None)
            if masking_rules
            else None
        )

        if isinstance(masks, list):
            for m in masks:
                if not isinstance(m, dict):
                    continue
                mask_name = str(m.get("name") or "").strip()
                if mask_name and mask_name != "default":
                    # Find the criterion field for this mask (e.g., left_outfit for left mask)
                    active_criteria = m.get("active_criteria") or []
                    mask_criterion = None
                    for crit_field in active_criteria:
                        # Look for outfit criteria (left_outfit, middle_outfit, right_outfit)
                        if isinstance(crit_field, str) and "_outfit" in crit_field:
                            mask_criterion = crit_field
                            break
                    if mask_criterion:
                        mask_sequence.append(
                            MaskSequenceItem(name=mask_name, criterion=mask_criterion)
                        )

        return mask_sequence

    def prepare_mask_iteration(
        self, mask_item: MaskSequenceItem, inpainting_boost_config: Dict
    ) -> None:
        """Prepare AIGen.yaml for a specific mask iteration.
        
        Sets active_mask and applies inpainting parameter boosts.
        """
        aigen_config = self.project.load_aigen_config()
        aigen_config.setdefault("masking", {})
        aigen_config["masking"]["enabled"] = True
        aigen_config["masking"]["active_mask"] = mask_item.name

        # Boost denoise/cfg for inpainting
        params = aigen_config.get("parameters", {})
        current_denoise = float(params.get("denoise", 0.5))
        current_cfg = float(params.get("cfg", 7.0))

        if current_denoise < inpainting_boost_config["denoise_threshold"]:
            params["denoise"] = inpainting_boost_config["denoise_min"]
            self.logger.info(
                f"Boosted denoise to {params['denoise']} for inpainting "
                f"(mask: {mask_item.name}, from rules.yaml)"
            )

        if current_cfg < inpainting_boost_config["cfg_threshold"]:
            params["cfg"] = inpainting_boost_config["cfg_min"]
            self.logger.info(
                f"Boosted cfg to {params['cfg']} for inpainting "
                f"(mask: {mask_item.name}, from rules.yaml)"
            )

        aigen_config["parameters"] = params
        self.project.save_aigen_config(aigen_config)

    def update_progress_image(self, iteration: int, run_id: Optional[str]) -> None:
        """Update input/progress.png with the result from the given iteration."""
        try:
            progress_path = self.project.project_root / "input" / "progress.png"
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            iteration_paths = self.project.get_iteration_paths(iteration, run_id=run_id)
            shutil.copy2(iteration_paths["image"], progress_path)
            self.logger.info(f"Updated progress image for next mask: {progress_path}")
        except Exception:
            pass
