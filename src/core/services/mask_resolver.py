"""Mask resolution and preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, NamedTuple

from core.constants import (
    MASK_WHITE_THRESHOLD,
    MASK_WHITE_PIXEL_VALUE,
    MASK_COVERAGE_WARNING_THRESHOLD,
    INPAINTING_DENOISE_MIN,
    INPAINTING_CFG_MIN,
    INPAINTING_DENOISE_THRESHOLD,
    INPAINTING_CFG_THRESHOLD,
)


class MaskInfo(NamedTuple):
    """Information about a resolved mask."""
    filename: Optional[str]  # Filename in ComfyUI input directory
    active_mask_name: Optional[str]  # Name of the active mask
    path: Optional[Path]  # Original mask file path
    coverage: Optional[float]  # Percentage coverage (if analysed)


class MaskResolver:
    """Handles mask resolution, validation, and parameter boosting."""

    def __init__(self, logger, project_root: Path, comfyui_input_dir: Path, prepare_image_fn):
        self.logger = logger
        self.project_root = project_root
        self.comfyui_input_dir = comfyui_input_dir
        self.prepare_image_fn = prepare_image_fn

    def resolve_mask(
        self,
        aigen_config: Dict,
        inpainting_boost_config: Optional[Dict] = None,
    ) -> MaskInfo:
        """Resolve and prepare the active mask for an iteration.
        
        Args:
            aigen_config: The AIGen configuration dict
            inpainting_boost_config: Optional boost config dict with keys:
                denoise_min, cfg_min, denoise_threshold, cfg_threshold
                
        Returns:
            MaskInfo with resolved mask details
        """
        mask_cfg = aigen_config.get("masking") or {}
        try:
            mask_enabled = bool(mask_cfg.get("enabled", True))
        except Exception:
            mask_enabled = True

        if not mask_enabled:
            return MaskInfo(filename=None, active_mask_name=None, path=None, coverage=None)

        # Resolve active mask name
        raw_active = mask_cfg.get("active_mask") or mask_cfg.get("active")
        active_mask_name: Optional[str] = None
        if isinstance(raw_active, str) and raw_active.strip():
            active_mask_name = raw_active.strip()

        # Resolve mask path
        resolved_mask_path: Optional[Path] = None
        masks = mask_cfg.get("masks")
        
        if masks:
            resolved_mask_path, active_mask_name = self._resolve_from_masks_config(
                masks, active_mask_name
            )

        # Fallback: try input/masks/{active_mask_name}.png
        if resolved_mask_path is None and active_mask_name:
            try:
                named_mask = self.project_root / "input" / "masks" / f"{active_mask_name}.png"
                if named_mask.exists():
                    resolved_mask_path = named_mask
                    self.logger.info(f"Found mask file: input/masks/{active_mask_name}.png")
            except Exception:
                pass

        # Fallback to legacy mask.png
        if resolved_mask_path is None:
            try:
                legacy = self.project_root / "input" / "mask.png"
                if legacy.exists():
                    resolved_mask_path = legacy
                    if not active_mask_name:
                        active_mask_name = "default"
            except Exception:
                resolved_mask_path = None

        if resolved_mask_path is None:
            return MaskInfo(filename=None, active_mask_name=None, path=None, coverage=None)

        # Check if mask is all white
        if self._mask_is_all_white(resolved_mask_path):
            self.logger.warning(
                f"Mask appears all-white (>{MASK_WHITE_THRESHOLD*100:.0f}% white pixels); "
                "treating as no-mask (non-inpaint) run."
            )
            return MaskInfo(filename=None, active_mask_name=None, path=None, coverage=None)

        # Prepare mask for ComfyUI
        try:
            mask_filename = self.prepare_image_fn(resolved_mask_path, self.comfyui_input_dir)
        except Exception as e:
            self.logger.warning(f"Failed to prepare mask image: {e}")
            return MaskInfo(filename=None, active_mask_name=None, path=None, coverage=None)

        # Analyse mask coverage
        coverage = self._analyse_mask_coverage(resolved_mask_path, active_mask_name)

        # Apply inpainting parameter boosts
        self._apply_inpainting_boosts(
            aigen_config, active_mask_name, inpainting_boost_config
        )

        self.logger.info(
            f"Using mask: {active_mask_name or 'default'} from file: {resolved_mask_path}"
        )

        return MaskInfo(
            filename=mask_filename,
            active_mask_name=active_mask_name,
            path=resolved_mask_path,
            coverage=coverage,
        )

    def _resolve_from_masks_config(
        self, masks: Dict | list, active_mask_name: Optional[str]
    ) -> tuple[Optional[Path], Optional[str]]:
        """Resolve mask path from masks configuration (list or dict form)."""
        resolved_mask_path: Optional[Path] = None
        resolved_name: Optional[str] = active_mask_name

        # List form: [{name, file}, ...]
        if isinstance(masks, list):
            for m in masks:
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name") or "").strip()
                file_ = str(m.get("file") or "").strip()
                if not name or not file_:
                    continue
                if active_mask_name and name != active_mask_name:
                    continue
                candidate = Path(file_)
                if not candidate.is_absolute():
                    candidate = (self.project_root / candidate).resolve()
                if candidate.exists():
                    resolved_mask_path = candidate
                    resolved_name = name
                    break

        # Dict form: {name: file, ...}
        elif isinstance(masks, dict):
            if active_mask_name and active_mask_name in masks:
                candidate = Path(str(masks.get(active_mask_name)))
                if not candidate.is_absolute():
                    candidate = (self.project_root / candidate).resolve()
                if candidate.exists():
                    resolved_mask_path = candidate
            else:
                # Pick first existing mask in dict (stable order in YAML)
                for name, file_ in masks.items():
                    n = str(name).strip()
                    f = str(file_).strip()
                    if not n or not f:
                        continue
                    candidate = Path(f)
                    if not candidate.is_absolute():
                        candidate = (self.project_root / candidate).resolve()
                    if candidate.exists():
                        resolved_mask_path = candidate
                        resolved_name = n
                        break

        return resolved_mask_path, resolved_name

    def _mask_is_all_white(self, mask_path: Path) -> bool:
        """Check if a mask image is all white (or nearly all white).
        
        Returns True if >95% of pixels are white (>= 240 in grayscale).
        """
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(mask_path).convert("L")
            arr = np.array(img)

            # Count pixels that are white (>= 240 to account for slight variations)
            white_pixels = np.sum(arr >= MASK_WHITE_PIXEL_VALUE)
            total_pixels = arr.size

            if total_pixels == 0:
                return False

            white_ratio = white_pixels / total_pixels
            return white_ratio >= MASK_WHITE_THRESHOLD
        except Exception as e:
            self.logger.warning(f"Could not check if mask is all-white: {e}")
            return False

    def _analyse_mask_coverage(
        self, mask_path: Path, mask_name: Optional[str]
    ) -> Optional[float]:
        """Analyse mask coverage and log warnings if needed."""
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(mask_path).convert("L")
            arr = np.array(img)
            white = np.sum(arr >= 128)  # Pixels that are white (editable)
            total = arr.size
            coverage = white / total * 100

            self.logger.info(
                f"Mask '{mask_name or 'default'}' coverage: {coverage:.1f}% white (editable), "
                f"{100-coverage:.1f}% black (preserved)"
            )

            if coverage > MASK_COVERAGE_WARNING_THRESHOLD:
                self.logger.warning(
                    f"Mask '{mask_name or 'default'}' covers {coverage:.1f}% of image - "
                    "this may affect multiple subjects, not just the target!"
                )

            return coverage
        except ImportError as e:
            self.logger.warning(
                f"Could not analyse mask coverage (PIL/numpy not available): {e}"
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"Could not analyse mask coverage: {e}", exc_info=True
            )
            return None

    def _apply_inpainting_boosts(
        self,
        aigen_config: Dict,
        active_mask_name: Optional[str],
        boost_config: Optional[Dict],
    ) -> None:
        """Apply denoise/cfg boosts for inpainting mode."""
        if boost_config is None:
            boost_config = {
                "denoise_min": INPAINTING_DENOISE_MIN,
                "cfg_min": INPAINTING_CFG_MIN,
                "denoise_threshold": INPAINTING_DENOISE_THRESHOLD,
                "cfg_threshold": INPAINTING_CFG_THRESHOLD,
            }

        params = aigen_config.get("parameters", {})
        current_denoise = float(params.get("denoise", 0.5))
        current_cfg = float(params.get("cfg", 7.0))

        # If denoise/cfg are at default or low values, boost them for inpainting
        if current_denoise < boost_config["denoise_threshold"]:
            params["denoise"] = boost_config["denoise_min"]
            self.logger.info(
                f"Boosted denoise to {params['denoise']} for inpainting "
                f"(mask: {active_mask_name or 'default'}, from rules.yaml)"
            )

        if current_cfg < boost_config["cfg_threshold"]:
            params["cfg"] = boost_config["cfg_min"]
            self.logger.info(
                f"Boosted cfg to {params['cfg']} for inpainting "
                f"(mask: {active_mask_name or 'default'}, from rules.yaml)"
            )

        aigen_config["parameters"] = params
