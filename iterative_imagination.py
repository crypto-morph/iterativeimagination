#!/usr/bin/env python3
"""
Iterative Imagination - AI-powered image generation with iterative improvement

Uses AI vision (qwen3-vl:4b) and AI image generation (ComfyUI) to generate images
from a source image with particular attributes, iterating until acceptance criteria are met.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, List
from contextlib import suppress

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from comfyui_client import ComfyUIClient
from aivis_client import AIVisClient
from workflow_manager import WorkflowManager
from project_manager import ProjectManager
from prompt_improver import PromptImprover
from cli.runner import build_run_parser, run_iteration
from core.services.evaluation_service import EvaluationService
from core.services.prompt_service import PromptUpdateService
from core.services.parameter_service import ParameterUpdateService


class IterativeImagination:
    """Main class for iterative image generation and improvement."""
    
    def __init__(self, project_name: str, verbose: bool = False):
        self.project_name = project_name
        self.verbose = verbose
        
        # Initialize project manager
        self.project = ProjectManager(project_name)
        self.project.ensure_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.rules = self.project.load_rules()
        self.input_image_path = self.project.get_input_image_path()
        self.aigen_config = self.project.load_aigen_config()
        
        # Initialize clients
        comfyui_config = self.aigen_config.get('comfyui', {})
        self.comfyui = ComfyUIClient(
            host=comfyui_config.get('host', 'localhost'),
            port=comfyui_config.get('port', 8188)
        )
        # Load prompts from project or defaults
        prompts_path = self.project.project_root / "config" / "prompts.yaml"
        if not prompts_path.exists():
            prompts_path = Path("defaults") / "prompts.yaml"
            if not prompts_path.exists():
                prompts_path = None  # Use default prompts in code
        
        # Load AIVis configuration from AIVis.yaml (separate from AIGen.yaml)
        aivis_config = self.project.load_aivis_config()
        import os
        
        # Setup fallback provider (Ollama) for rate limit situations
        fallback_provider = aivis_config.get('fallback_provider')
        fallback_model = aivis_config.get('fallback_model', 'llava-phi3:latest')  # Default to smaller model
        
        # Auto-configure fallback if using OpenRouter
        if aivis_config.get('provider', 'ollama') == 'openrouter' and not fallback_provider:
            fallback_provider = 'ollama'
            fallback_model = aivis_config.get('fallback_model', 'llava-phi3:latest')
        
        self.aivis = AIVisClient(
            model=aivis_config.get('model', 'qwen3-vl:4b'),
            base_url=aivis_config.get('base_url'),
            prompts_path=prompts_path,
            max_concurrent=aivis_config.get('max_concurrent', 1),
            provider=aivis_config.get('provider', 'ollama'),
            api_key=aivis_config.get('api_key') or os.environ.get('OPENROUTER_API_KEY'),
            fallback_provider=fallback_provider,
            fallback_model=fallback_model
        )
        
        # Setup ComfyUI input directory
        self.comfyui_input_dir = self.project.find_comfyui_input_dir()
        
        # Cache for original image description
        self.original_description = None
        
        # Initialize services
        self.evaluator = EvaluationService(
            logger=self.logger,
            rules=self.rules,
            aivis=self.aivis,
            describe_original_image_fn=self._describe_original_image,
        )
        prompt_improver = PromptImprover(
            logger=self.logger,
            aivis=self.aivis,
            describe_original_image_fn=self._describe_original_image,
            human_feedback_context=getattr(self, 'human_feedback_context', '')
        )
        self.prompt_service = PromptUpdateService(
            logger=self.logger,
            rules=self.rules,
            prompt_improver=prompt_improver,
            describe_original_image_fn=self._describe_original_image,
        )
        self.parameter_service = ParameterUpdateService(logger=self.logger, rules=self.rules)
        
        # Track best iteration
        self.best_score = 0
        self.best_iteration = None
        # Current run id for grouping working artefacts
        self.run_id: Optional[str] = None
        # Optional human feedback context (from viewer ranking) to guide prompt improvements
        self.human_feedback_context: str = ""
        # Optional seed locking (useful for preserve-heavy tasks to reduce randomness)
        self.locked_seed: Optional[int] = None

    def _maybe_seed_from_human_ranking(
        self,
        seed_run: Optional[str],
        seed_mode: str = "rank1",
        dry_run: bool = False,
    ) -> None:
        """Optionally seed working/AIGen.yaml from a previous run's human ranking.

        - seed_run: RUN_ID or "latest"
        - seed_mode:
            - "rank1": use top-ranked iteration’s prompts/params
            - "top3"/"top5": average numeric params across top K; use rank1 prompts; combine notes

        This does NOT change the project input image (so evaluation/comparison remain consistent).
        """
        if not seed_run:
            return

        mode = (seed_mode or "rank1").strip().lower()
        if mode not in ("rank1", "top3", "top5"):
            self.logger.warning(f"Unknown seed_mode={seed_mode!r}; falling back to 'rank1'")
            mode = "rank1"

        run_id = seed_run.strip()
        if run_id.lower() == "latest":
            latest = self.project.latest_run_id_with_human_feedback()
            if not latest:
                self.logger.warning("seed_from_ranking=latest but no previous runs found")
                return
            run_id = latest

        ranking = self.project.load_human_ranking(run_id)
        if not ranking:
            self.logger.warning(f"No human ranking found for run_id={run_id}")
            return

        order = ranking.get("ranking") or []
        if not isinstance(order, list) or not order:
            self.logger.warning(f"Human ranking for run_id={run_id} has no 'ranking' list")
            return

        # Determine which iterations to use
        k = 1
        if mode == "top3":
            k = 3
        elif mode == "top5":
            k = 5
        top_ids = [str(x).strip() for x in order[:k]]
        top_ids = [x for x in top_ids if x.isdigit()]
        if not top_ids:
            self.logger.warning(f"Human ranking for run_id={run_id} has no valid iteration numbers in top {k}")
            return

        top_iter = int(top_ids[0])  # rank1 always the first item
        md_top = self.project.load_iteration_metadata(run_id=run_id, iteration_num=top_iter)
        if not md_top:
            self.logger.warning(f"Could not load metadata for run_id={run_id} iteration={top_iter}")
            return

        # Collect metadata for averaging if needed
        metadatas = [md_top]
        if k > 1:
            for it_str in top_ids[1:]:
                md = self.project.load_iteration_metadata(run_id=run_id, iteration_num=int(it_str))
                if md:
                    metadatas.append(md)

        aigen = self.project.load_aigen_config()
        aigen.setdefault("prompts", {})
        aigen.setdefault("parameters", {})

        # Prompts: use rank1 prompts as the base (averaging prompts is ill-defined).
        prompts_used = (md_top.get("prompts_used") or {}) if isinstance(md_top, dict) else {}
        pos = prompts_used.get("positive", "")
        neg = prompts_used.get("negative", "")
        if isinstance(pos, str):
            aigen["prompts"]["positive"] = pos
        if isinstance(neg, str):
            aigen["prompts"]["negative"] = neg

        # Parameters: average numeric params across topK, pick mode for categorical.
        def _avg_num(key: str, default: float) -> float:
            vals = []
            for md in metadatas:
                v = (md.get("parameters_used") or {}).get(key)
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return sum(vals) / len(vals) if vals else default

        def _mode_str(key: str, default: str) -> str:
            counts = {}
            for md in metadatas:
                v = (md.get("parameters_used") or {}).get(key)
                if not isinstance(v, str) or not v.strip():
                    continue
                vv = v.strip()
                counts[vv] = counts.get(vv, 0) + 1
            if not counts:
                return default
            return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        if mode == "rank1":
            # Use rank1 params directly
            params_used = md_top.get("parameters_used") or {}
            for key in ("denoise", "cfg", "steps", "sampler_name", "scheduler"):
                if key in params_used:
                    aigen["parameters"][key] = params_used[key]
        else:
            aigen["parameters"]["denoise"] = round(_avg_num("denoise", float(aigen["parameters"].get("denoise", 0.5))), 3)
            aigen["parameters"]["cfg"] = round(_avg_num("cfg", float(aigen["parameters"].get("cfg", 7.0))), 3)
            aigen["parameters"]["steps"] = int(round(_avg_num("steps", float(aigen["parameters"].get("steps", 25)))))
            aigen["parameters"]["sampler_name"] = _mode_str("sampler_name", str(aigen["parameters"].get("sampler_name", "dpmpp_2m")))
            aigen["parameters"]["scheduler"] = _mode_str("scheduler", str(aigen["parameters"].get("scheduler", "karras")))

        # Always randomise seed on next run
        aigen["parameters"]["seed"] = None
        self.project.save_aigen_config(aigen)

        # Human feedback context: include notes for all topK (if present)
        notes_map = ranking.get("notes") or {}
        notes_lines = []
        for it_str in top_ids:
            note = notes_map.get(it_str, "")
            if isinstance(note, str) and note.strip():
                notes_lines.append(f"- Iteration {it_str}: {note.strip()}")
        if notes_lines:
            self.human_feedback_context = (
                f"\n\nHUMAN FEEDBACK (prior run {run_id}, seed_mode={mode}):\n" + "\n".join(notes_lines) + "\n"
            )
        else:
            self.human_feedback_context = ""

        self.logger.info(
            f"Seeded working/AIGen.yaml from human ranking: run_id={run_id} mode={mode} iterations={top_ids}"
            + (" (dry-run)" if dry_run else "")
        )
    
    def _setup_logging(self):
        """Setup application logging."""
        log_file = self.project.project_root / "logs" / "app.log"
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _describe_original_image(self) -> str:
        """Describe original image once and cache it."""
        if self.original_description is None:
            self.logger.info("Describing original image (caching for all iterations)...")
            self.original_description = self.aivis.describe_image(str(self.input_image_path))
        return self.original_description

    def _mask_is_all_white(self, mask_path: Path) -> bool:
        """Check if a mask image is all white (or nearly all white), indicating everything is editable.
        
        Returns True if >95% of pixels are white (>= 240 in grayscale).
        """
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(mask_path).convert("L")
            arr = np.array(img)
            
            # Count pixels that are white (>= 240 to account for slight variations)
            white_pixels = np.sum(arr >= 240)
            total_pixels = arr.size
            
            if total_pixels == 0:
                return False
            
            white_ratio = white_pixels / total_pixels
            return white_ratio >= 0.95
        except Exception as e:
            self.logger.warning(f"Could not check if mask is all-white: {e}")
            return False

    def _get_inpainting_boost_values(self) -> dict:
        """Get inpainting boost values from rules.yaml project section.
        
        Returns dict with keys: denoise_min, cfg_min, denoise_threshold, cfg_threshold
        """
        rules = self.rules if isinstance(self.rules, dict) else {}
        project = rules.get("project") or {}
        
        return {
            "denoise_min": float(project.get("inpainting_denoise_min", 0.85)),
            "cfg_min": float(project.get("inpainting_cfg_min", 10.0)),
            "denoise_threshold": float(project.get("inpainting_denoise_threshold", 0.7)),
            "cfg_threshold": float(project.get("inpainting_cfg_threshold", 9.0)),
        }

    def _get_rules_base_prompts(self, scope: Optional[str]) -> tuple[str, str]:
        """Fetch base prompts from rules.yaml `prompts` section.

        Priority:
        - prompts.masks[scope] (if scope provided)
        - prompts.masks["default"] (if scope is None)
        - prompts.global
        """
        rules = self.rules if isinstance(self.rules, dict) else {}
        prompts = rules.get("prompts") or {}
        if not isinstance(prompts, dict):
            return "", ""

        masks = prompts.get("masks") or {}
        global_p = prompts.get("global") or {}

        def _extract(obj) -> tuple[str, str]:
            if not isinstance(obj, dict):
                return "", ""
            return str(obj.get("positive") or ""), str(obj.get("negative") or "")

        if scope and isinstance(masks, dict) and scope in masks:
            pos, neg = _extract(masks.get(scope))
            if pos.strip() or neg.strip():
                return pos, neg

        if scope is None and isinstance(masks, dict) and "default" in masks:
            pos, neg = _extract(masks.get("default"))
            if pos.strip() or neg.strip():
                return pos, neg

        pos, neg = _extract(global_p)
        return pos, neg
    
    def _answer_questions(self, image_path: str) -> Dict:
        """Proxy to EvaluationService."""
        return self.evaluator.answer_questions(image_path)

    def _evaluate_acceptance_criteria(
        self,
        image_path: str,
        question_answers: Dict,
        criteria: Optional[List[Dict]] = None,
    ) -> Dict:
        """Proxy to EvaluationService."""
        return self.evaluator.evaluate_acceptance_criteria(
            image_path,
            question_answers,
            criteria=criteria,
        )
    
    def _run_iteration(self, iteration_num: int) -> Optional[Dict]:
        """Run a single iteration of generation and evaluation."""
        self.logger.info(f"{'='*60}\nIteration {iteration_num}\n{'='*60}")
        
        # Load current AIGen.yaml
        aigen_config = self.project.load_aigen_config()
        project_cfg = (self.rules.get("project") or {}) if isinstance(self.rules, dict) else {}
        
        # Auto-boost denoise/cfg when masking is enabled (inpainting allows higher edit strength)
        # This will be applied after mask detection, but we check here to avoid duplicate boosts
        mask_cfg = aigen_config.get("masking") or {}
        mask_enabled_precheck = bool(mask_cfg.get("enabled", True))
        lock_seed = bool(project_cfg.get("lock_seed"))
        if lock_seed:
            params = aigen_config.get("parameters") or {}
            seed = params.get("seed")
            if seed is None:
                # If we already discovered a good seed earlier in the run, re-use it.
                if self.locked_seed is not None:
                    params["seed"] = int(self.locked_seed)
                    aigen_config["parameters"] = params
                    # Persist so ComfyUI workflow gets a deterministic seed.
                    self.project.save_aigen_config(aigen_config)
            else:
                try:
                    self.locked_seed = int(seed)
                except Exception:
                    pass
        
        # Prepare input image for ComfyUI
        input_filename = self.project.prepare_input_image(
            self.input_image_path, self.comfyui_input_dir
        )

        # Optional mask support (single or multi-mask).
        #
        # Backwards compatible:
        # - If projects/<name>/input/mask.png exists, we use it (unless masking.enabled is false).
        #
        # Multi-mask:
        # - AIGen.yaml can define masking.active_mask + masking.masks to select a specific mask file.
        # - Criteria can be scoped per mask via acceptance_criteria.applies_to_masks in rules.yaml.
        mask_filename = None
        active_mask_name: Optional[str] = None

        mask_cfg = aigen_config.get("masking") or {}
        try:
            mask_enabled = bool(mask_cfg.get("enabled", True))
        except Exception:
            mask_enabled = True

        if mask_enabled:
            # Resolve active mask name (optional)
            raw_active = mask_cfg.get("active_mask") or mask_cfg.get("active")
            if isinstance(raw_active, str) and raw_active.strip():
                active_mask_name = raw_active.strip()

            # Resolve masks list (optional)
            resolved_mask_path: Optional[Path] = None
            masks = mask_cfg.get("masks")
            if masks:
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
                            candidate = (self.project.project_root / candidate).resolve()
                        if candidate.exists():
                            resolved_mask_path = candidate
                            active_mask_name = name
                            break
                # Dict form: {name: file, ...}
                elif isinstance(masks, dict):
                    if active_mask_name and active_mask_name in masks:
                        candidate = Path(str(masks.get(active_mask_name)))
                        if not candidate.is_absolute():
                            candidate = (self.project.project_root / candidate).resolve()
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
                                candidate = (self.project.project_root / candidate).resolve()
                            if candidate.exists():
                                resolved_mask_path = candidate
                                active_mask_name = n
                                break

            # Fallback: try input/masks/{active_mask_name}.png if active_mask_name is set
            if resolved_mask_path is None and active_mask_name:
                try:
                    named_mask = self.project.project_root / "input" / "masks" / f"{active_mask_name}.png"
                    if named_mask.exists():
                        resolved_mask_path = named_mask
                        self.logger.info(f"Found mask file: input/masks/{active_mask_name}.png")
                except Exception:
                    pass
            
            # Fallback to legacy mask.png if nothing selected
            if resolved_mask_path is None:
                try:
                    legacy = self.project.project_root / "input" / "mask.png"
                    if legacy.exists():
                        resolved_mask_path = legacy
                        if not active_mask_name:
                            active_mask_name = "default"
                except Exception:
                    resolved_mask_path = None

            if resolved_mask_path is not None:
                # Log which mask file is being used
                self.logger.info(f"Using mask: {active_mask_name or 'default'} from file: {resolved_mask_path}")
                
                # If the mask is all white (everything editable), prefer running without inpaint.
                if self._mask_is_all_white(resolved_mask_path):
                    self.logger.warning("Mask appears all-white (>95% white pixels); treating as no-mask (non-inpaint) run.")
                    resolved_mask_path = None
                if resolved_mask_path is not None:
                    try:
                        mask_filename = self.project.prepare_input_image(resolved_mask_path, self.comfyui_input_dir)
                        # Log mask info for debugging
                        try:
                            from PIL import Image
                            import numpy as np
                            img = Image.open(resolved_mask_path).convert("L")
                            arr = np.array(img)
                            white = np.sum(arr >= 128)  # Pixels that are white (editable)
                            total = arr.size
                            coverage = white / total * 100
                            self.logger.info(f"Mask '{active_mask_name or 'default'}' coverage: {coverage:.1f}% white (editable), {100-coverage:.1f}% black (preserved)")
                            if coverage > 50:
                                self.logger.warning(f"Mask '{active_mask_name or 'default'}' covers {coverage:.1f}% of image - this may affect multiple subjects, not just the target!")
                        except ImportError as e:
                            self.logger.warning(f"Could not analyze mask coverage (PIL/numpy not available): {e}")
                        except Exception as e:
                            self.logger.warning(f"Could not analyze mask coverage: {e}", exc_info=True)
                        # Boost denoise/cfg for inpainting (masks allow higher edit strength)
                        params = aigen_config.get("parameters", {})
                        current_denoise = float(params.get("denoise", 0.5))
                        current_cfg = float(params.get("cfg", 7.0))
                        
                        # Get configurable boost values from rules.yaml
                        boost = self._get_inpainting_boost_values()
                        
                        # If denoise/cfg are at default or low values, boost them for inpainting
                        if current_denoise < boost["denoise_threshold"]:
                            params["denoise"] = boost["denoise_min"]
                            self.logger.info(f"Boosted denoise to {params['denoise']} for inpainting (mask: {active_mask_name or 'default'}, from rules.yaml)")
                        if current_cfg < boost["cfg_threshold"]:
                            params["cfg"] = boost["cfg_min"]
                            self.logger.info(f"Boosted cfg to {params['cfg']} for inpainting (mask: {active_mask_name or 'default'}, from rules.yaml)")
                        
                        aigen_config["parameters"] = params
                        # Save the boosted values back to working/AIGen.yaml
                        try:
                            self.project.save_aigen_config(aigen_config)
                        except Exception:
                            pass
                    except Exception:
                        mask_filename = None

        # Optional ControlNet control image (for workflows that require a separate control image input).
        # Convention: projects/<name>/input/control.png
        control_filename = None
        try:
            control_path = self.project.project_root / "input" / "control.png"
            if control_path.exists():
                control_filename = self.project.prepare_input_image(control_path, self.comfyui_input_dir)
        except Exception:
            control_filename = None

        # When inpainting (mask present), allow seed to vary by default so the masked region can explore options.
        # You can force locking via project.lock_seed_inpaint: true
        inpaint_mode = bool(mask_filename)
        if inpaint_mode and not bool(project_cfg.get("lock_seed_inpaint", False)):
            lock_seed = False
            # Also clear any existing seed to ensure variation between iterations
            params = aigen_config.get("parameters", {})
            if "seed" in params and params.get("seed") is not None:
                params["seed"] = None
                aigen_config["parameters"] = params
                try:
                    self.project.save_aigen_config(aigen_config)
                    self.logger.info("Cleared seed for inpainting mode to allow variation")
                except Exception:
                    pass

        # If AIGen.yaml prompts are empty, initialise from rules.yaml base prompts (global or per-mask).
        prompts_cfg = aigen_config.get("prompts") or {}
        if not isinstance(prompts_cfg, dict):
            prompts_cfg = {}
        cur_pos = str(prompts_cfg.get("positive") or "").strip()
        cur_neg = str(prompts_cfg.get("negative") or "").strip()
        if not cur_pos or not cur_neg:
            scope_for_prompts = active_mask_name if active_mask_name else None
            base_pos, base_neg = self._get_rules_base_prompts(scope_for_prompts)
            
            # Validate: reject prompts that look like field names (snake_case identifiers matching criterion fields).
            def _looks_like_field_names(text: str) -> bool:
                """Detect if prompt text is just field names instead of actual SD tags."""
                if not text:
                    return False
                # Get all criterion field names
                all_fields = {str(c.get("field", "")).strip() for c in (self.rules.get("acceptance_criteria") or []) if isinstance(c, dict)}
                # Check if the prompt is mostly/entirely field names
                words = [w.strip().lower() for w in text.replace(",", " ").split() if w.strip()]
                if not words:
                    return False
                # If >50% of words match field names, it's probably field names
                matches = sum(1 for w in words if w in all_fields or w.replace("not_", "") in all_fields)
                return matches > len(words) * 0.5
            
            changed = False
            if not cur_pos and (base_pos or "").strip():
                if _looks_like_field_names(base_pos):
                    self.logger.warning(
                        f"Rejected rules.yaml base prompt (positive) - it contains field names, not SD tags. "
                        f"Use 'Generate base prompts' in Rules UI to create proper prompts. "
                        f"Found: {base_pos[:80]}"
                    )
                else:
                    prompts_cfg["positive"] = base_pos
                    changed = True
            if not cur_neg and (base_neg or "").strip():
                if _looks_like_field_names(base_neg):
                    self.logger.warning(
                        f"Rejected rules.yaml base prompt (negative) - it contains field names, not SD tags. "
                        f"Use 'Generate base prompts' in Rules UI to create proper prompts. "
                        f"Found: {base_neg[:80]}"
                    )
                else:
                    prompts_cfg["negative"] = base_neg
                    changed = True
            if changed:
                aigen_config["prompts"] = prompts_cfg
                try:
                    self.project.save_aigen_config(aigen_config)
                except Exception:
                    pass
                self.logger.info(
                    f"Initialised prompts from rules.yaml prompts (scope={scope_for_prompts or 'default/global'})."
                )
        
        # Load and update workflow
        # If a mask is present, prefer an inpaint workflow (project-local first, then defaults).
        workflow_path = aigen_config.get("workflow_file")
        workflow_switched = False
        if mask_filename:
            # If current workflow isn't already an inpaint workflow, try switching to one.
            wf_name = str(workflow_path or "")
            if "inpaint" not in wf_name.lower():
                candidate = self.project.project_root / "workflow" / "img2img_inpaint_api.json"
                if candidate.exists():
                    workflow_path = "workflow/img2img_inpaint_api.json"
                    workflow_switched = True
                else:
                    # Fall back to repo defaults path if project doesn't have it for some reason.
                    workflow_path = "defaults/workflow/img2img_inpaint_api.json"
                    workflow_switched = True
                # Persist the workflow switch to AIGen.yaml
                if workflow_switched:
                    aigen_config["workflow_file"] = workflow_path
                    try:
                        self.project.save_aigen_config(aigen_config)
                        self.logger.info(f"Switched to inpaint workflow: {workflow_path}")
                    except Exception:
                        pass

        workflow = WorkflowManager.load_workflow(
            workflow_path, self.project.project_root
        )
        updated_workflow = WorkflowManager.update_workflow(
            workflow,
            aigen_config,
            input_filename,
            mask_image_path=mask_filename,
            control_image_path=control_filename,
        )
        
        # Log checkpoint info (extract before removing metadata)
        wf_meta = updated_workflow.get("_workflow_metadata", {})
        ckpt_used = wf_meta.get("checkpoint_used", "unknown")
        ckpt_switched = wf_meta.get("checkpoint_switched", False)
        if ckpt_switched:
            self.logger.info(f"Using inpainting checkpoint: {ckpt_used}")
        else:
            self.logger.info(f"Using checkpoint: {ckpt_used}")
        
        # Remove metadata before sending to ComfyUI (it doesn't understand our custom fields)
        workflow_for_comfyui = {k: v for k, v in updated_workflow.items() if k != "_workflow_metadata"}
        
        # Queue workflow
        try:
            prompt_id = self.comfyui.queue_prompt(workflow_for_comfyui)
            self.logger.info(f"Prompt ID: {prompt_id}")
        except Exception as e:
            self.logger.error(f"Failed to queue workflow: {e}")
            return None
        
        # Wait for completion
        self.logger.info("Waiting for workflow completion...")
        if not self.comfyui.wait_for_completion(prompt_id):
            self.logger.warning("Workflow completion timeout, attempting to fetch result anyway...")
        
        # Get result
        self.logger.info("Fetching result...")
        image_info = None
        last_error = None
        for attempt in range(8):
            try:
                time.sleep(1 if attempt > 0 else 0.5)
                history = self.comfyui.get_history(prompt_id)
                
                # ComfyUI history API returns {prompt_id: {...}}}
                # But sometimes it's just the prompt_id directly
                if isinstance(history, dict):
                    if prompt_id in history:
                        execution = history[prompt_id]
                    elif len(history) == 1:
                        # Sometimes it's wrapped differently
                        execution = list(history.values())[0]
                    else:
                        # Try to find any entry that looks like our execution
                        execution = history
                else:
                    execution = history
                
                if not isinstance(execution, dict):
                    last_error = f"Unexpected history format: {type(execution)}"
                    continue
                
                # Check for status/error first
                status = execution.get('status', {})
                if isinstance(status, dict):
                    if status.get('completed', False) is False:
                        if status.get('error'):
                            last_error = f"Workflow error: {status.get('error')}"
                            self.logger.error(last_error)
                            return None
                        # Still running, continue waiting
                        continue
                
                outputs = execution.get('outputs', {})
                
                if not outputs:
                    # Check if it's in a different format
                    if 'images' in execution:
                        # Direct images array
                        if len(execution['images']) > 0:
                            image_info = execution['images'][0]
                            break
                    last_error = f"No outputs found in execution. Keys: {list(execution.keys())}"
                    continue
                
                # Look for SaveImage node outputs
                for node_id, node_output in outputs.items():
                    if not isinstance(node_output, dict):
                        continue
                    if 'images' in node_output and len(node_output['images']) > 0:
                        image_info = node_output['images'][0]
                        self.logger.info(f"Found image in node {node_id}: {image_info.get('filename', 'unknown')}")
                        break
                
                if image_info:
                    break
                    
            except Exception as e:
                last_error = f"Exception fetching history: {e}"
                self.logger.debug(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < 7:
                    continue
        
        if not image_info:
            error_msg = f"No image found in workflow outputs"
            if last_error:
                error_msg += f" (last error: {last_error})"
            self.logger.error(error_msg)
            # Try to get more debug info
            try:
                history = self.comfyui.get_history(prompt_id)
                self.logger.debug(f"History response: {json.dumps(history, indent=2)[:500]}")
            except Exception as e:
                self.logger.debug(f"Could not fetch history for debugging: {e}")
            return None
        
        # Download and save image
        self.logger.info("Downloading generated image...")
        image_data = self.comfyui.download_image(
            image_info['filename'],
            image_info.get('subfolder', ''),
            image_info.get('type', 'output')
        )
        
        paths = self.project.get_iteration_paths(iteration_num, run_id=self.run_id)
        with open(paths['image'], 'wb') as f:
            f.write(image_data)
        self.logger.info(f"Saved image: {paths['image']}")
        
        # Answer questions, evaluate, and compare
        questions = self._answer_questions(str(paths['image']))
        
        # If criteria are scoped per mask, evaluate only the relevant subset for this iteration.
        #
        # Preferred: rules.yaml membership model:
        #   masking:
        #     masks:
        #       - name: left
        #         active_criteria: [field1, field2]
        #
        # Backwards compatible: acceptance_criteria.applies_to_masks / mask_scope
        all_criteria = self.rules.get('acceptance_criteria', []) or []

        def _criteria_for_active_mask(criteria_defs: List[Dict], mask_name: Optional[str]) -> List[Dict]:
            # Membership model
            masking_rules = self.rules.get("masking") if isinstance(self.rules, dict) else None
            masks = (masking_rules.get("masks") if isinstance(masking_rules, dict) else None) if masking_rules else None
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
                return [c for c in criteria_defs if isinstance(c, dict) and str(c.get("field") or "").strip() in active_fields]

            # Legacy per-criterion scoping
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

        scoped_criteria = _criteria_for_active_mask(all_criteria, active_mask_name)
        evaluation = self._evaluate_acceptance_criteria(str(paths['image']), questions, criteria=scoped_criteria)
        
        score = evaluation.get('overall_score', 0)
        self.logger.info(f"Evaluation score: {score}%")
        
        comparison = self.aivis.compare_images(str(self.input_image_path), str(paths['image']))
        
        similarity = comparison.get('similarity_score', 0.5)
        self.logger.info(f"Similarity score: {similarity:.2f}")
        
        # Collect AIVis metadata (provider, model, route info)
        aivis_metadata = {
            "evaluation": evaluation.get("_metadata", {}),
            "comparison": comparison.get("_metadata", {}),
            "questions": questions.get("_metadata", {})
        }
        
        # Remove _metadata from results for cleaner output (but keep in saved files)
        questions_clean = {k: v for k, v in questions.items() if k != "_metadata"}
        evaluation_clean = {k: v for k, v in evaluation.items() if k != "_metadata"}
        comparison_clean = {k: v for k, v in comparison.items() if k != "_metadata"}
        
        # Save files with metadata included
        with open(paths['questions'], 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2)
        with open(paths['evaluation'], 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        with open(paths['comparison'], 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        
        # Build and save metadata
        parameters_used = aigen_config.get('parameters', {}).copy()
        used_seed: Optional[int] = None
        if 'seed' in parameters_used and parameters_used['seed'] is None:
            for node_id, node_data in updated_workflow.items():
                if isinstance(node_data, dict) and node_data.get('class_type') == 'KSampler':
                    try:
                        used_seed = int(node_data['inputs'].get('seed'))
                    except Exception:
                        used_seed = None
                    parameters_used['seed'] = used_seed
                    break
        else:
            try:
                used_seed = int(parameters_used.get("seed"))
            except Exception:
                used_seed = None

        # If seed locking is enabled and we haven't locked a seed yet, lock to the first used seed.
        if lock_seed and self.locked_seed is None and used_seed is not None:
            self.locked_seed = used_seed
            try:
                a2 = self.project.load_aigen_config()
                p2 = a2.get("parameters") or {}
                if p2.get("seed") is None:
                    p2["seed"] = used_seed
                    a2["parameters"] = p2
                    self.project.save_aigen_config(a2)
            except Exception:
                pass
        
        # Capture prompts used for this iteration
        prompts_used = aigen_config.get('prompts', {}).copy()
        
        # Get checkpoint info from workflow metadata (we already extracted it earlier)
        # Use the ckpt_used we extracted before removing metadata
        
        metadata = {
            "iteration": iteration_num,
            "timestamp": time.time(),
            "image_path": str(paths['image'].relative_to(self.project.project_root)),
            "workflow_file_used": str(workflow_path),
            "checkpoint_used": ckpt_used,
            "mask_used": mask_filename,
            "active_mask": active_mask_name,
            "control_used": control_filename,
            "parameters_used": parameters_used,
            "prompts_used": prompts_used,
            "questions": questions_clean,
            "evaluation": evaluation_clean,
            "comparison": comparison_clean,
            "aivis_metadata": aivis_metadata,
            "run_id": self.run_id
        }
        
        with open(paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _apply_improvements(self, iteration_result: Dict):
        """Apply improvements to AIGen.yaml for next iteration."""
        self.logger.info("Applying improvements for next iteration...")
        
        evaluation = iteration_result['evaluation']
        comparison = iteration_result['comparison']
        similarity = comparison.get('similarity_score', 0.5)
        
        # Load current AIGen.yaml
        aigen_config = self.project.load_aigen_config()
        
        # Improve prompts based on failed criteria (data-driven via rules.yaml tags)
        #
        # If this run is scoped to a particular mask (multi-mask projects), only consider
        # criteria active for that mask.
        all_criteria_defs = self.rules.get('acceptance_criteria', []) or []
        active_mask = iteration_result.get("active_mask")

        def _criteria_for_active_mask(criteria_defs: List[Dict], mask_name: Optional[str]) -> List[Dict]:
            masking_rules = self.rules.get("masking") if isinstance(self.rules, dict) else None
            masks = (masking_rules.get("masks") if isinstance(masking_rules, dict) else None) if masking_rules else None
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
                return [c for c in criteria_defs if isinstance(c, dict) and str(c.get("field") or "").strip() in active_fields]

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

        criteria_defs = _criteria_for_active_mask(all_criteria_defs, active_mask)
        criteria_by_field = {c.get('field'): c for c in criteria_defs if isinstance(c, dict) and c.get('field')}
        criteria_results = evaluation.get('criteria_results', {}) or {}

        def _strength_value(v: str) -> float:
            v = (v or "medium").strip().lower()
            if v in ("low", "l"):
                return 0.5
            if v in ("high", "h"):
                return 1.5
            return 1.0  # medium

        def _listify(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i).strip() for i in x if str(i).strip()]
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            return [str(x).strip()]

        failed_criteria: List[str] = []
        for field, result in criteria_results.items():
            crit = criteria_by_field.get(field)
            if not crit:
                continue
            ctype = (crit.get('type') or 'boolean').lower()
            min_v = crit.get('min', 0)
            if ctype == 'boolean':
                if result is not True:
                    failed_criteria.append(field)
            else:
                if isinstance(result, (int, float)):
                    if result < min_v:
                        failed_criteria.append(field)
                elif not result:
                    failed_criteria.append(field)

        self.parameter_service.update_parameters(
            aigen_config=aigen_config,
            iteration_result=iteration_result,
            evaluation=evaluation,
            comparison=comparison,
            criteria_defs=criteria_defs,
            criteria_by_field=criteria_by_field,
            failed_criteria=failed_criteria,
        )
        
        # Get AI-driven parameter recommendations (optional, can be enabled per-project)
        aivis_param_recommendations_enabled = proj_cfg.get("aivis_parameter_recommendations", False)
        if aivis_param_recommendations_enabled:
            try:
                self.logger.info("Getting AI parameter recommendations from AIVis...")
                image_path = iteration_result.get('image_path')
                if image_path:
                    # Resolve to absolute path if needed
                    if not Path(image_path).is_absolute():
                        image_path = self.project.project_root / image_path
                    else:
                        image_path = Path(image_path)
                    
                    # Get original description
                    try:
                        original_desc = self._describe_original_image()
                    except Exception:
                        original_desc = None
                    
                    # Get recommendations
                    recommendations = self.aivis.recommend_parameters(
                        image_path=str(image_path),
                        original_description=original_desc,
                        current_denoise=cur_d,
                        current_cfg=cur_cfg,
                        evaluation=evaluation,
                        comparison=comparison,
                        failed_criteria=failed_criteria,
                    )
                    
                    # Apply recommendations with confidence weighting
                    # We blend AI recommendations with algorithmic adjustments
                    denoise_rec = recommendations.get("denoise_recommendation", "keep")
                    denoise_conf = float(recommendations.get("denoise_confidence", 0.0))
                    denoise_reason = recommendations.get("denoise_reasoning", "")
                    
                    cfg_rec = recommendations.get("cfg_recommendation", "keep")
                    cfg_conf = float(recommendations.get("cfg_confidence", 0.0))
                    cfg_reason = recommendations.get("cfg_reasoning", "")
                    
                    quality_assessment = recommendations.get("overall_quality_assessment", "")
                    
                    self.logger.info(f"AIVis parameter recommendations:")
                    self.logger.info(f"  Denoise: {denoise_rec} (confidence: {denoise_conf:.2f}) - {denoise_reason}")
                    self.logger.info(f"  CFG: {cfg_rec} (confidence: {cfg_conf:.2f}) - {cfg_reason}")
                    if quality_assessment:
                        self.logger.info(f"  Quality assessment: {quality_assessment}")
                    
                    # Apply recommendations with confidence weighting
                    # Only apply if confidence is above threshold (0.5) and it doesn't conflict with caps
                    confidence_threshold = 0.5
                    
                    if denoise_conf >= confidence_threshold:
                        if denoise_rec == "increase":
                            # Weight the recommendation by confidence (0.5-1.0 maps to 0.5-1.0x of base increment)
                            rec_weight = 0.5 + (denoise_conf * 0.5)  # 0.5 to 1.0
                            rec_delta = 0.04 * rec_weight  # Base increment of 0.04, scaled by confidence
                            new_denoise = min(denoise_max, params.get("denoise", cur_d) + rec_delta)
                            if new_denoise > params.get("denoise", cur_d):
                                params["denoise"] = new_denoise
                                self.logger.info(f"  Applied AIVis denoise increase: {params['denoise']:.3f} (+{rec_delta:.3f})")
                        elif denoise_rec == "decrease":
                            rec_weight = 0.5 + (denoise_conf * 0.5)
                            rec_delta = 0.04 * rec_weight
                            new_denoise = max(denoise_min, params.get("denoise", cur_d) - rec_delta)
                            if new_denoise < params.get("denoise", cur_d):
                                params["denoise"] = new_denoise
                                self.logger.info(f"  Applied AIVis denoise decrease: {params['denoise']:.3f} (-{rec_delta:.3f})")
                    
                    if cfg_conf >= confidence_threshold:
                        if cfg_rec == "increase":
                            rec_weight = 0.5 + (cfg_conf * 0.5)
                            rec_delta = 0.5 * rec_weight  # Base increment of 0.5, scaled by confidence
                            new_cfg = min(cfg_max, params.get("cfg", cur_cfg) + rec_delta)
                            if new_cfg > params.get("cfg", cur_cfg):
                                params["cfg"] = new_cfg
                                self.logger.info(f"  Applied AIVis cfg increase: {params['cfg']:.1f} (+{rec_delta:.1f})")
                        elif cfg_rec == "decrease":
                            rec_weight = 0.5 + (cfg_conf * 0.5)
                            rec_delta = 0.5 * rec_weight
                            new_cfg = max(cfg_min, params.get("cfg", cur_cfg) - rec_delta)
                            if new_cfg < params.get("cfg", cur_cfg):
                                params["cfg"] = new_cfg
                                self.logger.info(f"  Applied AIVis cfg decrease: {params['cfg']:.1f} (-{rec_delta:.1f})")
                    
                    # Store recommendations in metadata for debugging
                    if not hasattr(self, '_last_param_recommendations'):
                        self._last_param_recommendations = []
                    self._last_param_recommendations.append({
                        "iteration": iteration_result.get('iteration'),
                        "recommendations": recommendations
                    })
            except Exception as e:
                self.logger.warning(f"Failed to get AIVis parameter recommendations: {e}")
                # Continue with algorithmic adjustments only
        
        # Check if prompt improvement is enabled (default: false for testing)
        improve_prompts_enabled = proj_cfg.get("improve_prompts", False)
        
        if failed_criteria and improve_prompts_enabled:
            changed = self.prompt_service.maybe_improve_prompts(
                aigen_config=aigen_config,
                evaluation=evaluation,
                comparison=comparison,
                failed_criteria=failed_criteria,
                criteria_defs=criteria_defs,
                criteria_by_field=criteria_by_field,
            )
            if not changed:
                self.logger.info("Prompt service skipped updates (conditions not met).")
        elif failed_criteria:
            self.logger.info("Prompt improvement disabled (project.improve_prompts=false). Keeping prompts unchanged.")
        # Save updated AIGen.yaml
        self.project.save_aigen_config(aigen_config)
    
    def run(
        self,
        dry_run: bool = False,
        resume_from: Optional[int] = None,
        seed_from_ranking: Optional[str] = None,
        seed_ranking_mode: str = "rank1",
    ):
        """Run the iterative improvement loop.
        
        Args:
            dry_run: If True, don't actually run iterations
            resume_from: Resume from a specific iteration number (None = start from beginning or checkpoint)
        """
        self.logger.info(f"{'='*60}\nIterative Imagination\n{'='*60}")
        self.logger.info(f"Project: {self.project_name}")
        self.logger.info(f"Input image: {self.input_image_path}")
        self.logger.info(f"Max iterations: {self.rules['project']['max_iterations']}\n{'='*60}")
        
        # Optional seed before doing anything else (even in dry-run, to show it is wired)
        self._maybe_seed_from_human_ranking(seed_from_ranking, seed_mode=seed_ranking_mode, dry_run=dry_run)

        if dry_run:
            self.logger.info("DRY RUN - Validating configuration only")
            if not self.comfyui.test_connection():
                self.logger.error("Cannot connect to ComfyUI!")
                return False
            if not self.aivis.test_connection():
                self.logger.error("Cannot connect to Ollama!")
                return False
            self.logger.info("Configuration valid!")
            return True
        
        # Test connections
        if not self.comfyui.test_connection():
            self.logger.error("Cannot connect to ComfyUI! Make sure ComfyUI is running")
            return False
        if not self.aivis.test_connection():
            self.logger.error("Cannot connect to Ollama! Make sure Ollama is running: ollama serve")
            return False
        
        self.logger.info("Connections successful!")
        max_iterations = self.rules['project']['max_iterations']
        output_paths = self.project.get_output_paths()
        
        # Check for resume
        start_iteration = 1
        if resume_from is not None:
            start_iteration = resume_from
            self.logger.info(f"Resuming from iteration {start_iteration}")
            # manual resume implies new run unless checkpoint exists
        else:
            # Check for checkpoint
            checkpoint = self.project.load_checkpoint()
            if checkpoint:
                start_iteration = checkpoint.get('last_iteration', 1) + 1
                self.best_iteration = checkpoint.get('best_iteration')
                self.best_score = checkpoint.get('best_score', 0)
                self.run_id = checkpoint.get('run_id')  # may be None for legacy runs
                self.logger.info(f"Found checkpoint: last iteration {checkpoint.get('last_iteration')}, best score {checkpoint.get('best_score')}%")
                self.logger.info(f"Resuming from iteration {start_iteration}")

        # If no run_id set by checkpoint, create one for this invocation.
        if self.run_id is None:
            self.run_id = self.project.create_run_id()
            self.logger.info(f"Run id: {self.run_id}")

        # Detect masks and their associated criteria for sequential mask cycling
        mask_sequence = []
        masking_rules = self.rules.get("masking") if isinstance(self.rules, dict) else None
        masks = (masking_rules.get("masks") if isinstance(masking_rules, dict) else None) if masking_rules else None
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
                        mask_sequence.append({"name": mask_name, "criterion": mask_criterion})
        
        # If we have a mask sequence, cycle through them sequentially
        if mask_sequence:
            self.logger.info(f"Multi-mask mode: will cycle through {len(mask_sequence)} mask(s): {[m['name'] for m in mask_sequence]}")
            iterations_run = 0
            last_iteration_run: Optional[int] = None
            current_iteration = start_iteration
            mask_index = 0
            
            while mask_index < len(mask_sequence) and current_iteration <= max_iterations:
                current_mask = mask_sequence[mask_index]
                mask_name = current_mask["name"]
                mask_criterion = current_mask["criterion"]
                
                self.logger.info(f"{'='*60}\nProcessing mask: {mask_name} (criterion: {mask_criterion})\n{'='*60}")
                
                # Set active_mask in working/AIGen.yaml and boost denoise/cfg for inpainting
                aigen_config = self.project.load_aigen_config()
                aigen_config.setdefault("masking", {})
                aigen_config["masking"]["enabled"] = True
                aigen_config["masking"]["active_mask"] = mask_name
                
                # Boost denoise/cfg for inpainting (masks allow higher edit strength)
                params = aigen_config.get("parameters", {})
                current_denoise = float(params.get("denoise", 0.5))
                current_cfg = float(params.get("cfg", 7.0))
                
                # Get configurable boost values from rules.yaml
                boost = self._get_inpainting_boost_values()
                
                # If denoise/cfg are at default or low values, boost them for inpainting
                if current_denoise < boost["denoise_threshold"]:
                    params["denoise"] = boost["denoise_min"]
                    self.logger.info(f"Boosted denoise to {params['denoise']} for inpainting (mask: {mask_name}, from rules.yaml)")
                if current_cfg < boost["cfg_threshold"]:
                    params["cfg"] = boost["cfg_min"]
                    self.logger.info(f"Boosted cfg to {params['cfg']} for inpainting (mask: {mask_name}, from rules.yaml)")
                
                aigen_config["parameters"] = params
                self.project.save_aigen_config(aigen_config)
                
                mask_passed = False
                mask_iterations = 0
                max_mask_iterations = max_iterations - current_iteration + 1  # Remaining iterations
                
                # Run iterations for this mask until criterion passes or iterations run out
                for iteration in range(current_iteration, min(current_iteration + max_mask_iterations, max_iterations + 1)):
                    last_iteration_run = iteration
                    iterations_run += 1
                    mask_iterations += 1
                    current_iteration = iteration + 1
                    
                    iteration_result = self._run_iteration(iteration)
                    if not iteration_result:
                        self.logger.error("Iteration failed, stopping")
                        break
                    
                    score = iteration_result['evaluation'].get('overall_score', 0)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_iteration = iteration
                    
                    # Save checkpoint after each iteration
                    self.project.save_checkpoint(iteration, self.best_iteration, self.best_score, run_id=self.run_id)
                    
                    # Check if this mask's specific criterion passed
                    criteria_results = iteration_result.get('evaluation', {}).get('criteria_results', {})
                    criterion_passed = criteria_results.get(mask_criterion, False)
                    
                    if criterion_passed:
                        self.logger.info(f"{'='*60}\nMask '{mask_name}' criterion '{mask_criterion}' PASSED! (Iteration {iteration})\n{'='*60}")
                        mask_passed = True
                        # Update progress image for next mask
                        try:
                            progress_path = self.project.project_root / "input" / "progress.png"
                            progress_path.parent.mkdir(parents=True, exist_ok=True)
                            iteration_paths = self.project.get_iteration_paths(iteration, run_id=self.run_id)
                            shutil.copy2(iteration_paths['image'], progress_path)
                            self.logger.info(f"Updated progress image for next mask: {progress_path}")
                        except Exception:
                            pass
                        # Don't apply improvements when mask passes - move to next mask
                        break
                    
                    # Check if we've achieved perfect score (all criteria pass)
                    if score >= 100:
                        self.logger.info(f"{'='*60}\nPerfect score achieved! (Score: {score}%)\n{'='*60}")
                        shutil.copy2(self.project.get_iteration_paths(iteration, run_id=self.run_id)['image'], output_paths['image'])
                        try:
                            progress_path = self.project.project_root / "input" / "progress.png"
                            progress_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(output_paths['image'], progress_path)
                            self.logger.info(f"Updated progress image: {progress_path}")
                        except Exception:
                            pass
                        with open(output_paths['metadata'], 'w', encoding='utf-8') as f:
                            json.dump({"final_iteration": iteration, "final_score": score, "metadata": iteration_result}, f, indent=2)
                        mask_passed = True  # All masks passed
                        break
                    
                    # Apply improvements for next iteration (only if mask hasn't passed yet)
                    if iteration < max_iterations:
                        self._apply_improvements(iteration_result)
                        time.sleep(2)
                
                if not mask_passed:
                    self.logger.warning(f"Mask '{mask_name}' did not pass after {mask_iterations} iteration(s). Moving to next mask anyway.")
                
                # Move to next mask
                mask_index += 1
                if current_iteration > max_iterations:
                    break
        else:
            # Single mask or no mask mode - original behavior
            iterations_run = 0
            last_iteration_run: Optional[int] = None

            for iteration in range(start_iteration, max_iterations + 1):
                last_iteration_run = iteration
                iterations_run += 1
                iteration_result = self._run_iteration(iteration)
                if not iteration_result:
                    self.logger.error("Iteration failed, stopping")
                    break
                
                score = iteration_result['evaluation'].get('overall_score', 0)
                if score > self.best_score:
                    self.best_score = score
                    self.best_iteration = iteration
                
                # Check if we've achieved perfect score
                if score >= 100:
                    self.logger.info(f"{'='*60}\nPerfect score achieved! (Score: {score}%)\n{'='*60}")
                    shutil.copy2(self.project.get_iteration_paths(iteration, run_id=self.run_id)['image'], output_paths['image'])
                    # Also update input/progress.png so multi-pass edits can chain without overwriting input/input.png.
                    try:
                        progress_path = self.project.project_root / "input" / "progress.png"
                        progress_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(output_paths['image'], progress_path)
                        self.logger.info(f"Updated progress image: {progress_path}")
                    except Exception:
                        pass
                    with open(output_paths['metadata'], 'w', encoding='utf-8') as f:
                        json.dump({"final_iteration": iteration, "final_score": score, "metadata": iteration_result}, f, indent=2)
                    break
                
                # Save checkpoint after each iteration
                self.project.save_checkpoint(iteration, self.best_iteration, self.best_score, run_id=self.run_id)
                
                # Apply improvements for next iteration
                if iteration < max_iterations:
                    self._apply_improvements(iteration_result)
                    time.sleep(2)
        
        # Summary
        self.logger.info(f"{'='*60}\nIteration Loop Complete\n{'='*60}")
        if iterations_run == 0:
            self.logger.info(
                f"No iterations to run (start_iteration={start_iteration} > max_iterations={max_iterations})."
            )
        else:
            self.logger.info(f"Iterations run this invocation: {iterations_run} (last={last_iteration_run})")

        if self.best_iteration and self.best_score < 100:
            self.logger.info(f"Best result: Iteration {self.best_iteration} (Score: {self.best_score}%)")
            best_paths = self.project.get_iteration_paths(self.best_iteration, run_id=self.run_id)
            shutil.copy2(best_paths['image'], output_paths['image'])
            # Also update input/progress.png so multi-pass edits can chain without overwriting input/input.png.
            try:
                progress_path = self.project.project_root / "input" / "progress.png"
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_paths['image'], progress_path)
                self.logger.info(f"Updated progress image: {progress_path}")
            except Exception:
                pass
            with open(best_paths['metadata'], 'r', encoding='utf-8') as f:
                best_metadata = json.load(f)
            with open(output_paths['metadata'], 'w', encoding='utf-8') as f:
                json.dump({"final_iteration": self.best_iteration, "final_score": self.best_score, "metadata": best_metadata}, f, indent=2)
        elif self.best_iteration and self.best_score >= 100:
            # In case we resumed after already hitting 100% earlier
            self.logger.info(f"Perfect score already achieved previously (Iteration {self.best_iteration}).")
        
        return True


def main():
    """Main entry point."""
    parser = build_run_parser()
    args = parser.parse_args()
    run_iteration(args, IterativeImagination, parser)


if __name__ == '__main__':
    main()
