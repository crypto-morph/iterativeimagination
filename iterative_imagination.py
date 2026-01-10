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
        """Answer all questions from rules.yaml about the generated image in a single batch request."""
        self.logger.info("Answering questions about generated image (batch request)...")
        questions = self.rules.get('questions', [])
        
        if not questions:
            return {"_metadata": {}}
        
        # Get original description for comparison questions
        original_desc = self._describe_original_image()
        
        try:
            # Use batch method to answer all questions in one request
            answers, metadata = self.aivis.ask_multiple_questions(
                image_path, 
                questions, 
                original_description=original_desc
            )
            
            # Log answers
            for field, answer in answers.items():
                self.logger.debug(f"  {field}: {answer}")
            
            # Store metadata (single request for all questions)
            answers["_metadata"] = {
                "batch_request": metadata  # All questions share the same request metadata
            }
            
            return answers
        except Exception as e:
            self.logger.warning(f"  Failed to answer questions in batch: {e}")
            # Fallback: return defaults for all questions
            answers = {}
            for question_def in questions:
                field = question_def['field']
                qtype = question_def.get('type', 'string')
                answers[field] = False if qtype == 'boolean' else (question_def.get('min', 0) if qtype in ['number', 'integer'] else ([] if qtype == 'array' else ""))
            
            # Store error metadata
            answers["_metadata"] = {
                "batch_request": {
                    "provider": self.aivis.provider,
                    "model": self.aivis.model,
                    "using_fallback": self.aivis._using_fallback,
                    "success": False,
                    "error": str(e)
                }
            }
            return answers
    
    def _evaluate_acceptance_criteria(self, image_path: str, question_answers: Dict, criteria: Optional[List[Dict]] = None) -> Dict:
        """Evaluate image against acceptance criteria.

        If `criteria` is provided, evaluate only those criteria (useful for per-mask scoped runs).
        """
        self.logger.info("Evaluating acceptance criteria...")
        criteria = criteria if criteria is not None else self.rules.get('acceptance_criteria', [])
        original_desc = self._describe_original_image()
        
        evaluation = self.aivis.evaluate_acceptance_criteria(
            image_path, original_desc, criteria, question_answers
        )
        
        # Ensure criteria_results has all fields (for the criteria we asked for)
        criteria_results = evaluation.get('criteria_results', {})
        for criterion in criteria:
            field = criterion['field']
            if field not in criteria_results:
                criteria_results[field] = False
        
        evaluation['criteria_results'] = criteria_results

        # Never trust model-provided overall_score: compute deterministically from criteria_results.
        def _is_pass(cdef: Dict, value) -> bool:
            ctype = str(cdef.get("type") or "boolean").lower()
            if ctype == "boolean":
                return bool(value) is True
            if ctype in ("number", "integer", "float"):
                try:
                    v = float(value)
                except Exception:
                    return False
                try:
                    mn = float(cdef.get("min")) if cdef.get("min") is not None else None
                    mx = float(cdef.get("max")) if cdef.get("max") is not None else None
                except Exception:
                    mn, mx = None, None
                if mn is not None and v < mn:
                    return False
                if mx is not None and v > mx:
                    return False
                return True
            if ctype == "string":
                return isinstance(value, str) and value.strip() != ""
            if ctype == "array":
                if not isinstance(value, list):
                    return False
                try:
                    mn = int(cdef.get("min")) if cdef.get("min") is not None else None
                    mx = int(cdef.get("max")) if cdef.get("max") is not None else None
                except Exception:
                    mn, mx = None, None
                if mn is not None and len(value) < mn:
                    return False
                if mx is not None and len(value) > mx:
                    return False
                return True
            return False

        passed = []
        failed = []
        for cdef in criteria:
            if not isinstance(cdef, dict):
                continue
            field = cdef.get("field")
            if not field:
                continue
            val = criteria_results.get(field)
            if _is_pass(cdef, val):
                passed.append(field)
            else:
                failed.append(field)

        total = len(passed) + len(failed)
        overall = int(round((len(passed) / total) * 100)) if total else 0
        evaluation["overall_score"] = overall
        evaluation["passed_fields"] = passed
        evaluation["failed_fields"] = failed
        evaluation["criteria_total"] = total
        return evaluation
    
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
                            self.logger.info(f"Mask coverage: {coverage:.1f}% white (editable), {100-coverage:.1f}% black (preserved)")
                            if coverage > 50:
                                self.logger.warning(f"Mask covers {coverage:.1f}% of image - this may affect all subjects, not just the target!")
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
        
        # Adjust parameters based on similarity (kept conservative; main tuning is intent-based below).
        params = aigen_config.get('parameters', {})
        current_denoise = params.get('denoise', 0.5)
        current_cfg = params.get('cfg', 7.0)
        
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

        # Determine which criteria are failing
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

        # Parameter tuning based on intent + edit_strength, not hard-coded field names.
        failed_change = []
        failed_preserve = []
        for field in failed_criteria:
            crit = criteria_by_field.get(field, {})
            intent = (crit.get('intent') or 'preserve').strip().lower()
            if intent == 'change':
                failed_change.append(field)
            else:
                failed_preserve.append(field)

        # Per-project bounds/caps (optional)
        proj_cfg = (self.rules.get("project") or {}) if isinstance(self.rules, dict) else {}
        try:
            denoise_min = float(proj_cfg.get("denoise_min", 0.20))
        except Exception:
            denoise_min = 0.20
        try:
            denoise_max = float(proj_cfg.get("denoise_max", 0.80))
        except Exception:
            denoise_max = 0.80
        try:
            cfg_max = float(proj_cfg.get("cfg_max", 12.0))
        except Exception:
            cfg_max = 12.0
        try:
            cfg_min = float(proj_cfg.get("cfg_min", 4.0))
        except Exception:
            cfg_min = 4.0

        # If we're *not* inpainting (no mask), cap "edit strength" more aggressively.
        # Rationale: raising denoise on full-frame img2img causes global drift and makes targeted edits unreliable.
        # For local edits, we want the user to provide input/mask.png so we can switch to the inpaint workflow.
        mask_used = bool(iteration_result.get("mask_used"))
        if not mask_used:
            try:
                denoise_max_no_mask = float(proj_cfg.get("denoise_max_no_mask", 0.55))
            except Exception:
                denoise_max_no_mask = 0.55
            try:
                cfg_max_no_mask = float(proj_cfg.get("cfg_max_no_mask", 9.0))
            except Exception:
                cfg_max_no_mask = 9.0
            denoise_max = min(denoise_max, denoise_max_no_mask)
            cfg_max = min(cfg_max, cfg_max_no_mask)

        # Heuristic: if most criteria are preserve, cap denoise/cfg harder to prevent "wandering".
        # BUT: when inpainting (mask active), don't apply these caps - the mask protects the rest of the image.
        intents = [(c.get("intent") or "preserve").strip().lower() for c in criteria_defs if isinstance(c, dict)]
        n_preserve = sum(1 for x in intents if x != "change")
        n_change = sum(1 for x in intents if x == "change")
        preserve_heavy = bool(proj_cfg.get("preserve_heavy")) or (n_preserve >= 3 and n_change <= 2)
        # Important: preserve-heavy tasks still need enough edit strength to achieve the *one* change goal.
        # So we only tighten caps when preserve criteria are failing (i.e. we're drifting).
        # BUT: skip this cap when inpainting is active - masks allow higher edit strength safely.
        if preserve_heavy and failed_preserve and not mask_used:
            denoise_max = min(denoise_max, 0.62)
            cfg_max = min(cfg_max, 8.5)

        cur_d = float(params.get("denoise", current_denoise) or current_denoise)
        cur_cfg = float(params.get("cfg", current_cfg) or current_cfg)

        # Priority: if we have change failures, prioritize those (increase denoise to make the change happen).
        # Only reduce denoise for preserve failures if we DON'T have change failures.
        if failed_change:
            if not mask_used:
                self.logger.warning(
                    "Change goals are failing but no mask is in use. "
                    "For precise edits, add projects/<name>/input/mask.png (white=edit, black=keep) "
                    "so the runner can switch to an inpaint workflow automatically. "
                    f"Meanwhile, keeping conservative no-mask caps: denoise<= {denoise_max:.2f}, cfg<= {cfg_max:.1f}."
                )
            # Change failures with preserve OK: allow a bit more freedom, but keep within caps.
            max_strength = max(_strength_value(criteria_by_field[f].get('edit_strength')) for f in failed_change)
            # Bigger step size here to actually make progress on stubborn edits (e.g. removing garments).
            delta = 0.06 + 0.02 * max_strength
            # If change is failing and preserve is OK, we should steadily increase denoise until the change lands,
            # while staying within the project caps.
            params["denoise"] = min(denoise_max, max(denoise_min, cur_d + delta))

            cfg_delta = 0.5 + 0.5 * max_strength
            params["cfg"] = min(cfg_max, max(cfg_min, cur_cfg + cfg_delta))
            self.logger.info(
                f"Change goals failing ({failed_change}) - increasing denoise to {params['denoise']:.2f}, cfg to {params['cfg']:.1f} (caps: denoise<= {denoise_max:.2f}, cfg<= {cfg_max:.1f})"
            )
        elif failed_preserve:
            # Preserve failures (and no change failures): pull closer to original immediately.
            max_strength = max(_strength_value(criteria_by_field[f].get('edit_strength')) for f in failed_preserve)
            delta = 0.05 + 0.02 * max_strength
            params["denoise"] = max(denoise_min, cur_d - delta)
            params["cfg"] = max(cfg_min, cur_cfg - (0.5 + 0.5 * max_strength))
            self.logger.info(
                f"Preserve goals failing ({failed_preserve}) - decreasing denoise to {params['denoise']:.2f}, cfg to {params['cfg']:.1f}"
            )

        # If we diverged too far, pull back regardless.
        if similarity <= 0.30:
            params["denoise"] = max(denoise_min, float(params.get("denoise", cur_d)) - 0.05)
            self.logger.info(f"Images too different (similarity={similarity:.2f}) - reducing denoise to {params['denoise']:.2f}")
        
        if failed_criteria:
            self.logger.info(f"Improving prompts based on failed criteria: {failed_criteria}")
            # Include criteria tags in the text shown to AIVis, so it can act on them.
            def _crit_line(c: Dict) -> str:
                q = c.get('question', '')
                intent = (c.get('intent') or 'preserve').strip().lower()
                strength = (c.get('edit_strength') or 'medium').strip().lower()
                must = ", ".join(_listify(c.get('must_include')))
                ban = ", ".join(_listify(c.get('ban_terms')))
                avoid = ", ".join(_listify(c.get('avoid_terms')))
                bits = [f"intent={intent}", f"strength={strength}"]
                if must:
                    bits.append(f"must_include=[{must}]")
                if ban:
                    bits.append(f"ban_terms=[{ban}]")
                if avoid:
                    bits.append(f"avoid_terms=[{avoid}]")
                return f"- {c.get('field')}: {q} ({'; '.join(bits)})"

            rules_text = "\n".join([_crit_line(c) for c in criteria_defs])
            if self.human_feedback_context:
                rules_text = rules_text + self.human_feedback_context
            # Ground the prompt improver to avoid hallucinating an unrelated scene/background.
            # We include the original description and explicit constraints, but still post-process heavily.
            try:
                original_desc = self._describe_original_image()
            except Exception:
                original_desc = ""
            if original_desc:
                rules_text = (
                    "ORIGINAL IMAGE DESCRIPTION (authoritative grounding):\n"
                    + original_desc.strip()
                    + "\n\nIMPORTANT: Do NOT invent new scene elements or backgrounds not present in the original description.\n"
                    + "Keep the scene/background consistent unless a change criterion explicitly requires it.\n\n"
                    + rules_text
                )
            current_positive = aigen_config.get('prompts', {}).get('positive', '')
            current_negative = aigen_config.get('prompts', {}).get('negative', '')

            def _tokenise_prompt(s: str) -> List[str]:
                """Rough tokeniser for prompt diffs: split on commas, normalise whitespace."""
                if not s:
                    return []
                parts = [p.strip() for p in str(s).split(",")]
                out = []
                seen = set()
                for p in parts:
                    if not p:
                        continue
                    k = p.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(p)
                return out

            def _diff_tokens(before: str, after: str) -> Dict[str, List[str]]:
                b = _tokenise_prompt(before)
                a = _tokenise_prompt(after)
                bs = {x.lower() for x in b}
                as_ = {x.lower() for x in a}
                added = [x for x in a if x.lower() not in bs]
                removed = [x for x in b if x.lower() not in as_]
                return {"added": added, "removed": removed}
            
            improved_positive, improved_negative = self.aivis.improve_prompts(
                current_positive, current_negative, evaluation, comparison,
                failed_criteria, rules_text
            )

            # Post-process prompts using tags from failed criteria.
            import re
            must_include_terms: List[str] = []
            ban_terms: List[str] = []
            avoid_terms: List[str] = []
            for field in failed_criteria:
                crit = criteria_by_field.get(field, {})
                must_include_terms.extend(_listify(crit.get('must_include')))
                ban_terms.extend(_listify(crit.get('ban_terms')))
                avoid_terms.extend(_listify(crit.get('avoid_terms')))

            # De-duplicate while preserving order
            def _dedupe(seq: List[str]) -> List[str]:
                seen = set()
                out = []
                for s in seq:
                    key = s.strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    out.append(s.strip())
                return out

            must_include_terms = _dedupe(must_include_terms)
            ban_terms = _dedupe(ban_terms)
            avoid_terms = _dedupe(avoid_terms)

            def _prune_weird_tokens(s: str, max_chars: int = 80, max_words: int = 6) -> str:
                """Keep prompts as short tag lists; drop sentence-like fragments.

                This is intentionally strict: we want comma-separated tags, not prose.
                """
                parts = [p.strip() for p in (s or "").split(",")]
                out = []
                for p in parts:
                    if not p:
                        continue
                    pl = p.lower().strip()
                    # Drop sentence-y fragments and long clauses
                    # Drop sentence-y fragments and long clauses
                    if any(ch in p for ch in (".", "!", "?", ";", ":")):
                        continue
                    # Drop fragments that look like prose / descriptions
                    if pl.startswith(("a ", "an ", "the ")):
                        continue
                    if any(w in pl for w in (" standing", " wearing", " against ", " as in ", " maintaining", " changing", " in the ", " in a ")):
                        continue
                    if len(p) > max_chars:
                        continue
                    if len(p.split()) > max_words:
                        continue
                    out.append(p)
                return ", ".join(out).strip(" ,\n")

            def _canonicalise_csv_tags(s: str, max_words: int = 4, max_chars: int = 60) -> str:
                """Final normalisation pass to enforce tag-like comma-separated tokens."""
                parts = [p.strip() for p in (s or "").split(",")]
                out = []
                seen = set()
                for p in parts:
                    if not p:
                        continue
                    pl = p.lower().strip()
                    if any(ch in p for ch in (".", "!", "?", ";", ":")):
                        continue
                    if pl.startswith(("a ", "an ", "the ")):
                        continue
                    if any(w in pl for w in (" standing", " wearing", " against ", " as in ", " maintaining", " changing", " in the ", " in a ")):
                        continue
                    if len(p) > max_chars:
                        continue
                    if len(p.split()) > max_words:
                        continue
                    if pl in seen:
                        continue
                    seen.add(pl)
                    out.append(p)
                return ", ".join(out).strip(" ,\n")

            def _filter_csv_by_terms(s: str, forbidden_terms: List[str]) -> str:
                """Remove any comma-separated token that contains any forbidden term (substring match, catches plurals).
                
                Also checks individual words from multi-word forbidden terms (e.g., "summer dress" from "this lady wearing a summer dress").
                """
                if not s or not forbidden_terms:
                    return (s or "").strip(" ,\n")
                forb = [t.strip().lower() for t in forbidden_terms if (t or "").strip()]
                # Extract individual words from multi-word forbidden terms for partial matching
                # e.g., "this lady wearing a summer dress" -> also check for "summer" and "dress"
                forb_words = set()
                for t in forb:
                    words = t.split()
                    for w in words:
                        if len(w) > 3:  # Only meaningful words (skip "the", "a", etc.)
                            forb_words.add(w)
                
                parts = [p.strip() for p in (s or "").split(",")]
                out = []
                for p in parts:
                    if not p:
                        continue
                    pl = p.lower()
                    # Check if prompt part contains any full forbidden term
                    if any(t in pl for t in forb):
                        continue
                    # Also check if prompt part contains any word from a forbidden term
                    if any(w in pl for w in forb_words):
                        continue
                    out.append(p)
                return ", ".join(out).strip(" ,\n")

            def _remove_terms(text: str, terms: List[str]) -> str:
                if not text or not terms:
                    return text or ""
                out = text
                for t in terms:
                    tt = (t or "").strip()
                    if not tt:
                        continue
                    # Use word boundaries for single tokens; plain replace for phrases
                    if " " in tt:
                        out = re.sub(re.escape(tt), "", out, flags=re.IGNORECASE)
                    else:
                        out = re.sub(rf"\b{re.escape(tt)}\b", "", out, flags=re.IGNORECASE)
                out = re.sub(r"\s{2,}", " ", out)
                out = re.sub(r"\s*,\s*", ", ", out)
                return out.strip(" ,\n")

            # First prune obvious garbage (long, sentence-like fragments) before tag logic.
            # BUT: protect must_include terms from being pruned - they're critical.
            # Extract must_include terms first, prune, then add them back.
            protected_terms = []
            protected_words = set()  # Individual words from protected terms (for partial matching)
            for t in must_include_terms:
                t_str = str(t).strip()
                if t_str:
                    protected_terms.append(t_str)
                    # Also extract individual words for partial matching
                    # e.g., "lady wearing snowsuit" -> protect "snowsuit" if it appears alone
                    words = t_str.lower().split()
                    for w in words:
                        if len(w) > 3:  # Only protect meaningful words (skip "the", "a", etc.)
                            protected_words.add(w)
            
            # Temporarily remove protected terms, prune, then add them back
            # IMPORTANT: Use substring matching for multi-word terms like "lady wearing snowsuit"
            # Also protect individual words that appear in protected terms (e.g., "snowsuit" from "lady wearing snowsuit")
            temp_positive = improved_positive
            import re
            for pt in protected_terms:
                # Remove protected term temporarily (case-insensitive, substring match for phrases)
                pt_escaped = re.escape(pt)
                # For multi-word terms, use substring match; for single words, use word boundary
                if " " in pt:
                    temp_positive = re.sub(pt_escaped, "", temp_positive, flags=re.IGNORECASE)
                else:
                    temp_positive = re.sub(rf"\b{pt_escaped}\b", "", temp_positive, flags=re.IGNORECASE)
            
            # Also protect individual words from protected terms (e.g., "snowsuit" from "lady wearing snowsuit")
            # Split prompt into parts and protect any that contain protected words
            pos_parts_before_prune = [p.strip() for p in temp_positive.split(",") if p.strip()]
            protected_parts = []
            remaining_parts = []
            for part in pos_parts_before_prune:
                part_lower = part.lower()
                is_protected = False
                # Check if this part contains any protected word
                for pw in protected_words:
                    if pw in part_lower:
                        protected_parts.append(part)
                        is_protected = True
                        break
                if not is_protected:
                    remaining_parts.append(part)
            
            # Prune only the non-protected parts
            temp_positive = ", ".join(remaining_parts)
            improved_positive = _prune_weird_tokens(temp_positive)
            # Re-add protected parts (they'll be properly ordered later)
            if protected_parts:
                improved_positive = ", ".join(protected_parts + [improved_positive]).strip(" ,\n")
            improved_negative = _prune_weird_tokens(improved_negative)
            
            # Protected terms will be added back in correct order below

            for t in avoid_terms:
                improved_positive = re.sub(rf"\\b{re.escape(t)}\\b", "", improved_positive, flags=re.IGNORECASE)
            improved_positive = re.sub(r"\\s{2,}", " ", improved_positive).strip(" ,\n")

            # Ensure must-include terms appear in positive (prioritize change terms at front)
            # Separate change vs preserve intent terms
            change_must_include = []
            preserve_must_include = []
            for crit in criteria_defs:
                if not isinstance(crit, dict):
                    continue
                intent = str(crit.get("intent") or "preserve").strip().lower()
                terms = crit.get("must_include") or []
                if intent == "change":
                    change_must_include.extend(terms)
                else:
                    preserve_must_include.extend(terms)
            
            # Extract existing parts, filter out must_include terms (we'll re-add in correct order)
            pos_parts = [p.strip() for p in improved_positive.split(",") if p.strip()]
            all_must_include_lower = [str(t).strip().lower() for t in (change_must_include + preserve_must_include) if t]
            # Also extract individual words from must_include terms for partial matching
            # e.g., "lady wearing snowsuit" -> protect "snowsuit" if it appears alone
            must_include_words = set()
            for term in all_must_include_lower:
                words = term.split()
                for w in words:
                    if len(w) > 3:  # Only meaningful words (skip "the", "a", etc.)
                        must_include_words.add(w)
            
            filtered_parts = []
            protected_parts = []
            for part in pos_parts:
                part_lower = part.lower()
                # Check if this part matches a full must_include term
                is_must_include = any(term_lower in part_lower or part_lower in term_lower for term_lower in all_must_include_lower)
                # Also check if this part contains a word from a must_include term (partial match)
                contains_protected_word = any(pw in part_lower for pw in must_include_words)
                if is_must_include or contains_protected_word:
                    protected_parts.append(part)  # Keep this part, don't filter it
                else:
                    filtered_parts.append(part)
            
            # Add change terms first (for weight), then preserve terms, then rest
            # IMPORTANT: Always add must_include terms even if they appear in pos_parts,
            # because the AI might have removed them or they might be in a different form.
            # We'll deduplicate later, but we want to ensure they're at the front for weight.
            change_terms_to_add = []
            for t in change_must_include:
                t_str = str(t).strip()
                if t_str:
                    # Check if a close variant exists (substring match)
                    t_lower = t_str.lower()
                    found_variant = False
                    # First check protected_parts (already identified as must_include related)
                    for i, p in enumerate(protected_parts):
                        if t_lower in p.lower() or p.lower() in t_lower:
                            change_terms_to_add.append(p)
                            protected_parts.pop(i)
                            found_variant = True
                            break
                    # If not found, check filtered_parts
                    if not found_variant:
                        for i, p in enumerate(filtered_parts):
                            if t_lower in p.lower() or p.lower() in t_lower:
                                change_terms_to_add.append(p)
                                filtered_parts.pop(i)
                                found_variant = True
                                break
                    # If still not found, add the full term
                    if not found_variant:
                        change_terms_to_add.append(t_str)
                        self.logger.info(f"  Adding missing must_include term to positive prompt: {t_str}")
            
            preserve_terms_to_add = []
            for t in preserve_must_include:
                t_str = str(t).strip()
                if t_str:
                    t_lower = t_str.lower()
                    found_variant = False
                    # First check protected_parts
                    for i, p in enumerate(protected_parts):
                        if t_lower in p.lower() or p.lower() in t_lower:
                            preserve_terms_to_add.append(p)
                            protected_parts.pop(i)
                            found_variant = True
                            break
                    # If not found, check filtered_parts
                    if not found_variant:
                        for i, p in enumerate(filtered_parts):
                            if t_lower in p.lower() or p.lower() in t_lower:
                                preserve_terms_to_add.append(p)
                                filtered_parts.pop(i)
                                found_variant = True
                                break
                    # If still not found, add the full term
                    if not found_variant:
                        preserve_terms_to_add.append(t_str)
            
            # Deduplicate final list (case-insensitive)
            # Include protected_parts that weren't moved to change/preserve sections
            final_parts = []
            seen_lower = set()
            for part in (change_terms_to_add + preserve_terms_to_add + protected_parts + filtered_parts):
                part_lower = part.lower().strip()
                if part_lower and part_lower not in seen_lower:
                    seen_lower.add(part_lower)
                    final_parts.append(part)
            
            improved_positive = ", ".join(final_parts).strip(" ,\n")
            
            # Final safety check: ensure all change-intent must_include terms are in the positive prompt
            # This is a last resort in case they were removed by pruning or deduplication
            # Use word-level matching to avoid false positives (e.g., "summer" appearing alone doesn't mean "summer dress" is present)
            for t in change_must_include:
                t_str = str(t).strip()
                if t_str:
                    t_lower = t_str.lower()
                    # Extract key words from the must_include term (skip common words like "this", "lady", "wearing", "a")
                    t_words = [w for w in t_lower.split() if len(w) > 3 and w not in ("this", "lady", "wearing", "the", "that")]
                    # Check if ALL key words appear together in the prompt (not just individually)
                    # This prevents false matches like "summer" alone matching "summer dress"
                    prompt_lower = improved_positive.lower()
                    all_words_present = all(w in prompt_lower for w in t_words) if t_words else False
                    # Also check if the full phrase appears
                    full_phrase_present = t_lower in prompt_lower
                    
                    if not (full_phrase_present or (len(t_words) > 1 and all_words_present)):
                        # Force add it at the front for maximum weight
                        improved_positive = f"{t_str}, {improved_positive}".strip(" ,\n")
                        self.logger.warning(f"  Force-added missing must_include term to front of positive prompt: {t_str}")

            # Ensure banned terms appear in negative
            if ban_terms:
                neg = (improved_negative or "").strip()
                for t in ban_terms:
                    if t.lower() not in neg.lower():
                        if neg and not neg.endswith(','):
                            neg += ", "
                        neg += t
                improved_negative = neg.strip(" ,\n")

            # Sanity guards to prevent contradictory prompts:
            # - must_include should not be in negative
            # - ban_terms should not be in positive
            # Use a substring-based filter as well to catch plurals like "swimsuits"/"bikinis".
            improved_negative = _filter_csv_by_terms(improved_negative, must_include_terms)
            improved_positive = _filter_csv_by_terms(improved_positive, ban_terms)
            # Also keep avoid_terms out of positive as a stronger guard.
            improved_positive = _filter_csv_by_terms(improved_positive, avoid_terms)
            
            # Aggressive filter: if we have change-intent criteria with must_include terms,
            # remove any contradictory clothing terms from positive that match ban_terms or common opposites.
            # This prevents the AI from adding "formal business attire" when we want "snowsuit" or "summer dress".
            if change_must_include:
                # Build a list of terms to aggressively remove from positive
                contradictory_terms = list(ban_terms)  # Start with explicit ban_terms
                # Add common business/formal clothing terms if we're changing to casual/outdoor wear
                change_terms_lower = [str(t).strip().lower() for t in change_must_include]
                # Check for casual/outdoor wear terms (not just the original list)
                # Be specific: "summer dress" is casual, but "dress" alone could be formal
                is_casual_wear = any(
                    "snowsuit" in t or "swimsuit" in t or "bikini" in t or "hawaiian" in t or "coconut" in t or
                    "summer dress" in t or ("summer" in t and "dress" in t) or "casual" in t or "outdoor" in t
                    for t in change_terms_lower
                )
                if is_casual_wear:
                    contradictory_terms.extend([
                        "formal business attire", "business suit", "business attire", "formal attire",
                        "suit", "blazer", "dress shirt", "tie", "professional attire", "corporate attire",
                        "black blazer", "dark blue blazer", "gray blazer", "blue blazer", "navy blazer"
                    ])
                # Remove these from positive
                improved_positive = _filter_csv_by_terms(improved_positive, contradictory_terms)

            # Avoid dangling "no" tokens (these can confuse the image model and encourage drift).
            def _clean_tail(s: str) -> str:
                ss = (s or "").strip()
                if ss.lower().endswith(", no"):
                    ss = ss[:-4].strip(" ,\n")
                elif ss.lower().endswith(" no"):
                    # only remove if "no" is the last token
                    parts = ss.split()
                    if parts and parts[-1].lower() == "no":
                        ss = " ".join(parts[:-1]).strip(" ,\n")
                return ss

            def _strip_no_phrases_from_positive(pos: str) -> str:
                """Keep positive prompts positive.

                We frequently see LLMs emit lots of 'no X' fragments which weaken conditioning and
                lead to unstable results. Negations belong in the negative prompt.
                """
                import re

                p = (pos or "").strip()
                if not p:
                    return p
                # Remove common broken fragments
                p = re.sub(r"\bno\s+on\s+body\b", "", p, flags=re.IGNORECASE)
                # Remove other unhelpful fragments we’ve seen the improver emit
                p = re.sub(r"\bon\s+the\s+body\b", "", p, flags=re.IGNORECASE)
                p = re.sub(r"\bon\s+body\b", "", p, flags=re.IGNORECASE)
                p = re.sub(r"\btint\b", "", p, flags=re.IGNORECASE)
                # Remove 'no <word>' phrases
                p = re.sub(r"\bno\s+[a-zA-Z_]+\b", "", p, flags=re.IGNORECASE)
                # Remove standalone 'no'
                p = re.sub(r"\bno\b", "", p, flags=re.IGNORECASE)
                # Normalise separators/spaces
                p = re.sub(r"\s{2,}", " ", p)
                p = re.sub(r"\s*,\s*", ", ", p)
                p = re.sub(r"(,\s*){2,}", ", ", p)
                return p.strip(" ,\n")

            def _dedupe_csv(s: str) -> str:
                parts = [p.strip() for p in (s or "").split(",")]
                out = []
                seen = set()
                for p in parts:
                    if not p:
                        continue
                    k = p.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(p)
                return ", ".join(out).strip()

            improved_positive = _clean_tail(improved_positive)
            improved_negative = _clean_tail(improved_negative)
            improved_positive = _strip_no_phrases_from_positive(improved_positive)
            # Strip leading negation words from negative tokens (we want raw terms, not "no X" phrases).
            improved_negative = re.sub(r"(?i)\\b(no|not|without)\\s+", "", improved_negative).strip(" ,\n")
            improved_positive = _dedupe_csv(improved_positive)
            improved_negative = _dedupe_csv(improved_negative)

            # If we're preserving the background, aggressively remove common scene/background words
            # that are not present in the original description.
            try:
                od = (original_desc or "").lower()
            except Exception:
                od = ""
            preserve_background = any(
                isinstance(c, dict) and (
                    str(c.get("field") or "").lower() in ("background_identical", "background")
                    or "identical background" in ", ".join(_listify(c.get("must_include"))).lower()
                )
                for c in criteria_defs
            )
            if preserve_background and od:
                env_words = [
                    "forest", "jungle", "beach", "desert", "mountain", "snow", "street", "city",
                    "bedroom", "kitchen", "bathroom", "studio", "office", "conference", "park",
                ]
                def _filter_env(s: str) -> str:
                    parts = [p.strip() for p in (s or "").split(",")]
                    out = []
                    for p in parts:
                        if not p:
                            continue
                        pl = p.lower()
                        if any(w in pl for w in env_words) and not any(w in od for w in env_words if w in pl):
                            # If token mentions an environment keyword not found in original description, drop it.
                            # (e.g. "forest backdrop" when original is an office.)
                            continue
                        out.append(p)
                    return ", ".join(out).strip(" ,\n")
                improved_positive = _filter_env(improved_positive)
                improved_negative = _filter_env(improved_negative)

            # Final canonicalisation to keep prompts tag-like.
            improved_positive = _canonicalise_csv_tags(improved_positive)
            improved_negative = _canonicalise_csv_tags(improved_negative)

            # Keep negatives focused: if we have explicit ban_terms, don't keep extra fluff.
            if ban_terms:
                improved_negative = _canonicalise_csv_tags(", ".join(ban_terms + _listify(improved_negative.split(","))))

            # Log a concise diff so it's clear what changed iteration-to-iteration.
            try:
                pos_diff = _diff_tokens(current_positive, improved_positive)
                neg_diff = _diff_tokens(current_negative, improved_negative)
                # Also show which tag-derived terms were used to shape prompts
                if must_include_terms:
                    self.logger.info(f"  Tag must_include terms (from failed criteria): {must_include_terms}")
                if ban_terms:
                    self.logger.info(f"  Tag ban_terms (from failed criteria): {ban_terms}")
                if avoid_terms:
                    self.logger.info(f"  Tag avoid_terms (from failed criteria): {avoid_terms}")

                if pos_diff["added"] or pos_diff["removed"]:
                    self.logger.info(f"  Positive prompt changes: +{pos_diff['added']} -{pos_diff['removed']}")
                if neg_diff["added"] or neg_diff["removed"]:
                    self.logger.info(f"  Negative prompt changes: +{neg_diff['added']} -{neg_diff['removed']}")

                # In verbose mode, include the full before/after (trimmed to keep logs readable).
                if self.verbose:
                    def _trim(s: str, n: int = 500) -> str:
                        ss = (s or "").strip()
                        return ss if len(ss) <= n else ss[:n] + " …"
                    self.logger.debug(f"  Positive before: {_trim(current_positive)}")
                    self.logger.debug(f"  Positive after:  {_trim(improved_positive)}")
                    self.logger.debug(f"  Negative before: {_trim(current_negative)}")
                    self.logger.debug(f"  Negative after:  {_trim(improved_negative)}")
            except Exception:
                pass
            
            if 'prompts' not in aigen_config:
                aigen_config['prompts'] = {}
            aigen_config['prompts']['positive'] = improved_positive
            aigen_config['prompts']['negative'] = improved_negative
        
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
    parser = argparse.ArgumentParser(
        description="Iterative Imagination - AI-powered image generation with iterative improvement"
    )
    parser.add_argument('--project', type=str, help='Project name (uses projects/{NAME}/config/rules.yaml)')
    parser.add_argument('--rules', type=str, help='Path to rules.yaml (alternative to --project)')
    parser.add_argument('--input', type=str, help='Path to input image (if not using project structure)')
    parser.add_argument('--dry-run', action='store_true', help='Validate configs but don\'t run')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--resume-from', type=int, help='Resume from a specific iteration number (or use checkpoint if not specified)')
    # New naming (preferred)
    parser.add_argument('--seed-from-ranking', type=str, help='Seed from a prior run human ranking (RUN_ID or "latest")')
    parser.add_argument('--seed-ranking-mode', type=str, default="rank1", help='Seeding mode: rank1|top3|top5 (default: rank1)')
    # Backwards compatible alias (deprecated)
    parser.add_argument('--seed-from-human', type=str, help='(deprecated) alias for --seed-from-ranking')
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset run state for this project (deletes working/checkpoint.json and resets working/AIGen.yaml from config)'
    )
    
    args = parser.parse_args()
    
    # Determine project name
    if args.project:
        project_name = args.project
    elif args.rules:
        rules_path = Path(args.rules)
        if 'projects' in rules_path.parts:
            project_idx = rules_path.parts.index('projects')
            if project_idx + 1 < len(rules_path.parts):
                project_name = rules_path.parts[project_idx + 1]
            else:
                print("Error: Could not determine project name from rules path", file=sys.stderr)
                sys.exit(1)
        else:
            print("Error: Rules path must be within projects/ directory", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Must specify --project or --rules", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    try:
        app = IterativeImagination(project_name, verbose=args.verbose)
        if args.reset:
            # Remove checkpoint and reset working AIGen.yaml back to project config (or defaults via loader).
            chk = app.project.get_checkpoint_path()
            try:
                if chk.exists():
                    chk.unlink()
                    print(f"Reset: removed {chk}")
            except Exception as e:
                print(f"Warning: failed to remove checkpoint {chk}: {e}", file=sys.stderr)

            # Archive any legacy flat iteration_* files into a timestamped run folder so old runs stay tidy.
            try:
                legacy_dir = app.project.project_root / "working"
                legacy_files = list(legacy_dir.glob("iteration_*.*"))
                if legacy_files:
                    run_id = app.project.create_run_id()
                    app.project.ensure_run_directories(run_id)
                    run_root = app.project.get_run_root(run_id)

                    def _dest_for(p: Path) -> Path:
                        name = p.name
                        if name.endswith(".png"):
                            return run_root / "images" / name
                        if name.endswith("_questions.json"):
                            return run_root / "questions" / name
                        if name.endswith("_evaluation.json"):
                            return run_root / "evaluation" / name
                        if name.endswith("_comparison.json"):
                            return run_root / "comparison" / name
                        if name.endswith("_metadata.json"):
                            return run_root / "metadata" / name
                        return run_root / name

                    for p in legacy_files:
                        dest = _dest_for(p)
                        p.rename(dest)
                    print(f"Reset: archived {len(legacy_files)} legacy iteration files into {run_root}")
            except Exception as e:
                print(f"Warning: failed to archive legacy iteration files: {e}", file=sys.stderr)

            # Reset working/AIGen.yaml to match config/AIGen.yaml if present, otherwise rely on defaults loader.
            working_aigen = app.project.project_root / "working" / "AIGen.yaml"
            config_aigen = app.project.project_root / "config" / "AIGen.yaml"
            try:
                if config_aigen.exists():
                    import shutil as _shutil
                    _shutil.copy2(config_aigen, working_aigen)
                    print(f"Reset: restored {working_aigen} from {config_aigen}")
                else:
                    # Ensure loader will rebuild working/AIGen.yaml from defaults on next run.
                    if working_aigen.exists():
                        working_aigen.unlink()
                        print(f"Reset: removed {working_aigen} (will be recreated from defaults)")
            except Exception as e:
                print(f"Warning: failed to reset working AIGen.yaml: {e}", file=sys.stderr)

            # Remove progress.png so the run starts from the original input.png
            progress_path = app.project.project_root / "input" / "progress.png"
            try:
                if progress_path.exists():
                    progress_path.unlink()
                    print(f"Reset: removed {progress_path} (will start from original input.png)")
            except Exception as e:
                print(f"Warning: failed to remove progress.png: {e}", file=sys.stderr)

            # Note: we deliberately do NOT delete iteration_*.png/json so you keep history.
            # If you want a totally clean slate, delete projects/{project}/working/iteration_* manually.

        seed_run = args.seed_from_ranking or args.seed_from_human
        success = app.run(
            dry_run=args.dry_run,
            resume_from=args.resume_from,
            seed_from_ranking=seed_run,
            seed_ranking_mode=args.seed_ranking_mode,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
