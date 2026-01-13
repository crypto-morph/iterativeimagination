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
from core.services.iteration_orchestrator import IterationOrchestrator
from core.services.multi_mask_sequencer import MultiMaskSequencer
from core.services.criteria_filter import filter_criteria_for_mask
from core.constants import (
    INPAINTING_DENOISE_MIN,
    INPAINTING_CFG_MIN,
    INPAINTING_DENOISE_THRESHOLD,
    INPAINTING_CFG_THRESHOLD,
)


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
        
        # Initialize iteration orchestrator
        # WorkflowManager uses static methods, so we pass the class
        self.orchestrator = IterationOrchestrator(
            logger=self.logger,
            project=self.project,
            comfyui_client=self.comfyui,
            aivis_client=self.aivis,
            evaluator=self.evaluator,
            workflow_manager=WorkflowManager,  # Static methods, so class is fine
            input_image_path=self.input_image_path,
            comfyui_input_dir=self.comfyui_input_dir,
            rules=self.rules,
            run_id=None,  # Will be set in run()
        )
        
        # Initialize multi-mask sequencer
        self.mask_sequencer = MultiMaskSequencer(
            logger=self.logger,
            project=self.project,
            rules=self.rules,
        )
        
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


    def _get_inpainting_boost_values(self) -> dict:
        """Get inpainting boost values from rules.yaml project section.
        
        Returns dict with keys: denoise_min, cfg_min, denoise_threshold, cfg_threshold
        """
        rules = self.rules if isinstance(self.rules, dict) else {}
        project = rules.get("project") or {}
        
        return {
            "denoise_min": float(project.get("inpainting_denoise_min", INPAINTING_DENOISE_MIN)),
            "cfg_min": float(project.get("inpainting_cfg_min", INPAINTING_CFG_MIN)),
            "denoise_threshold": float(project.get("inpainting_denoise_threshold", INPAINTING_DENOISE_THRESHOLD)),
            "cfg_threshold": float(project.get("inpainting_cfg_threshold", INPAINTING_CFG_THRESHOLD)),
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
        # Load current AIGen.yaml
        aigen_config = self.project.load_aigen_config()
        project_cfg = (self.rules.get("project") or {}) if isinstance(self.rules, dict) else {}
        
        # Handle seed locking
        lock_seed = bool(project_cfg.get("lock_seed"))
        if lock_seed:
            params = aigen_config.get("parameters") or {}
            seed = params.get("seed")
            if seed is None:
                if self.locked_seed is not None:
                    params["seed"] = int(self.locked_seed)
                    aigen_config["parameters"] = params
                    self.project.save_aigen_config(aigen_config)
            else:
                try:
                    self.locked_seed = int(seed)
                except Exception:
                    pass
        
        # Update orchestrator run_id
        self.orchestrator.run_id = self.run_id
        
        # Get inpainting boost config
        boost_config = self._get_inpainting_boost_values()
        
        # Execute iteration via orchestrator
        metadata = self.orchestrator.execute_iteration(
            iteration_num=iteration_num,
            aigen_config=aigen_config,
            project_cfg=project_cfg,
            inpainting_boost_config=boost_config,
            locked_seed=self.locked_seed,
            get_rules_base_prompts_fn=self._get_rules_base_prompts,
        )
        
        # Update locked_seed if it was extracted from workflow
        if metadata and lock_seed:
            used_seed = metadata.get("parameters_used", {}).get("seed")
            if used_seed is not None and self.locked_seed is None:
                self.locked_seed = used_seed
        
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
        criteria_defs = filter_criteria_for_mask(all_criteria_defs, active_mask, self.rules)
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
        project_cfg = (self.rules.get("project") or {}) if isinstance(self.rules, dict) else {}
        aivis_param_recommendations_enabled = project_cfg.get("aivis_parameter_recommendations", False)
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
                    
                    # Get current parameters
                    params = aigen_config.get("parameters", {})
                    cur_d = float(params.get("denoise", 0.5))
                    cur_cfg = float(params.get("cfg", 7.0))
                    
                    # Get parameter bounds
                    denoise_min = float(project_cfg.get("denoise_min", 0.20))
                    denoise_max = float(project_cfg.get("denoise_max", 0.80))
                    cfg_min = float(project_cfg.get("cfg_min", 4.0))
                    cfg_max = float(project_cfg.get("cfg_max", 12.0))
                    
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
                    from core.constants import CONFIDENCE_THRESHOLD
                    
                    if denoise_conf >= CONFIDENCE_THRESHOLD:
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
                    
                    if cfg_conf >= CONFIDENCE_THRESHOLD:
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
                    
                    # Update aigen_config with modified params
                    aigen_config["parameters"] = params
                    
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
        improve_prompts_enabled = project_cfg.get("improve_prompts", False)
        
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
        mask_sequence = self.mask_sequencer.detect_mask_sequence()
        
        # If we have a mask sequence, cycle through them sequentially
        if mask_sequence:
            self.logger.info(f"Multi-mask mode: will cycle through {len(mask_sequence)} mask(s): {[m.name for m in mask_sequence]}")
            iterations_run = 0
            last_iteration_run: Optional[int] = None
            current_iteration = start_iteration
            mask_index = 0
            boost_config = self._get_inpainting_boost_values()
            
            while mask_index < len(mask_sequence) and current_iteration <= max_iterations:
                current_mask = mask_sequence[mask_index]
                mask_name = current_mask.name
                mask_criterion = current_mask.criterion
                
                self.logger.info(f"{'='*60}\nProcessing mask: {mask_name} (criterion: {mask_criterion})\n{'='*60}")
                
                # Prepare mask iteration (set active_mask and boost parameters)
                self.mask_sequencer.prepare_mask_iteration(current_mask, boost_config)
                
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
                        self.mask_sequencer.update_progress_image(iteration, self.run_id)
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
