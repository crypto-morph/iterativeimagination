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
    
    def _evaluate_acceptance_criteria(self, image_path: str, question_answers: Dict) -> Dict:
        """Evaluate image against acceptance criteria."""
        self.logger.info("Evaluating acceptance criteria...")
        criteria = self.rules.get('acceptance_criteria', [])
        original_desc = self._describe_original_image()
        
        evaluation = self.aivis.evaluate_acceptance_criteria(
            image_path, original_desc, criteria, question_answers
        )
        
        # Ensure criteria_results has all fields
        criteria_results = evaluation.get('criteria_results', {})
        for criterion in criteria:
            field = criterion['field']
            if field not in criteria_results:
                criteria_results[field] = False
        
        evaluation['criteria_results'] = criteria_results
        return evaluation
    
    def _run_iteration(self, iteration_num: int) -> Optional[Dict]:
        """Run a single iteration of generation and evaluation."""
        self.logger.info(f"{'='*60}\nIteration {iteration_num}\n{'='*60}")
        
        # Load current AIGen.yaml
        aigen_config = self.project.load_aigen_config()
        
        # Prepare input image for ComfyUI
        input_filename = self.project.prepare_input_image(
            self.input_image_path, self.comfyui_input_dir
        )
        
        # Load and update workflow
        workflow = WorkflowManager.load_workflow(
            aigen_config['workflow_file'], self.project.project_root
        )
        updated_workflow = WorkflowManager.update_workflow(
            workflow, aigen_config, input_filename
        )
        
        # Queue workflow
        self.logger.info("Queueing workflow to ComfyUI...")
        try:
            prompt_id = self.comfyui.queue_prompt(updated_workflow)
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
        for attempt in range(8):
            try:
                time.sleep(1 if attempt > 0 else 0.5)
                history = self.comfyui.get_history(prompt_id)
                
                if prompt_id in history:
                    execution = history[prompt_id]
                    outputs = execution.get('outputs', {})
                    
                    if outputs:
                        for node_id, node_output in outputs.items():
                            if 'images' in node_output and len(node_output['images']) > 0:
                                image_info = node_output['images'][0]
                                break
                        if image_info:
                            break
            except Exception:
                if attempt < 7:
                    continue
        
        if not image_info:
            self.logger.error("No image found in workflow outputs")
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
        
        evaluation = self._evaluate_acceptance_criteria(str(paths['image']), questions)
        
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
        if 'seed' in parameters_used and parameters_used['seed'] is None:
            for node_id, node_data in updated_workflow.items():
                if isinstance(node_data, dict) and node_data.get('class_type') == 'KSampler':
                    parameters_used['seed'] = node_data['inputs'].get('seed')
                    break
        
        # Capture prompts used for this iteration
        prompts_used = aigen_config.get('prompts', {}).copy()
        
        metadata = {
            "iteration": iteration_num,
            "timestamp": time.time(),
            "image_path": str(paths['image'].relative_to(self.project.project_root)),
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
        
        # Adjust parameters based on similarity
        params = aigen_config.get('parameters', {})
        current_denoise = params.get('denoise', 0.5)
        current_cfg = params.get('cfg', 7.0)
        
        if similarity > 0.85:
            params['denoise'] = min(1.0, current_denoise + 0.05)
            params['cfg'] = min(20.0, current_cfg + 1.0)
            self.logger.info(f"Images too similar - increasing denoise to {params['denoise']:.2f}, cfg to {params['cfg']:.1f}")
        elif similarity < 0.3:
            params['denoise'] = max(0.1, current_denoise - 0.05)
            self.logger.info(f"Images too different - decreasing denoise to {params['denoise']:.2f}")
        
        # Improve prompts based on failed criteria (data-driven via rules.yaml tags)
        criteria_defs = self.rules.get('acceptance_criteria', []) or []
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

        if failed_change:
            # If any "change" goal is failing, increase denoise and cfg to allow stronger edits.
            max_strength = max(_strength_value(criteria_by_field[f].get('edit_strength')) for f in failed_change)
            # Target a higher minimum denoise as edit strength increases.
            desired_min_denoise = min(0.90, 0.70 + 0.05 * max_strength)  # ~0.72..0.78 typical
            if params.get('denoise', current_denoise) < desired_min_denoise:
                bump = 0.05 + 0.03 * max_strength
                params['denoise'] = min(0.95, max(desired_min_denoise, current_denoise + bump))
                self.logger.info(
                    f"Change goals failing ({failed_change}) - increasing denoise to {params['denoise']:.2f}"
                )
            # Increase CFG modestly to strengthen prompt guidance.
            desired_min_cfg = 8.5 + 0.5 * max_strength
            if params.get('cfg', current_cfg) < desired_min_cfg:
                params['cfg'] = min(20.0, max(desired_min_cfg, current_cfg + (0.5 + 0.5 * max_strength)))
                self.logger.info(
                    f"Change goals failing ({failed_change}) - increasing cfg to {params['cfg']:.1f}"
                )

        if failed_preserve and not failed_change:
            # If only "preserve" goals are failing, reduce denoise a little to keep closer to original.
            max_strength = max(_strength_value(criteria_by_field[f].get('edit_strength')) for f in failed_preserve)
            bump = 0.03 + 0.03 * max_strength
            params['denoise'] = max(0.10, current_denoise - bump)
            self.logger.info(
                f"Preserve goals failing ({failed_preserve}) - decreasing denoise to {params['denoise']:.2f}"
            )
        
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
            current_positive = aigen_config.get('prompts', {}).get('positive', '')
            current_negative = aigen_config.get('prompts', {}).get('negative', '')
            
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

            for t in avoid_terms:
                improved_positive = re.sub(rf"\\b{re.escape(t)}\\b", "", improved_positive, flags=re.IGNORECASE)
            improved_positive = re.sub(r"\\s{2,}", " ", improved_positive).strip(" ,\n")

            # Ensure must-include terms appear in positive
            for t in must_include_terms:
                if t.lower() not in improved_positive.lower():
                    improved_positive = (improved_positive + f", {t}").strip(" ,\n")

            # Ensure banned terms appear in negative
            if ban_terms:
                neg = (improved_negative or "").strip()
                for t in ban_terms:
                    if t.lower() not in neg.lower():
                        if neg and not neg.endswith(','):
                            neg += ", "
                        neg += t
                improved_negative = neg.strip(" ,\n")
            
            if 'prompts' not in aigen_config:
                aigen_config['prompts'] = {}
            aigen_config['prompts']['positive'] = improved_positive
            aigen_config['prompts']['negative'] = improved_negative
        
        # Save updated AIGen.yaml
        self.project.save_aigen_config(aigen_config)
    
    def run(self, dry_run: bool = False, resume_from: Optional[int] = None):
        """Run the iterative improvement loop.
        
        Args:
            dry_run: If True, don't actually run iterations
            resume_from: Resume from a specific iteration number (None = start from beginning or checkpoint)
        """
        self.logger.info(f"{'='*60}\nIterative Imagination\n{'='*60}")
        self.logger.info(f"Project: {self.project_name}")
        self.logger.info(f"Input image: {self.input_image_path}")
        self.logger.info(f"Max iterations: {self.rules['project']['max_iterations']}\n{'='*60}")
        
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
                shutil.copy2(self.project.get_iteration_paths(iteration)['image'], output_paths['image'])
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

            # Note: we deliberately do NOT delete iteration_*.png/json so you keep history.
            # If you want a totally clean slate, delete projects/{project}/working/iteration_* manually.

        success = app.run(dry_run=args.dry_run, resume_from=args.resume_from)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
