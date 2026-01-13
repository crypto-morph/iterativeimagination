"""Iteration orchestration - coordinates workflow execution and evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

from core.services.criteria_filter import filter_criteria_for_mask
from core.services.mask_resolver import MaskResolver, MaskInfo


class IterationOrchestrator:
    """Orchestrates a single iteration: preparation, execution, evaluation, and metadata collection."""

    def __init__(
        self,
        logger,
        project,
        comfyui_client,
        aivis_client,
        evaluator,
        workflow_manager,
        input_image_path: Path,
        comfyui_input_dir: Path,
        rules: Dict,
        run_id: Optional[str] = None,
    ):
        self.logger = logger
        self.project = project
        self.comfyui = comfyui_client
        self.aivis = aivis_client
        self.evaluator = evaluator
        self.workflow_manager = workflow_manager
        self.input_image_path = input_image_path
        self.comfyui_input_dir = comfyui_input_dir
        self.rules = rules
        self.run_id = run_id

        # Create mask resolver
        self.mask_resolver = MaskResolver(
            logger=logger,
            project_root=project.project_root,
            comfyui_input_dir=comfyui_input_dir,
            prepare_image_fn=project.prepare_input_image,
        )

    def execute_iteration(
        self,
        iteration_num: int,
        aigen_config: Dict,
        project_cfg: Dict,
        inpainting_boost_config: Optional[Dict] = None,
        locked_seed: Optional[int] = None,
        get_rules_base_prompts_fn=None,
    ) -> Optional[Dict]:
        """Execute a complete iteration: prepare, generate, evaluate, and save metadata.
        
        Args:
            iteration_num: Iteration number
            aigen_config: Current AIGen configuration
            project_cfg: Project configuration from rules.yaml
            inpainting_boost_config: Optional inpainting boost configuration
            locked_seed: Optional locked seed value
            get_rules_base_prompts_fn: Function to get base prompts from rules
            
        Returns:
            Metadata dict for the iteration, or None if execution failed
        """
        self.logger.info(f"{'='*60}\nIteration {iteration_num}\n{'='*60}")

        # Handle seed locking
        lock_seed = bool(project_cfg.get("lock_seed"))
        if lock_seed and locked_seed is not None:
            params = aigen_config.get("parameters") or {}
            if params.get("seed") is None:
                params["seed"] = int(locked_seed)
                aigen_config["parameters"] = params
                self.project.save_aigen_config(aigen_config)

        # Prepare input image
        input_filename = self.project.prepare_input_image(
            self.input_image_path, self.comfyui_input_dir
        )

        # Resolve mask
        mask_info = self.mask_resolver.resolve_mask(
            aigen_config, inpainting_boost_config
        )

        # Handle seed clearing for inpainting mode
        inpaint_mode = bool(mask_info.filename)
        if inpaint_mode and not bool(project_cfg.get("lock_seed_inpaint", False)):
            lock_seed = False
            params = aigen_config.get("parameters", {})
            if "seed" in params and params.get("seed") is not None:
                params["seed"] = None
                aigen_config["parameters"] = params
                self.project.save_aigen_config(aigen_config)
                self.logger.info("Cleared seed for inpainting mode to allow variation")

        # Resolve control image
        control_filename = self._resolve_control_image()

        # Initialise prompts if empty
        self._initialise_prompts_if_empty(
            aigen_config, mask_info.active_mask_name, get_rules_base_prompts_fn
        )

        # Prepare workflow
        workflow_path, updated_workflow, ckpt_used = self._prepare_workflow(
            aigen_config, mask_info.filename, control_filename
        )

        # Execute workflow
        image_info = self._execute_workflow(updated_workflow)
        if not image_info:
            return None

        # Download and save generated image
        paths = self.project.get_iteration_paths(iteration_num, run_id=self.run_id)
        image_data = self.comfyui.download_image(
            image_info["filename"],
            image_info.get("subfolder", ""),
            image_info.get("type", "output"),
        )
        with open(paths["image"], "wb") as f:
            f.write(image_data)
        self.logger.info(f"Saved image: {paths['image']}")

        # Evaluate and compare
        questions = self.evaluator.answer_questions(str(paths["image"]))
        all_criteria = self.rules.get("acceptance_criteria", []) or []
        scoped_criteria = filter_criteria_for_mask(
            all_criteria, mask_info.active_mask_name, self.rules
        )
        evaluation = self.evaluator.evaluate_acceptance_criteria(
            str(paths["image"]), questions, criteria=scoped_criteria
        )
        comparison = self.aivis.compare_images(
            str(self.input_image_path), str(paths["image"])
        )

        score = evaluation.get("overall_score", 0)
        similarity = comparison.get("similarity_score", 0.5)
        self.logger.info(f"Evaluation score: {score}%")
        self.logger.info(f"Similarity score: {similarity:.2f}")

        # Collect metadata
        aivis_metadata = {
            "evaluation": evaluation.get("_metadata", {}),
            "comparison": comparison.get("_metadata", {}),
            "questions": questions.get("_metadata", {}),
        }

        # Save evaluation files
        self._save_evaluation_files(paths, questions, evaluation, comparison)

        # Extract used seed from workflow if needed
        used_seed = self._extract_used_seed(
            aigen_config, updated_workflow, lock_seed, locked_seed
        )

        # Build and save metadata
        metadata = self._build_metadata(
            iteration_num,
            paths,
            workflow_path,
            ckpt_used,
            mask_info,
            control_filename,
            aigen_config,
            questions,
            evaluation,
            comparison,
            aivis_metadata,
            used_seed,
        )

        with open(paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _resolve_control_image(self) -> Optional[str]:
        """Resolve optional ControlNet control image."""
        try:
            control_path = self.project.project_root / "input" / "control.png"
            if control_path.exists():
                return self.project.prepare_input_image(
                    control_path, self.comfyui_input_dir
                )
        except Exception:
            pass
        return None

    def _initialise_prompts_if_empty(
        self,
        aigen_config: Dict,
        active_mask_name: Optional[str],
        get_rules_base_prompts_fn,
    ) -> None:
        """Initialise prompts from rules.yaml if AIGen.yaml prompts are empty."""
        if not get_rules_base_prompts_fn:
            return

        prompts_cfg = aigen_config.get("prompts") or {}
        if not isinstance(prompts_cfg, dict):
            prompts_cfg = {}

        cur_pos = str(prompts_cfg.get("positive") or "").strip()
        cur_neg = str(prompts_cfg.get("negative") or "").strip()

        if not cur_pos or not cur_neg:
            scope_for_prompts = active_mask_name if active_mask_name else None
            base_pos, base_neg = get_rules_base_prompts_fn(scope_for_prompts)

            # Validate: reject prompts that look like field names
            def _looks_like_field_names(text: str) -> bool:
                if not text:
                    return False
                all_fields = {
                    str(c.get("field", "")).strip()
                    for c in (self.rules.get("acceptance_criteria") or [])
                    if isinstance(c, dict)
                }
                words = [
                    w.strip().lower()
                    for w in text.replace(",", " ").split()
                    if w.strip()
                ]
                if not words:
                    return False
                matches = sum(
                    1
                    for w in words
                    if w in all_fields or w.replace("not_", "") in all_fields
                )
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
                    f"Initialised prompts from rules.yaml prompts "
                    f"(scope={scope_for_prompts or 'default/global'})."
                )

    def _prepare_workflow(
        self,
        aigen_config: Dict,
        mask_filename: Optional[str],
        control_filename: Optional[str],
    ) -> tuple[str, Dict, str]:
        """Prepare workflow: load, switch if needed, and update with parameters."""
        workflow_path = aigen_config.get("workflow_file")
        workflow_switched = False

        # Switch to inpaint workflow if mask is present
        if mask_filename:
            wf_name = str(workflow_path or "")
            if "inpaint" not in wf_name.lower():
                candidate = (
                    self.project.project_root / "workflow" / "img2img_inpaint_api.json"
                )
                if candidate.exists():
                    workflow_path = "workflow/img2img_inpaint_api.json"
                    workflow_switched = True
                else:
                    workflow_path = "defaults/workflow/img2img_inpaint_api.json"
                    workflow_switched = True

                if workflow_switched:
                    aigen_config["workflow_file"] = workflow_path
                    try:
                        self.project.save_aigen_config(aigen_config)
                        self.logger.info(f"Switched to inpaint workflow: {workflow_path}")
                    except Exception:
                        pass

        # Load and update workflow
        workflow = self.workflow_manager.load_workflow(
            workflow_path, self.project.project_root
        )
        updated_workflow = self.workflow_manager.update_workflow(
            workflow,
            aigen_config,
            self.project.prepare_input_image(self.input_image_path, self.comfyui_input_dir),
            mask_image_path=mask_filename,
            control_image_path=control_filename,
        )

        # Extract checkpoint info
        wf_meta = updated_workflow.get("_workflow_metadata", {})
        ckpt_used = wf_meta.get("checkpoint_used", "unknown")
        ckpt_switched = wf_meta.get("checkpoint_switched", False)
        if ckpt_switched:
            self.logger.info(f"Using inpainting checkpoint: {ckpt_used}")
        else:
            self.logger.info(f"Using checkpoint: {ckpt_used}")

        return workflow_path, updated_workflow, ckpt_used

    def _execute_workflow(self, workflow: Dict) -> Optional[Dict]:
        """Execute workflow in ComfyUI and retrieve result image."""
        # Remove metadata before sending to ComfyUI
        workflow_for_comfyui = {
            k: v for k, v in workflow.items() if k != "_workflow_metadata"
        }

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
            self.logger.warning(
                "Workflow completion timeout, attempting to fetch result anyway..."
            )

        # Get result
        self.logger.info("Fetching result...")
        image_info = None
        last_error = None
        for attempt in range(8):
            try:
                time.sleep(1 if attempt > 0 else 0.5)
                history = self.comfyui.get_history(prompt_id)

                # Parse history response
                if isinstance(history, dict):
                    if prompt_id in history:
                        execution = history[prompt_id]
                    elif len(history) == 1:
                        execution = list(history.values())[0]
                    else:
                        execution = history
                else:
                    execution = history

                if not isinstance(execution, dict):
                    last_error = f"Unexpected history format: {type(execution)}"
                    continue

                # Check for status/error
                status = execution.get("status", {})
                if isinstance(status, dict):
                    if status.get("completed", False) is False:
                        if status.get("error"):
                            last_error = f"Workflow error: {status.get('error')}"
                            self.logger.error(last_error)
                            return None
                        continue

                outputs = execution.get("outputs", {})

                if not outputs:
                    if "images" in execution:
                        if len(execution["images"]) > 0:
                            image_info = execution["images"][0]
                            break
                    last_error = f"No outputs found in execution. Keys: {list(execution.keys())}"
                    continue

                # Look for SaveImage node outputs
                for node_id, node_output in outputs.items():
                    if not isinstance(node_output, dict):
                        continue
                    if "images" in node_output and len(node_output["images"]) > 0:
                        image_info = node_output["images"][0]
                        self.logger.info(
                            f"Found image in node {node_id}: "
                            f"{image_info.get('filename', 'unknown')}"
                        )
                        break

                if image_info:
                    break

            except Exception as e:
                last_error = f"Exception fetching history: {e}"
                self.logger.debug(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < 7:
                    continue

        if not image_info:
            error_msg = "No image found in workflow outputs"
            if last_error:
                error_msg += f" (last error: {last_error})"
            self.logger.error(error_msg)
            try:
                history = self.comfyui.get_history(prompt_id)
                self.logger.debug(
                    f"History response: {json.dumps(history, indent=2)[:500]}"
                )
            except Exception as e:
                self.logger.debug(f"Could not fetch history for debugging: {e}")
            return None

        return image_info

    def _save_evaluation_files(
        self, paths: Dict[str, Path], questions: Dict, evaluation: Dict, comparison: Dict
    ) -> None:
        """Save evaluation files (questions, evaluation, comparison)."""
        with open(paths["questions"], "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2)
        with open(paths["evaluation"], "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)
        with open(paths["comparison"], "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

    def _extract_used_seed(
        self,
        aigen_config: Dict,
        updated_workflow: Dict,
        lock_seed: bool,
        locked_seed: Optional[int],
    ) -> Optional[int]:
        """Extract the seed that was actually used in the workflow."""
        parameters_used = aigen_config.get("parameters", {}).copy()
        used_seed: Optional[int] = None

        if "seed" in parameters_used and parameters_used["seed"] is None:
            # Seed was None, extract from workflow
            for node_id, node_data in updated_workflow.items():
                if (
                    isinstance(node_data, dict)
                    and node_data.get("class_type") == "KSampler"
                ):
                    try:
                        used_seed = int(node_data["inputs"].get("seed"))
                    except Exception:
                        used_seed = None
                    parameters_used["seed"] = used_seed
                    break
        else:
            try:
                used_seed = int(parameters_used.get("seed"))
            except Exception:
                used_seed = None

        # Lock seed if enabled and not already locked
        if lock_seed and locked_seed is None and used_seed is not None:
            try:
                a2 = self.project.load_aigen_config()
                p2 = a2.get("parameters") or {}
                if p2.get("seed") is None:
                    p2["seed"] = used_seed
                    a2["parameters"] = p2
                    self.project.save_aigen_config(a2)
            except Exception:
                pass

        return used_seed

    def _build_metadata(
        self,
        iteration_num: int,
        paths: Dict[str, Path],
        workflow_path: str,
        ckpt_used: str,
        mask_info: MaskInfo,
        control_filename: Optional[str],
        aigen_config: Dict,
        questions: Dict,
        evaluation: Dict,
        comparison: Dict,
        aivis_metadata: Dict,
        used_seed: Optional[int],
    ) -> Dict:
        """Build iteration metadata dict."""
        parameters_used = aigen_config.get("parameters", {}).copy()
        if used_seed is not None:
            parameters_used["seed"] = used_seed

        # Remove _metadata from results for cleaner output
        questions_clean = {k: v for k, v in questions.items() if k != "_metadata"}
        evaluation_clean = {k: v for k, v in evaluation.items() if k != "_metadata"}
        comparison_clean = {k: v for k, v in comparison.items() if k != "_metadata"}

        return {
            "iteration": iteration_num,
            "timestamp": time.time(),
            "image_path": str(paths["image"].relative_to(self.project.project_root)),
            "workflow_file_used": str(workflow_path),
            "checkpoint_used": ckpt_used,
            "mask_used": mask_info.filename,
            "active_mask": mask_info.active_mask_name,
            "control_used": control_filename,
            "parameters_used": parameters_used,
            "prompts_used": aigen_config.get("prompts", {}).copy(),
            "questions": questions_clean,
            "evaluation": evaluation_clean,
            "comparison": comparison_clean,
            "aivis_metadata": aivis_metadata,
            "run_id": self.run_id,
        }
