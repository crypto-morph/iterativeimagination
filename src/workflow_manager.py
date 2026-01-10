#!/usr/bin/env python3
"""
Workflow Manager

Handles loading and updating ComfyUI workflow JSON files with parameters from AIGen.yaml.
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple
from PIL import Image


class WorkflowManager:
    """Manages ComfyUI workflow loading and parameter updates."""
    
    @staticmethod
    def load_workflow(workflow_path: str, project_root: Path) -> Dict:
        """Load workflow JSON (or ComfyUI workflow PNG) file.

        Supports:
        - JSON: the API prompt dict (node-id -> {class_type, inputs, ...})
        - PNG: a ComfyUI-exported image that contains a JSON prompt in its metadata (commonly under `prompt`)
        """
        workflow_file = Path(workflow_path)
        # project_root is typically projects/<name>/, so repo_root is two levels up.
        # This matters when workflow_path is like "defaults/workflow/..." (repo-relative).
        try:
            repo_root = project_root.resolve().parent.parent
        except Exception:
            repo_root = project_root.parent.parent
        if not workflow_file.is_absolute():
            # Try relative to project root first
            workflow_file = project_root / workflow_path
            if not workflow_file.exists():
                # Try relative to project root's workflow directory
                if not workflow_path.startswith('workflow/'):
                    workflow_file = project_root / "workflow" / Path(workflow_path).name
                if not workflow_file.exists():
                    # Try repo-root-relative path directly (common: "defaults/workflow/...")
                    defaults_path = repo_root / workflow_path
                    if defaults_path.exists():
                        workflow_file = defaults_path
                    elif not workflow_path.startswith('workflow/'):
                        # Try defaults/workflow/
                        defaults_path = repo_root / "defaults" / "workflow" / Path(workflow_path).name
                        if defaults_path.exists():
                            workflow_file = defaults_path
        
        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path} (tried: {workflow_file})")
        
        suffix = workflow_file.suffix.lower()
        if suffix == ".json":
            with open(workflow_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        if suffix == ".png":
            im = Image.open(workflow_file)
            info = getattr(im, "info", {}) or {}

            # ComfyUI typically stores the API prompt JSON under the 'prompt' key in PNG metadata.
            candidates = []
            for key in ("prompt", "workflow"):
                if key in info and info[key]:
                    candidates.append((key, info[key]))

            if not candidates:
                raise ValueError(
                    f"Workflow PNG has no embedded prompt/workflow metadata: {workflow_file}. "
                    "Export from ComfyUI with metadata enabled, or use a JSON workflow."
                )

            last_err: Exception | None = None
            for key, raw in candidates:
                try:
                    s = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    data = json.loads(s)
                    if isinstance(data, dict) and data:
                        return data
                except Exception as e:
                    last_err = e

            raise ValueError(
                f"Could not parse embedded workflow metadata from PNG: {workflow_file}. "
                f"Tried keys {[k for k, _ in candidates]}; last_error={last_err}"
            )

        raise ValueError(f"Unsupported workflow file type: {workflow_file} (expected .json or .png)")
    
    @staticmethod
    def update_workflow(
        workflow: Dict,
        aigen_config: Dict,
        input_image_path: str,
        mask_image_path: str | None = None,
        control_image_path: str | None = None,
    ) -> Dict:
        """Update workflow with AIGen.yaml parameters.

        If the workflow includes a mask node (e.g. LoadImageMask), you can pass mask_image_path
        (typically a filename already in the ComfyUI input directory).
        """
        workflow_copy = json.loads(json.dumps(workflow))

        def _get_node(nid: str) -> Optional[Dict]:
            n = workflow_copy.get(str(nid))
            return n if isinstance(n, dict) else None

        def _iter_nodes_by_type(class_type: str) -> Iterable[Tuple[str, Dict]]:
            for node_id, node_data in workflow_copy.items():
                if isinstance(node_data, dict) and node_data.get("class_type") == class_type:
                    yield str(node_id), node_data

        def _is_link(v) -> bool:
            # ComfyUI JSON links are usually like ["9", 0]
            return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)

        def _follow_link(v) -> Optional[str]:
            if not _is_link(v):
                return None
            return str(v[0])

        def _collect_textencode_nodes_from(start_node_id: str, max_depth: int = 6) -> Set[str]:
            """Walk upstream through common conditioning nodes to find CLIPTextEncode nodes."""
            out: Set[str] = set()
            seen: Set[str] = set()
            stack: list[Tuple[str, int]] = [(str(start_node_id), 0)]
            while stack:
                nid, depth = stack.pop()
                if nid in seen or depth > max_depth:
                    continue
                seen.add(nid)
                n = _get_node(nid)
                if not n:
                    continue
                ct = n.get("class_type")
                if ct == "CLIPTextEncode":
                    out.add(nid)
                    continue
                # Common conditioning nodes that point at CLIPTextEncode via inputs like "positive"/"negative"
                inputs = n.get("inputs") or {}
                if isinstance(inputs, dict):
                    for key in ("positive", "negative", "conditioning", "cond", "prompt"):
                        nxt = _follow_link(inputs.get(key))
                        if nxt:
                            stack.append((nxt, depth + 1))
            return out

        def _update_prompt_texts(positive_text: str | None, negative_text: str | None) -> None:
            """Update CLIPTextEncode nodes referenced by the sampler graph (works across different node IDs)."""
            if not positive_text and not negative_text:
                return

            # Prefer following KSampler's links, since that is the ground truth for what is used in sampling.
            pos_targets: Set[str] = set()
            neg_targets: Set[str] = set()
            for ks_id, ks in _iter_nodes_by_type("KSampler"):
                ks_inputs = (ks.get("inputs") or {}) if isinstance(ks, dict) else {}
                if not isinstance(ks_inputs, dict):
                    continue
                pos_root = _follow_link(ks_inputs.get("positive"))
                neg_root = _follow_link(ks_inputs.get("negative"))
                if pos_root:
                    pos_targets |= _collect_textencode_nodes_from(pos_root)
                if neg_root:
                    neg_targets |= _collect_textencode_nodes_from(neg_root)

            # Fallback: if we couldn't trace, update by _meta title hints.
            if not pos_targets and positive_text:
                for node_id, node_data in _iter_nodes_by_type("CLIPTextEncode"):
                    title = ((node_data.get("_meta") or {}).get("title") or "").lower()
                    if "negative" not in title:
                        pos_targets.add(node_id)
            if not neg_targets and negative_text:
                for node_id, node_data in _iter_nodes_by_type("CLIPTextEncode"):
                    title = ((node_data.get("_meta") or {}).get("title") or "").lower()
                    if "negative" in title:
                        neg_targets.add(node_id)

            if positive_text:
                for nid in sorted(pos_targets):
                    n = _get_node(nid)
                    if n and isinstance(n.get("inputs"), dict):
                        n["inputs"]["text"] = positive_text
            if negative_text:
                for nid in sorted(neg_targets):
                    n = _get_node(nid)
                    if n and isinstance(n.get("inputs"), dict):
                        n["inputs"]["text"] = negative_text
        
        # Update CheckpointLoaderSimple
        model_cfg = aigen_config.get("model") or {}
        ckpt_name = model_cfg.get("ckpt_name")
        ckpt_switched = False
        # Optional: allow a dedicated inpaint checkpoint when a mask is used.
        if mask_image_path:
            inpaint_ckpt = model_cfg.get("ckpt_name_inpaint") or model_cfg.get("inpaint_ckpt_name")
            if inpaint_ckpt:
                ckpt_name = inpaint_ckpt
                ckpt_switched = True
        for node_id, node_data in _iter_nodes_by_type("CheckpointLoaderSimple"):
            if isinstance(node_data.get("inputs"), dict) and ckpt_name:
                node_data["inputs"]["ckpt_name"] = ckpt_name
        
        # Store checkpoint info in workflow metadata for logging
        if "_workflow_metadata" not in workflow_copy:
            workflow_copy["_workflow_metadata"] = {}
        workflow_copy["_workflow_metadata"]["checkpoint_used"] = ckpt_name
        workflow_copy["_workflow_metadata"]["checkpoint_switched"] = ckpt_switched
        
        # Update KSampler
        params = aigen_config.get('parameters', {}) or {}
        for node_id, node_data in _iter_nodes_by_type("KSampler"):
            inputs = node_data.get("inputs")
            if not isinstance(inputs, dict):
                continue
            if 'denoise' in params:
                inputs['denoise'] = params['denoise']
            if 'cfg' in params:
                inputs['cfg'] = params['cfg']
            if 'steps' in params:
                inputs['steps'] = params['steps']
            if 'seed' in params and params['seed'] is not None:
                inputs['seed'] = params['seed']
            else:
                # Use random seed if not specified
                inputs['seed'] = random.randint(1, 2**31 - 1)
            if 'sampler_name' in params:
                inputs['sampler_name'] = params['sampler_name']
            if 'scheduler' in params:
                inputs['scheduler'] = params['scheduler']
        
        # Update prompts (robust to node-id differences by following the sampler graph)
        prompts = aigen_config.get('prompts', {}) or {}
        _update_prompt_texts(prompts.get("positive"), prompts.get("negative"))
        
        # Update LoadImage node(s).
        #
        # - First LoadImage is assumed to be the main input image (to avoid clobbering auxiliary image loads).
        # - If control_image_path is provided, we also update any additional LoadImage nodes that are clearly
        #   marked as a control image input (by placeholder filename or meta title).
        main_input_filename = input_image_path if (isinstance(input_image_path, str) and '/' not in input_image_path) else Path(input_image_path).name
        control_filename: str | None = None
        if control_image_path:
            control_filename = control_image_path if (isinstance(control_image_path, str) and '/' not in control_image_path) else Path(control_image_path).name

        updated_main = False
        for node_id, node_data in _iter_nodes_by_type("LoadImage"):
            inputs = node_data.get("inputs")
            if not isinstance(inputs, dict):
                continue
            title = ((node_data.get("_meta") or {}).get("title") or "").lower()
            current = str(inputs.get("image") or "")
            current_l = current.lower()

            is_control_slot = False
            if control_filename:
                if current in ("CONTROL_IMAGE.png", "POSE_IMAGE.png", "DEPTH_IMAGE.png", "CONTROL.png"):
                    is_control_slot = True
                elif "control" in title or "pose" in title or "depth" in title:
                    is_control_slot = True

            if (not updated_main) and (not is_control_slot):
                inputs["image"] = main_input_filename
                updated_main = True
                continue

            if is_control_slot and control_filename:
                inputs["image"] = control_filename
                continue

        # Update LoadImageMask node (mask image), if present and provided
        if mask_image_path:
            for node_id, node_data in _iter_nodes_by_type("LoadImageMask"):
                inputs = node_data.get("inputs")
                if not isinstance(inputs, dict):
                    continue
                if isinstance(mask_image_path, str) and '/' not in mask_image_path:
                    inputs['image'] = mask_image_path
                else:
                    inputs['image'] = Path(mask_image_path).name
                # Only update the first LoadImageMask by default
                break
        
        return workflow_copy
