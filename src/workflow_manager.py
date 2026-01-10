#!/usr/bin/env python3
"""
Workflow Manager

Handles loading and updating ComfyUI workflow JSON files with parameters from AIGen.yaml.
"""

import json
import random
from pathlib import Path
from typing import Dict


class WorkflowManager:
    """Manages ComfyUI workflow loading and parameter updates."""
    
    @staticmethod
    def load_workflow(workflow_path: str, project_root: Path) -> Dict:
        """Load workflow JSON file (supports relative and absolute paths)."""
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
        
        with open(workflow_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def update_workflow(workflow: Dict, aigen_config: Dict, input_image_path: str) -> Dict:
        """Update workflow with AIGen.yaml parameters."""
        workflow_copy = json.loads(json.dumps(workflow))
        
        # Update CheckpointLoaderSimple
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and node_data.get('class_type') == 'CheckpointLoaderSimple':
                workflow_copy[node_id]['inputs']['ckpt_name'] = aigen_config['model']['ckpt_name']
                break
        
        # Update KSampler
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and node_data.get('class_type') == 'KSampler':
                inputs = workflow_copy[node_id]['inputs']
                params = aigen_config['parameters']
                
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
                break
        
        # Update CLIPTextEncode nodes (node "9" = positive, node "10" = negative)
        prompts = aigen_config.get('prompts', {})
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and node_data.get('class_type') == 'CLIPTextEncode':
                if node_id == "9" and prompts.get('positive'):
                    workflow_copy[node_id]['inputs']['text'] = prompts['positive']
                elif node_id == "10" and prompts.get('negative'):
                    workflow_copy[node_id]['inputs']['text'] = prompts['negative']
        
        # Update LoadImage node
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and node_data.get('class_type') == 'LoadImage':
                # Use the filename (should already be in ComfyUI input directory)
                if isinstance(input_image_path, str) and '/' not in input_image_path:
                    # Already a filename
                    workflow_copy[node_id]['inputs']['image'] = input_image_path
                else:
                    # Extract filename
                    input_filename = Path(input_image_path).name
                    workflow_copy[node_id]['inputs']['image'] = input_filename
                break
        
        return workflow_copy
