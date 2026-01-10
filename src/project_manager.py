#!/usr/bin/env python3
"""
Project Manager

Handles project structure, configuration loading, and ComfyUI input directory setup.
"""

import os
import shutil
import time
from pathlib import Path
import json
from typing import Dict, Optional
import yaml


class ProjectManager:
    """Manages project structure and configuration."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_root = Path("projects") / project_name
    
    def load_rules(self) -> Dict:
        """Load rules.yaml from project config."""
        rules_path = self.project_root / "config" / "rules.yaml"
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_path}")
        
        with open(rules_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_input_image_path(self) -> Path:
        """Get path to the current input image.

        Preferred:
        - `input/progress.png` if present (allows multi-pass workflows without overwriting the original input)
        Fallback:
        - `input/input.png`
        
        Returns an absolute resolved path.
        """
        progress_path = (self.project_root / "input" / "progress.png").resolve()
        if progress_path.exists():
            return progress_path
        input_path = (self.project_root / "input" / "input.png").resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {input_path}\n"
                f"Expected one of:\n"
                f"  - {self.project_root / 'input' / 'progress.png'}\n"
                f"  - {self.project_root / 'input' / 'input.png'}"
            )
        return input_path
    
    def ensure_directories(self):
        """Ensure all required project directories exist."""
        (self.project_root / "working").mkdir(parents=True, exist_ok=True)
        (self.project_root / "logs").mkdir(parents=True, exist_ok=True)
        (self.project_root / "output").mkdir(parents=True, exist_ok=True)

    def create_run_id(self) -> str:
        """Create a filesystem-safe run id for grouping iteration artefacts."""
        # Avoid ':' for portability, even though Linux allows it.
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def get_run_root(self, run_id: str) -> Path:
        """Get the root directory for a run under working/."""
        return self.project_root / "working" / run_id

    def ensure_run_directories(self, run_id: str):
        """Ensure run subdirectories exist under working/{run_id}/."""
        run_root = self.get_run_root(run_id)
        (run_root / "images").mkdir(parents=True, exist_ok=True)
        (run_root / "questions").mkdir(parents=True, exist_ok=True)
        (run_root / "evaluation").mkdir(parents=True, exist_ok=True)
        (run_root / "comparison").mkdir(parents=True, exist_ok=True)
        (run_root / "metadata").mkdir(parents=True, exist_ok=True)
        (run_root / "human").mkdir(parents=True, exist_ok=True)

    def list_run_ids(self) -> list[str]:
        """List run IDs under working/, newest first."""
        working_dir = self.project_root / "working"
        if not working_dir.exists():
            return []
        runs = []
        for p in working_dir.iterdir():
            if not p.is_dir():
                continue
            # run ids are timestamp-like and start with a digit (e.g. 2026-01-09_22-25-59)
            if not p.name or not p.name[0].isdigit():
                continue
            runs.append(p.name)
        runs.sort(reverse=True)
        return runs

    def latest_run_id(self) -> Optional[str]:
        runs = self.list_run_ids()
        return runs[0] if runs else None

    def latest_run_id_with_human_feedback(self) -> Optional[str]:
        """Return the most recent run id that has human feedback (human/ranking.json)."""
        for run_id in self.list_run_ids():
            p = self.get_run_root(run_id) / "human" / "ranking.json"
            if p.exists():
                return run_id
        return None

    def load_human_ranking(self, run_id: str) -> Optional[Dict]:
        """Load working/<run_id>/human/ranking.json if present."""
        p = self.get_run_root(run_id) / "human" / "ranking.json"
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def load_iteration_metadata(self, run_id: str, iteration_num: int) -> Optional[Dict]:
        """Load metadata for an iteration in a run."""
        paths = self.get_iteration_paths(iteration_num, run_id=run_id)
        p = paths.get("metadata")
        if not p or not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    
    def load_aigen_config(self) -> Dict:
        """Load AIGen.yaml, preferring working directory, then config, then defaults.
        Merges aivis config from project config into working file if both exist."""
        working_aigen = self.project_root / "working" / "AIGen.yaml"
        project_aigen = self.project_root / "config" / "AIGen.yaml"
        defaults_aigen = Path("defaults") / "config" / "AIGen.yaml"
        
        # Load working config if it exists
        if working_aigen.exists():
            with open(working_aigen, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Merge aivis config from project config if it exists (project config takes precedence)
        if project_aigen.exists():
            with open(project_aigen, 'r', encoding='utf-8') as f:
                project_config = yaml.safe_load(f) or {}
            if 'aivis' in project_config:
                config['aivis'] = project_config['aivis']
        
        # If no working file and no project config, use defaults
        if not working_aigen.exists() and not project_aigen.exists():
            if defaults_aigen.exists():
                with open(defaults_aigen, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                raise FileNotFoundError("No AIGen.yaml found in project, config, or defaults")
        
        # Save merged config to working directory
        with open(working_aigen, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config
    
    def save_aigen_config(self, config: Dict):
        """Save AIGen.yaml to working directory."""
        working_aigen = self.project_root / "working" / "AIGen.yaml"
        with open(working_aigen, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def load_aivis_config(self) -> Dict:
        """Load AIVis.yaml, preferring project config, then defaults."""
        project_aivis = self.project_root / "config" / "AIVis.yaml"
        defaults_aivis = Path("defaults") / "config" / "AIVis.yaml"
        
        # Try project config first
        if project_aivis.exists():
            with open(project_aivis, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        
        # Fall back to defaults
        if defaults_aivis.exists():
            with open(defaults_aivis, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        
        # If neither exists, return minimal defaults
        return {
            "provider": "ollama",
            "model": "qwen3-vl:4b",
            "fallback_provider": "ollama",
            "fallback_model": "llava-phi3:latest"
        }
    
    def find_comfyui_input_dir(self) -> Path:
        """Find ComfyUI input directory."""
        # Check environment variable
        env_path = os.environ.get("COMFYUI_DIR")
        if env_path:
            comfyui_dir = Path(env_path).expanduser().resolve()
            if (comfyui_dir / "main.py").exists():
                return comfyui_dir / "input"
        
        # Check if we're inside ComfyUI (legacy location)
        parent = Path(__file__).parent.parent
        if (parent / "main.py").exists():
            return parent / "input"
        
        # Check common locations
        common_paths = [
            Path.home() / "ComfyUI" / "input",
            Path("/opt/ComfyUI/input"),
            Path("/usr/local/ComfyUI/input"),
        ]
        
        for path in common_paths:
            if path.exists() and path.is_dir():
                return path
        
        # Default fallback
        default_path = Path.home() / "ComfyUI" / "input"
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path
    
    def prepare_input_image(self, input_image_path: Path, comfyui_input_dir: Path) -> str:
        """Copy input image to ComfyUI input directory and return filename."""
        # Resolve path to absolute to avoid issues with relative paths
        input_image_path = input_image_path.resolve()
        
        # Verify the file exists
        if not input_image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {input_image_path}\n"
                f"Expected one of:\n"
                f"  - {self.project_root / 'input' / 'progress.png'}\n"
                f"  - {self.project_root / 'input' / 'input.png'}"
            )
        
        # Generate a unique filename to avoid conflicts.
        #
        # IMPORTANT: We may copy multiple files in the same second (e.g. input.png + mask.png),
        # so second-resolution timestamps can collide and overwrite the prior copy.
        stamp = time.time_ns()
        stem = (input_image_path.stem or "input").replace(" ", "_")
        filename = f"iterative_imagination_{self.project_name}_{stem}_{stamp}.png"
        comfyui_input_path = comfyui_input_dir.resolve() / filename
        
        # Copy image
        shutil.copy2(input_image_path, comfyui_input_path)
        
        return filename
    
    def get_iteration_paths(self, iteration_num: int, run_id: Optional[str] = None) -> Dict[str, Path]:
        """Get paths for iteration files.

        If run_id is provided, files are stored under:
          working/{run_id}/{images|questions|evaluation|comparison|metadata}/
        Otherwise uses legacy flat layout in working/.
        """
        if run_id:
            self.ensure_run_directories(run_id)
            run_root = self.get_run_root(run_id)
            return {
                "image": run_root / "images" / f"iteration_{iteration_num}.png",
                "questions": run_root / "questions" / f"iteration_{iteration_num}_questions.json",
                "evaluation": run_root / "evaluation" / f"iteration_{iteration_num}_evaluation.json",
                "comparison": run_root / "comparison" / f"iteration_{iteration_num}_comparison.json",
                "metadata": run_root / "metadata" / f"iteration_{iteration_num}_metadata.json",
            }

        working_dir = self.project_root / "working"
        return {
            "image": working_dir / f"iteration_{iteration_num}.png",
            "questions": working_dir / f"iteration_{iteration_num}_questions.json",
            "evaluation": working_dir / f"iteration_{iteration_num}_evaluation.json",
            "comparison": working_dir / f"iteration_{iteration_num}_comparison.json",
            "metadata": working_dir / f"iteration_{iteration_num}_metadata.json",
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get paths for output files."""
        output_dir = self.project_root / "output"
        return {
            "image": output_dir / "output.png",
            "metadata": output_dir / "output_metadata.json",
        }
    
    def get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file for resume functionality."""
        return self.project_root / "working" / "checkpoint.json"
    
    def save_checkpoint(self, iteration: int, best_iteration: Optional[int], best_score: float, run_id: Optional[str] = None):
        """Save checkpoint for resume functionality."""
        checkpoint = {
            "last_iteration": iteration,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "timestamp": time.time(),
            "run_id": run_id
        }
        checkpoint_path = self.get_checkpoint_path()
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists."""
        checkpoint_path = self.get_checkpoint_path()
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
