"""Legacy ProjectManager wrapper around new core utilities."""

import os
import shutil
import time
from pathlib import Path
import json
from typing import Dict, Optional

from core.config.project_paths import ProjectPaths
from core.config.config_store import ConfigStore
from core.services.project_state import ProjectState


class ProjectManager:
    """Manages project structure, configuration, and ComfyUI prep (legacy facade)."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.paths = ProjectPaths(project_name)
        self.config = ConfigStore(self.paths)
        self.paths.ensure_project_directories()
        self.state = ProjectState(self.paths)
    
    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return self.paths.project_root
    
    def load_rules(self) -> Dict:
        return self.config.load_rules()
    
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
        self.paths.ensure_project_directories()

    def create_run_id(self) -> str:
        """Create a filesystem-safe run id for grouping iteration artefacts."""
        # Avoid ':' for portability, even though Linux allows it.
        return time.strftime("%Y-%m-%d_%H-%M-%S")

    def get_run_root(self, run_id: str) -> Path:
        """Get the root directory for a run under working/."""
        return self.paths.run_root(run_id)

    def ensure_run_directories(self, run_id: str):
        """Ensure run subdirectories exist under working/{run_id}/."""
        self.paths.ensure_run_directories(run_id)

    def list_run_ids(self) -> list[str]:
        """List run IDs under working/, newest first."""
        return self.state.list_run_ids()

    def latest_run_id(self) -> Optional[str]:
        return self.state.latest_run_id()

    def latest_run_id_with_human_feedback(self) -> Optional[str]:
        """Return the most recent run id that has human feedback (human/ranking.json)."""
        return self.state.latest_run_id_with_human_feedback()

    def load_human_ranking(self, run_id: str) -> Optional[Dict]:
        """Load working/<run_id>/human/ranking.json if present."""
        return self.state.load_human_ranking(run_id)

    def load_iteration_metadata(self, run_id: str, iteration_num: int) -> Optional[Dict]:
        """Load metadata for an iteration in a run."""
        return self.state.load_iteration_metadata(run_id=run_id, iteration_num=iteration_num)
    
    def load_aigen_config(self) -> Dict:
        return self.config.load_aigen_config()
    
    def save_aigen_config(self, config: Dict):
        self.config.save_aigen_config(config)
    
    def load_aivis_config(self) -> Dict:
        return self.config.load_aivis_config()
    
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
        return self.paths.iteration_paths(iteration_num, run_id=run_id)
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get paths for output files."""
        return self.paths.output_paths()
    
    def get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file for resume functionality."""
        return self.state.checkpoint_path
    
    def save_checkpoint(self, iteration: int, best_iteration: Optional[int], best_score: float, run_id: Optional[str] = None):
        """Save checkpoint for resume functionality."""
        self.state.save_checkpoint(iteration, best_iteration, best_score, run_id)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists."""
        return self.state.load_checkpoint()
