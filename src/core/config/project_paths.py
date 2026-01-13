"""Path helpers for Iterative Imagination projects."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class ProjectPaths:
    """Utility for computing and ensuring project-related paths."""

    def __init__(self, project_name: str, projects_root: Path | str | None = None):
        self.project_name = project_name
        self.projects_root = Path(projects_root or "projects")
        self.project_root = self.projects_root / project_name

    # ------------------------------------------------------------------
    # Core directories
    # ------------------------------------------------------------------
    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def working_dir(self) -> Path:
        return self.project_root / "working"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    def ensure_project_directories(self) -> None:
        """Ensure the basic directories exist."""
        for path in (self.working_dir, self.logs_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config file helpers
    # ------------------------------------------------------------------
    def config_file(self, relative_path: str) -> Path:
        return self.config_dir / relative_path

    # ------------------------------------------------------------------
    # Run directories
    # ------------------------------------------------------------------
    def run_root(self, run_id: str) -> Path:
        return self.working_dir / run_id

    def ensure_run_directories(self, run_id: str) -> None:
        run_root = self.run_root(run_id)
        subdirs = ("images", "questions", "evaluation", "comparison", "metadata", "human")
        for name in subdirs:
            (run_root / name).mkdir(parents=True, exist_ok=True)

    def human_feedback_file(self, run_id: str) -> Path:
        return self.run_root(run_id) / "human" / "ranking.json"

    # ------------------------------------------------------------------
    # Artefact paths
    # ------------------------------------------------------------------
    def iteration_paths(self, iteration_num: int, run_id: Optional[str] = None) -> Dict[str, Path]:
        if run_id:
            self.ensure_run_directories(run_id)
            run_root = self.run_root(run_id)
            return {
                "image": run_root / "images" / f"iteration_{iteration_num}.png",
                "questions": run_root / "questions" / f"iteration_{iteration_num}_questions.json",
                "evaluation": run_root / "evaluation" / f"iteration_{iteration_num}_evaluation.json",
                "comparison": run_root / "comparison" / f"iteration_{iteration_num}_comparison.json",
                "metadata": run_root / "metadata" / f"iteration_{iteration_num}_metadata.json",
            }

        base = self.working_dir
        return {
            "image": base / f"iteration_{iteration_num}.png",
            "questions": base / f"iteration_{iteration_num}_questions.json",
            "evaluation": base / f"iteration_{iteration_num}_evaluation.json",
            "comparison": base / f"iteration_{iteration_num}_comparison.json",
            "metadata": base / f"iteration_{iteration_num}_metadata.json",
        }

    def output_paths(self) -> Dict[str, Path]:
        return {
            "image": self.output_dir / "output.png",
            "metadata": self.output_dir / "output_metadata.json",
        }

    @property
    def checkpoint_path(self) -> Path:
        return self.working_dir / "checkpoint.json"
