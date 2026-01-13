"""Project state helpers (runs, checkpoints, metadata)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.config.project_paths import ProjectPaths


class ProjectState:
    """Handles run history and checkpoint persistence for a project."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    # ------------------------------------------------------------------
    # Run listings
    # ------------------------------------------------------------------
    def list_run_ids(self) -> List[str]:
        working_dir = self.paths.working_dir
        if not working_dir.exists():
            return []
        runs: List[str] = []
        for entry in working_dir.iterdir():
            if entry.is_dir() and entry.name and entry.name[0].isdigit():
                runs.append(entry.name)
        runs.sort(reverse=True)
        return runs

    def latest_run_id(self) -> Optional[str]:
        runs = self.list_run_ids()
        return runs[0] if runs else None

    def latest_run_id_with_human_feedback(self) -> Optional[str]:
        for run_id in self.list_run_ids():
            if self.paths.human_feedback_file(run_id).exists():
                return run_id
        return None

    # ------------------------------------------------------------------
    # Human feedback & metadata
    # ------------------------------------------------------------------
    def load_human_ranking(self, run_id: str) -> Optional[Dict]:
        file_path = self.paths.human_feedback_file(run_id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def load_iteration_metadata(self, run_id: str, iteration_num: int) -> Optional[Dict]:
        metadata_path = self.paths.iteration_paths(iteration_num, run_id=run_id).get("metadata")
        if not metadata_path or not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        return self.paths.checkpoint_path

    def save_checkpoint(self, iteration: int, best_iteration: Optional[int], best_score: float, run_id: Optional[str] = None) -> None:
        checkpoint = {
            "last_iteration": iteration,
            "best_iteration": best_iteration,
            "best_score": best_score,
            "timestamp": time.time(),
            "run_id": run_id,
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)

    def load_checkpoint(self) -> Optional[Dict]:
        if not self.checkpoint_path.exists():
            return None
        with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
