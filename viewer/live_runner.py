#!/usr/bin/env python3
"""
Live runner controller for the viewer web app.

Runs one iteration at a time and pauses between iterations waiting for human feedback.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LiveRunState:
    project: str
    run_id: str
    status: str = "starting"  # starting|running|waiting|finished|error
    current_iteration: int = 0
    last_score: Optional[float] = None
    best_iteration: Optional[int] = None
    best_score: float = 0.0
    message: str = ""
    updated_at: float = field(default_factory=lambda: time.time())
    waiting_for_feedback: bool = False
    expected_feedback_for_iteration: Optional[int] = None


class LiveRunController:
    def __init__(self, project: str, run_id: str, max_iterations: int = 20):
        self.project = project
        self.run_id = run_id
        self.max_iterations = max_iterations
        self.repo_root = Path(__file__).parent.parent.resolve()

        self._lock = threading.Lock()
        self._state = LiveRunState(project=project, run_id=run_id)
        self._thread: Optional[threading.Thread] = None
        self._feedback_event = threading.Event()
        self._last_feedback: Dict[str, Any] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state.__dict__)

    def submit_feedback(self, iteration: int, comment: str, nudge: Dict[str, bool]) -> None:
        with self._lock:
            self._last_feedback = {
                "iteration": int(iteration),
                "comment": (comment or "").strip(),
                "nudge": {
                    "less_random": bool(nudge.get("less_random")),
                    "too_similar": bool(nudge.get("too_similar")),
                },
                "timestamp": time.time(),
            }
            self._state.waiting_for_feedback = False
            self._state.message = "Feedback received"
            self._state.updated_at = time.time()
        self._feedback_event.set()

    def _human_dir(self) -> Path:
        return self.repo_root / "projects" / self.project / "working" / self.run_id / "human"

    def _save_feedback_file(self, feedback: Dict[str, Any]) -> None:
        human_dir = self._human_dir()
        human_dir.mkdir(parents=True, exist_ok=True)
        it = int(feedback.get("iteration", 0))
        path = human_dir / f"feedback_iteration_{it}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2)

        # Also maintain an aggregate file for convenience.
        agg_path = human_dir / "feedback.json"
        agg = {"entries": []}
        if agg_path.exists():
            try:
                with open(agg_path, "r", encoding="utf-8") as f:
                    agg = json.load(f) or {"entries": []}
            except Exception:
                agg = {"entries": []}
        agg_entries = agg.get("entries") if isinstance(agg, dict) else []
        if not isinstance(agg_entries, list):
            agg_entries = []
        agg_entries.append(feedback)
        agg = {"entries": agg_entries, "updated_at": time.time()}
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)

    def _apply_nudges_to_working_aigen(self, nudge: Dict[str, bool]) -> None:
        """Directly adjust working/AIGen.yaml params as a gentle manual hint."""
        try:
            import sys
            from pathlib import Path as _Path

            sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
            from project_manager import ProjectManager  # noqa: E402

            pm = ProjectManager(self.project)
            aigen = pm.load_aigen_config()
            params = aigen.get("parameters", {}) or {}
            denoise = float(params.get("denoise", 0.5))
            cfg = float(params.get("cfg", 7.0))

            if nudge.get("less_random"):
                denoise = max(0.10, denoise - 0.05)
                cfg = max(1.0, cfg - 0.5)
            if nudge.get("too_similar"):
                denoise = min(0.95, denoise + 0.05)
                cfg = min(20.0, cfg + 1.0)

            params["denoise"] = denoise
            params["cfg"] = cfg
            aigen["parameters"] = params
            pm.save_aigen_config(aigen)
        except Exception:
            # Best-effort: do not fail the run if nudges cannot be applied.
            return

    def _run_loop(self):
        try:
            import sys
            from pathlib import Path as _Path

            # Import iterative_imagination from repo root
            repo_root = _Path(__file__).parent.parent.resolve()
            os.chdir(str(repo_root))
            sys.path.insert(0, str(repo_root))
            from iterative_imagination import IterativeImagination  # noqa: E402

            app = IterativeImagination(self.project, verbose=False)

            # Ensure run_id is fixed and directories exist
            app.run_id = self.run_id
            app.project.ensure_run_directories(self.run_id)

            with self._lock:
                self._state.status = "running"
                self._state.message = "Running"
                self._state.updated_at = time.time()

            for iteration in range(1, int(self.max_iterations) + 1):
                # Reload rules before each iteration so edits from the web UI apply immediately.
                try:
                    app.rules = app.project.load_rules()
                except Exception:
                    pass

                with self._lock:
                    self._state.current_iteration = iteration
                    self._state.status = "running"
                    self._state.updated_at = time.time()

                md = app._run_iteration(iteration)
                if not md:
                    raise RuntimeError("Iteration failed")

                score = (md.get("evaluation") or {}).get("overall_score", 0)
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    score_val = 0.0

                with self._lock:
                    self._state.last_score = score_val
                    if score_val > self._state.best_score:
                        self._state.best_score = score_val
                        self._state.best_iteration = iteration
                    self._state.updated_at = time.time()

                if score_val >= 100:
                    with self._lock:
                        self._state.status = "finished"
                        self._state.message = f"Perfect score at iteration {iteration}"
                        self._state.updated_at = time.time()
                    break

                if iteration >= int(self.max_iterations):
                    with self._lock:
                        self._state.status = "finished"
                        self._state.message = "Max iterations reached"
                        self._state.updated_at = time.time()
                    break

                # Wait for feedback for this iteration before applying improvements
                with self._lock:
                    self._state.status = "waiting"
                    self._state.waiting_for_feedback = True
                    self._state.expected_feedback_for_iteration = iteration
                    self._state.message = "Waiting for feedback"
                    self._state.updated_at = time.time()

                self._feedback_event.clear()
                self._feedback_event.wait()  # waits indefinitely until UI submits feedback

                feedback = {}
                with self._lock:
                    feedback = dict(self._last_feedback)

                if feedback:
                    self._save_feedback_file(feedback)
                    nudge = feedback.get("nudge") or {}
                    if isinstance(nudge, dict):
                        self._apply_nudges_to_working_aigen(nudge)

                    # Add comment as context to improve_prompts step
                    comment = feedback.get("comment") or ""
                    if isinstance(comment, str) and comment.strip():
                        app.human_feedback_context = (
                            "\n\nHUMAN FEEDBACK (live iteration loop):\n"
                            f"- iteration {iteration}: {comment.strip()}\n"
                        )
                    else:
                        app.human_feedback_context = ""

                # Apply improvements (updates working/AIGen.yaml)
                app._apply_improvements(md)

        except Exception as e:
            with self._lock:
                self._state.status = "error"
                self._state.message = str(e)
                self._state.updated_at = time.time()

