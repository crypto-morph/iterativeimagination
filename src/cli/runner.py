"""Shared CLI helpers for running Iterative Imagination."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Type, Any

from project_manager import ProjectManager


def build_run_parser(
    parser: Optional[argparse.ArgumentParser] = None,
    *,
    require_project: bool = False,
) -> argparse.ArgumentParser:
    """Create or augment a parser with run-related arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Iterative Imagination - AI-powered image generation with iterative improvement"
        )

    parser.add_argument(
        '--project',
        type=str,
        help='Project name (uses projects/{NAME}/config/rules.yaml)',
        required=require_project,
    )
    parser.add_argument('--rules', type=str, help='Path to rules.yaml (alternative to --project)')
    parser.add_argument('--input', type=str, help='Path to input image (if not using project structure)')
    parser.add_argument('--dry-run', action='store_true', help="Validate configs but don't run")
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--resume-from', type=int, help='Resume from a specific iteration number (or use checkpoint if not specified)')
    parser.add_argument('--seed-from-ranking', type=str, help='Seed from a prior run human ranking (RUN_ID or "latest")')
    parser.add_argument('--seed-ranking-mode', type=str, default="rank1", help='Seeding mode: rank1|top3|top5 (default: rank1)')
    parser.add_argument('--seed-from-human', type=str, help='(deprecated) alias for --seed-from-ranking')
    parser.add_argument('--reset', action='store_true', help='Reset run state for this project')
    return parser


def run_iteration(args: argparse.Namespace, iteration_cls: Type[Any], parser: Optional[argparse.ArgumentParser] = None) -> None:
    """Execute a run using the provided arguments and iteration class."""
    project_name = _determine_project_name(args, parser)

    if args.reset:
        _perform_project_reset(project_name)

    try:
        app = iteration_cls(project_name, verbose=args.verbose)
        seed_run = args.seed_from_ranking or getattr(args, 'seed_from_human', None)
        success = app.run(
            dry_run=args.dry_run,
            resume_from=args.resume_from,
            seed_from_ranking=seed_run,
            seed_ranking_mode=args.seed_ranking_mode,
        )
        sys.exit(0 if success else 1)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _determine_project_name(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser]) -> str:
    if getattr(args, 'project', None):
        return args.project
    if getattr(args, 'rules', None):
        rules_path = Path(args.rules)
        if 'projects' in rules_path.parts:
            project_idx = rules_path.parts.index('projects')
            if project_idx + 1 < len(rules_path.parts):
                return rules_path.parts[project_idx + 1]
        print("Error: Rules path must be within projects/ directory", file=sys.stderr)
        if parser:
            parser.print_help()
        sys.exit(1)

    print("Error: Must specify --project or --rules", file=sys.stderr)
    if parser:
        parser.print_help()
    sys.exit(1)


def _perform_project_reset(project_name: str) -> None:
    from contextlib import suppress

    temp_project = ProjectManager(project_name)

    chk = temp_project.get_checkpoint_path()
    with suppress(Exception):
        if chk.exists():
            chk.unlink()
            print(f"Reset: removed {chk}")

    try:
        legacy_dir = temp_project.project_root / "working"
        legacy_files = list(legacy_dir.glob("iteration_*.*"))
        if legacy_files:
            run_id = temp_project.create_run_id()
            temp_project.ensure_run_directories(run_id)
            run_root = temp_project.get_run_root(run_id)

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

            for legacy_file in legacy_files:
                legacy_file.rename(_dest_for(legacy_file))
            print(f"Reset: archived {len(legacy_files)} legacy iteration files into {run_root}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to archive legacy iteration files: {exc}", file=sys.stderr)

    working_aigen = temp_project.project_root / "working" / "AIGen.yaml"
    config_aigen = temp_project.project_root / "config" / "AIGen.yaml"
    try:
        if config_aigen.exists():
            import shutil as _shutil
            _shutil.copy2(config_aigen, working_aigen)
            print(f"Reset: restored {working_aigen} from {config_aigen}")
        elif working_aigen.exists():
            working_aigen.unlink()
            print(f"Reset: removed {working_aigen} (will be recreated from defaults)")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to reset working AIGen.yaml: {exc}", file=sys.stderr)

    progress_path = temp_project.project_root / "input" / "progress.png"
    with suppress(Exception):
        if progress_path.exists():
            progress_path.unlink()
            print(f"Reset: removed {progress_path} (will start from original input.png)")
