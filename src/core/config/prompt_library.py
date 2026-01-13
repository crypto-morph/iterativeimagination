"""Prompt template loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_prompt_templates(
    prompts_path: Optional[Path] = None,
    defaults_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Load prompt templates with layered fallbacks.

    Precedence:
      1. Baseline hardcoded prompts (ensures keys always exist)
      2. defaults/prompts.yaml (if present)
      3. repo root prompts.yaml (if present)
      4. Provided `prompts_path` (project overrides)
    """

    repo_root = Path(repo_root) if repo_root else _default_repo_root()
    defaults_root = Path(defaults_root) if defaults_root else repo_root / "defaults"
    defaults_prompts = defaults_root / "prompts.yaml"
    canonical_prompts = repo_root / "prompts.yaml"

    baseline: Dict[str, str] = {
        "ask_question": "Answer this question about the image: {question}\n\n"
        "Question type: {type_info}{enum_info}{min_max_info}\n\n"
        "Provide your answer in JSON format matching the question type:\n{\n    \"answer\": <your_answer_here>\n}\n\n"
        "For boolean questions, use true/false.\n"
        "For number/integer questions, provide a numeric value.\n"
        "For string questions, provide a text description.\n"
        "For array questions, provide a JSON array.",
        "evaluate_acceptance_criteria": "Evaluate this GENERATED image against acceptance criteria, comparing it to the ORIGINAL image.\n\n"
        "ORIGINAL IMAGE DESCRIPTION:\n{original_description}\n\n"
        "QUESTION ANSWERS (for context):\n{questions_summary}\n\n"
        "ACCEPTANCE CRITERIA:\n{criteria_text}\n\n"
        "For each criterion, determine if it passes (true/false or within min/max range).\n"
        "Then calculate overall score: (passed_criteria / total_criteria) * 100\n\n"
        "Provide your evaluation in JSON format:\n{\n    \"overall_score\": <0-100>,\n    \"criteria_results\": {\n        \"<field1>\": <true/false or value>,\n        \"<field2>\": <true/false or value>,\n        ...\n    }\n}",
        "compare_images": "Compare these two images and analyze the differences.\n\n"
        "Focus on:\n1. Similarity score (0.0 = completely different, 1.0 = identical)\n"
        "2. What differences exist? (clothing, pose, features, background, proportions)\n"
        "3. Overall analysis\n\n"
        "Provide your comparison in JSON format:\n{\n    \"similarity_score\": <0.0-1.0>,\n    \"differences\": [\"difference1\", \"difference2\", ...],\n    \"analysis\": \"detailed analysis text\"\n}",
        "describe_image": "Describe this image in detail. Focus on:\n1. Main subject (person, object, scene)\n"
        "2. If person: appearance, pose, clothing\n3. Background/setting\n4. Overall composition and notable details\n\n"
        "Provide a detailed description.",
        "improve_prompts": "You are helping improve image generation prompts for Stable Diffusion.\n\n"
        "Current positive prompt: {current_positive}\n"
        "Current negative prompt: {current_negative}\n\n"
        "Evaluation results:\n- Overall score: {overall_score}%\n- Failed criteria: {failed_criteria}\n\n"
        "Image comparison:\n- Similarity: {similarity_score}\n- Differences: {differences}\n- Analysis: {analysis}\n\n"
        "Rules to achieve:\n{rules_text}\n\n"
        "Suggest improved positive and negative prompts that:\n"
        "1. Address the failed criteria\n2. Maintain what's working well\n3. Better achieve the rules\n4. Follow Stable Diffusion prompt best practices\n\n"
        "Provide your suggestions in JSON format:\n{\n    \"positive\": \"<improved positive prompt>\",\n    \"negative\": \"<improved negative prompt>\"\n}",
    }

    merged: Dict[str, str] = dict(baseline)

    def _merge_file(path: Path) -> None:
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            merged.update(loaded)

    _merge_file(defaults_prompts)
    _merge_file(canonical_prompts)

    if prompts_path:
        _merge_file(Path(prompts_path))

    return merged
