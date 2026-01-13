"""Evaluation and question-answering helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


class EvaluationService:
    """Handles AIVis-backed question answering and acceptance evaluation."""

    def __init__(
        self,
        logger,
        rules: Dict,
        aivis_client,
        describe_original_image_fn: Callable[[], str],
    ) -> None:
        self.logger = logger
        self.rules = rules
        self.aivis = aivis_client
        self._describe_original_image = describe_original_image_fn

    # ------------------------------------------------------------------ #
    # Question answering
    # ------------------------------------------------------------------ #
    def answer_questions(self, image_path: str) -> Dict:
        """Answer all questions from rules.yaml about the generated image."""
        self.logger.info("Answering questions about generated image (batch request)...")
        questions = self.rules.get("questions", [])

        if not questions:
            return {"_metadata": {}}

        original_desc = self._describe_original_image()

        try:
            answers, metadata = self.aivis.ask_multiple_questions(
                image_path,
                questions,
                original_description=original_desc,
            )

            for field, answer in answers.items():
                self.logger.debug("  %s: %s", field, answer)

            answers["_metadata"] = {"batch_request": metadata}
            return answers
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("  Failed to answer questions in batch: %s", exc)
            answers = {}
            for question_def in questions:
                field = question_def["field"]
                qtype = question_def.get("type", "string")
                if qtype == "boolean":
                    answers[field] = False
                elif qtype in ("number", "integer"):
                    answers[field] = question_def.get("min", 0)
                elif qtype == "array":
                    answers[field] = []
                else:
                    answers[field] = ""

            answers["_metadata"] = {
                "batch_request": {
                    "provider": self.aivis.provider,
                    "model": self.aivis.model,
                    "using_fallback": getattr(self.aivis, "_using_fallback", False),
                    "success": False,
                    "error": str(exc),
                }
            }
            return answers

    # ------------------------------------------------------------------ #
    # Acceptance criteria evaluation
    # ------------------------------------------------------------------ #
    def evaluate_acceptance_criteria(
        self,
        image_path: str,
        question_answers: Dict,
        criteria: Optional[List[Dict]] = None,
    ) -> Dict:
        """Evaluate image against acceptance criteria."""
        self.logger.info("Evaluating acceptance criteria...")
        criteria = criteria if criteria is not None else self.rules.get("acceptance_criteria", [])
        original_desc = self._describe_original_image()

        evaluation = self.aivis.evaluate_acceptance_criteria(
            image_path,
            original_desc,
            criteria,
            question_answers,
        )

        criteria_results = evaluation.get("criteria_results", {})
        for criterion in criteria:
            field = criterion["field"]
            criteria_results.setdefault(field, False)
        evaluation["criteria_results"] = criteria_results

        passed, failed = self._partition_passed(criteria, criteria_results)
        total = len(passed) + len(failed)
        overall = int(round((len(passed) / total) * 100)) if total else 0

        evaluation["overall_score"] = overall
        evaluation["passed_fields"] = passed
        evaluation["failed_fields"] = failed
        evaluation["criteria_total"] = total
        return evaluation

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _partition_passed(
        self, criteria: List[Dict], criteria_results: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        def _is_pass(defn: Dict, value: Any) -> bool:
            ctype = str(defn.get("type") or "boolean").lower()
            if ctype == "boolean":
                return bool(value) is True
            if ctype in {"number", "integer", "float"}:
                try:
                    val = float(value)
                except Exception:
                    return False
                mn = defn.get("min")
                mx = defn.get("max")
                if mn is not None and val < float(mn):
                    return False
                if mx is not None and val > float(mx):
                    return False
                return True
            if ctype == "string":
                return isinstance(value, str) and value.strip() != ""
            if ctype == "array":
                if not isinstance(value, list):
                    return False
                mn = defn.get("min")
                mx = defn.get("max")
                if mn is not None and len(value) < int(mn):
                    return False
                if mx is not None and len(value) > int(mx):
                    return False
                return True
            return False

        passed: List[str] = []
        failed: List[str] = []
        for cdef in criteria:
            if not isinstance(cdef, dict):
                continue
            field = cdef.get("field")
            if not field:
                continue
            if _is_pass(cdef, criteria_results.get(field)):
                passed.append(field)
            else:
                failed.append(field)
        return passed, failed
