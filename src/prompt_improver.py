#!/usr/bin/env python3
"""
Prompt Improvement Module

Handles AI-driven prompt improvement based on evaluation results.
Can be disabled per-project via rules.yaml: project.improve_prompts: false
"""

import re
from typing import Dict, List, Optional, Tuple


class PromptImprover:
    """Handles prompt improvement based on evaluation results."""
    
    def __init__(self, logger, aivis, describe_original_image_fn, human_feedback_context: str = ""):
        self.logger = logger
        self.aivis = aivis
        self._describe_original_image = describe_original_image_fn
        self.human_feedback_context = human_feedback_context
    
    def improve_prompts(
        self,
        current_positive: str,
        current_negative: str,
        evaluation: Dict,
        comparison: Dict,
        failed_criteria: List[str],
        criteria_defs: List[Dict],
        criteria_by_field: Dict[str, Dict],
        original_description: Optional[str] = None,
    ) -> Tuple[str, str, Dict]:
        """Improve prompts based on failed criteria.
        
        Returns (improved_positive, improved_negative, diff_info)
        where diff_info contains {"must_include_terms", "ban_terms", "avoid_terms", "pos_diff", "neg_diff"}
        """
        # Include criteria tags in the text shown to AIVis, so it can act on them.
        def _listify(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i).strip() for i in x if str(i).strip()]
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            return [str(x).strip()]
        
        def _crit_line(c: Dict) -> str:
            q = c.get('question', '')
            intent = (c.get('intent') or 'preserve').strip().lower()
            strength = (c.get('edit_strength') or 'medium').strip().lower()
            must = ", ".join(_listify(c.get('must_include')))
            ban = ", ".join(_listify(c.get('ban_terms')))
            avoid = ", ".join(_listify(c.get('avoid_terms')))
            bits = [f"intent={intent}", f"strength={strength}"]
            if must:
                bits.append(f"must_include=[{must}]")
            if ban:
                bits.append(f"ban_terms=[{ban}]")
            if avoid:
                bits.append(f"avoid_terms=[{avoid}]")
            return f"- {c.get('field')}: {q} ({'; '.join(bits)})"

        rules_text = "\n".join([_crit_line(c) for c in criteria_defs])
        if self.human_feedback_context:
            rules_text = rules_text + self.human_feedback_context
        
        # Ground the prompt improver to avoid hallucinating an unrelated scene/background.
        # We include the original description and explicit constraints, but still post-process heavily.
        if not original_description:
            try:
                original_description = self._describe_original_image()
            except Exception:
                original_description = ""
        
        if original_description:
            rules_text = (
                "ORIGINAL IMAGE DESCRIPTION (authoritative grounding):\n"
                + original_description.strip()
                + "\n\nIMPORTANT: Do NOT invent new scene elements or backgrounds not present in the original description.\n"
                + "Keep the scene/background consistent unless a change criterion explicitly requires it.\n\n"
                + rules_text
            )
        
        improved_positive, improved_negative = self.aivis.improve_prompts(
            current_positive, current_negative, evaluation, comparison,
            failed_criteria, rules_text
        )
        
        # Post-process prompts using tags from failed criteria.
        must_include_terms: List[str] = []
        ban_terms: List[str] = []
        avoid_terms: List[str] = []
        for field in failed_criteria:
            crit = criteria_by_field.get(field, {})
            must_include_terms.extend(_listify(crit.get('must_include')))
            ban_terms.extend(_listify(crit.get('ban_terms')))
            avoid_terms.extend(_listify(crit.get('avoid_terms')))

        # De-duplicate while preserving order
        def _dedupe(seq: List[str]) -> List[str]:
            seen = set()
            out = []
            for s in seq:
                key = s.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(s.strip())
            return out

        must_include_terms = _dedupe(must_include_terms)
        ban_terms = _dedupe(ban_terms)
        avoid_terms = _dedupe(avoid_terms)

        def _prune_weird_tokens(s: str, max_chars: int = 80, max_words: int = 6) -> str:
            """Keep prompts as short tag lists; drop sentence-like fragments.

            This is intentionally strict: we want comma-separated tags, not prose.
            """
            parts = [p.strip() for p in (s or "").split(",")]
            out = []
            for p in parts:
                if not p:
                    continue
                pl = p.lower().strip()
                # Drop sentence-y fragments and long clauses
                if any(ch in p for ch in (".", "!", "?", ";", ":")):
                    continue
                # Drop fragments that look like prose / descriptions
                if pl.startswith(("a ", "an ", "the ")):
                    continue
                if any(w in pl for w in (" standing", " wearing", " against ", " as in ", " maintaining", " changing", " in the ", " in a ")):
                    continue
                if len(p) > max_chars:
                    continue
                if len(p.split()) > max_words:
                    continue
                out.append(p)
            return ", ".join(out).strip(" ,\n")

        def _canonicalise_csv_tags(s: str, max_words: int = 4, max_chars: int = 60) -> str:
            """Final normalisation pass to enforce tag-like comma-separated tokens."""
            parts = [p.strip() for p in (s or "").split(",")]
            out = []
            seen = set()
            for p in parts:
                if not p:
                    continue
                pl = p.lower().strip()
                if any(ch in p for ch in (".", "!", "?", ";", ":")):
                    continue
                if pl.startswith(("a ", "an ", "the ")):
                    continue
                if any(w in pl for w in (" standing", " wearing", " against ", " as in ", " maintaining", " changing", " in the ", " in a ")):
                    continue
                if len(p) > max_chars:
                    continue
                if len(p.split()) > max_words:
                    continue
                if pl in seen:
                    continue
                seen.add(pl)
                out.append(p)
            return ", ".join(out).strip(" ,\n")

        def _filter_csv_by_terms(s: str, forbidden_terms: List[str]) -> str:
            """Remove any comma-separated token that contains any forbidden term (substring match, catches plurals).
            
            Also checks individual words from multi-word forbidden terms (e.g., "summer dress" from "this lady wearing a summer dress").
            """
            if not s or not forbidden_terms:
                return (s or "").strip(" ,\n")
            forb = [t.strip().lower() for t in forbidden_terms if (t or "").strip()]
            # Extract individual words from multi-word forbidden terms for partial matching
            # e.g., "this lady wearing a summer dress" -> also check for "summer" and "dress"
            forb_words = set()
            for t in forb:
                words = t.split()
                for w in words:
                    if len(w) > 3:  # Only meaningful words (skip "the", "a", etc.)
                        forb_words.add(w)
            
            parts = [p.strip() for p in (s or "").split(",")]
            out = []
            for p in parts:
                if not p:
                    continue
                pl = p.lower()
                # Check if prompt part contains any full forbidden term
                if any(t in pl for t in forb):
                    continue
                # Also check if prompt part contains any word from a forbidden term
                if any(w in pl for w in forb_words):
                    continue
                out.append(p)
            return ", ".join(out).strip(" ,\n")

        # First prune obvious garbage (long, sentence-like fragments) before tag logic.
        # BUT: protect must_include terms from being pruned - they're critical.
        # Extract must_include terms first, prune, then add them back.
        protected_terms = []
        protected_words = set()  # Individual words from protected terms (for partial matching)
        for t in must_include_terms:
            t_str = str(t).strip()
            if t_str:
                protected_terms.append(t_str)
                # Also extract individual words for partial matching
                # e.g., "lady wearing snowsuit" -> protect "snowsuit" if it appears alone
                words = t_str.lower().split()
                for w in words:
                    if len(w) > 3:  # Only protect meaningful words (skip "the", "a", etc.)
                        protected_words.add(w)
        
        # Temporarily remove protected terms, prune, then add them back
        # IMPORTANT: Use substring matching for multi-word terms like "lady wearing snowsuit"
        # Also protect individual words that appear in protected terms (e.g., "snowsuit" from "lady wearing snowsuit")
        temp_positive = improved_positive
        for pt in protected_terms:
            # Remove protected term temporarily (case-insensitive, substring match for phrases)
            pt_escaped = re.escape(pt)
            # For multi-word terms, use substring match; for single words, use word boundary
            if " " in pt:
                temp_positive = re.sub(pt_escaped, "", temp_positive, flags=re.IGNORECASE)
            else:
                temp_positive = re.sub(rf"\b{pt_escaped}\b", "", temp_positive, flags=re.IGNORECASE)
        
        # Also protect individual words from protected terms (e.g., "snowsuit" from "lady wearing snowsuit")
        # Split prompt into parts and protect any that contain protected words
        pos_parts_before_prune = [p.strip() for p in temp_positive.split(",") if p.strip()]
        protected_parts = []
        remaining_parts = []
        for part in pos_parts_before_prune:
            part_lower = part.lower()
            is_protected = False
            # Check if this part contains any protected word
            for pw in protected_words:
                if pw in part_lower:
                    protected_parts.append(part)
                    is_protected = True
                    break
            if not is_protected:
                remaining_parts.append(part)
        
        # Prune only the non-protected parts
        temp_positive = ", ".join(remaining_parts)
        improved_positive = _prune_weird_tokens(temp_positive)
        # Re-add protected parts (they'll be properly ordered later)
        if protected_parts:
            improved_positive = ", ".join(protected_parts + [improved_positive]).strip(" ,\n")
        improved_negative = _prune_weird_tokens(improved_negative)
        
        # Protected terms will be added back in correct order below

        for t in avoid_terms:
            improved_positive = re.sub(rf"\\b{re.escape(t)}\\b", "", improved_positive, flags=re.IGNORECASE)
        improved_positive = re.sub(r"\\s{2,}", " ", improved_positive).strip(" ,\n")

        # Ensure must-include terms appear in positive (prioritize change terms at front)
        # Separate change vs preserve intent terms
        change_must_include = []
        preserve_must_include = []
        for crit in criteria_defs:
            if not isinstance(crit, dict):
                continue
            intent = str(crit.get("intent") or "preserve").strip().lower()
            terms = crit.get("must_include") or []
            if intent == "change":
                change_must_include.extend(terms)
            else:
                preserve_must_include.extend(terms)
        
        # Extract existing parts, filter out must_include terms (we'll re-add in correct order)
        pos_parts = [p.strip() for p in improved_positive.split(",") if p.strip()]
        all_must_include_lower = [str(t).strip().lower() for t in (change_must_include + preserve_must_include) if t]
        # Also extract individual words from must_include terms for partial matching
        # e.g., "lady wearing snowsuit" -> protect "snowsuit" if it appears alone
        must_include_words = set()
        for term in all_must_include_lower:
            words = term.split()
            for w in words:
                if len(w) > 3:  # Only meaningful words (skip "the", "a", etc.)
                    must_include_words.add(w)
        
        filtered_parts = []
        protected_parts = []
        for part in pos_parts:
            part_lower = part.lower()
            # Check if this part matches a full must_include term
            is_must_include = any(term_lower in part_lower or part_lower in term_lower for term_lower in all_must_include_lower)
            # Also check if this part contains a word from a must_include term (partial match)
            contains_protected_word = any(pw in part_lower for pw in must_include_words)
            if is_must_include or contains_protected_word:
                protected_parts.append(part)  # Keep this part, don't filter it
            else:
                filtered_parts.append(part)
        
        # Add change terms first (for weight), then preserve terms, then rest
        # IMPORTANT: Always add must_include terms even if they appear in pos_parts,
        # because the AI might have removed them or they might be in a different form.
        # We'll deduplicate later, but we want to ensure they're at the front for weight.
        change_terms_to_add = []
        for t in change_must_include:
            t_str = str(t).strip()
            if t_str:
                # Check if a close variant exists (substring match)
                t_lower = t_str.lower()
                found_variant = False
                # First check protected_parts (already identified as must_include related)
                for i, p in enumerate(protected_parts):
                    if t_lower in p.lower() or p.lower() in t_lower:
                        change_terms_to_add.append(p)
                        protected_parts.pop(i)
                        found_variant = True
                        break
                # If not found, check filtered_parts
                if not found_variant:
                    for i, p in enumerate(filtered_parts):
                        if t_lower in p.lower() or p.lower() in t_lower:
                            change_terms_to_add.append(p)
                            filtered_parts.pop(i)
                            found_variant = True
                            break
                # If still not found, add the full term
                if not found_variant:
                    change_terms_to_add.append(t_str)
                    self.logger.info(f"  Adding missing must_include term to positive prompt: {t_str}")
        
        preserve_terms_to_add = []
        for t in preserve_must_include:
            t_str = str(t).strip()
            if t_str:
                t_lower = t_str.lower()
                found_variant = False
                # First check protected_parts
                for i, p in enumerate(protected_parts):
                    if t_lower in p.lower() or p.lower() in t_lower:
                        preserve_terms_to_add.append(p)
                        protected_parts.pop(i)
                        found_variant = True
                        break
                # If not found, check filtered_parts
                if not found_variant:
                    for i, p in enumerate(filtered_parts):
                        if t_lower in p.lower() or p.lower() in t_lower:
                            preserve_terms_to_add.append(p)
                            filtered_parts.pop(i)
                            found_variant = True
                            break
                # If still not found, add the full term
                if not found_variant:
                    preserve_terms_to_add.append(t_str)
        
        # Deduplicate final list (case-insensitive)
        # Include protected_parts that weren't moved to change/preserve sections
        final_parts = []
        seen_lower = set()
        for part in (change_terms_to_add + preserve_terms_to_add + protected_parts + filtered_parts):
            part_lower = part.lower().strip()
            if part_lower and part_lower not in seen_lower:
                seen_lower.add(part_lower)
                final_parts.append(part)
        
        improved_positive = ", ".join(final_parts).strip(" ,\n")
        
        # Final safety check: ensure all change-intent must_include terms are in the positive prompt
        # This is a last resort in case they were removed by pruning or deduplication
        # Use word-level matching to avoid false positives (e.g., "summer" appearing alone doesn't mean "summer dress" is present)
        for t in change_must_include:
            t_str = str(t).strip()
            if t_str:
                t_lower = t_str.lower()
                # Extract key words from the must_include term (skip common words like "this", "lady", "wearing", "a")
                t_words = [w for w in t_lower.split() if len(w) > 3 and w not in ("this", "lady", "wearing", "the", "that")]
                # Check if ALL key words appear together in the prompt (not just individually)
                # This prevents false matches like "summer" alone matching "summer dress"
                prompt_lower = improved_positive.lower()
                all_words_present = all(w in prompt_lower for w in t_words) if t_words else False
                # Also check if the full phrase appears
                full_phrase_present = t_lower in prompt_lower
                
                if not (full_phrase_present or (len(t_words) > 1 and all_words_present)):
                    # Force add it at the front for maximum weight
                    improved_positive = f"{t_str}, {improved_positive}".strip(" ,\n")
                    self.logger.warning(f"  Force-added missing must_include term to front of positive prompt: {t_str}")

        # Ensure banned terms appear in negative
        if ban_terms:
            neg = (improved_negative or "").strip()
            for t in ban_terms:
                if t.lower() not in neg.lower():
                    if neg and not neg.endswith(','):
                        neg += ", "
                    neg += t
            improved_negative = neg.strip(" ,\n")

        # Sanity guards to prevent contradictory prompts:
        # - must_include should not be in negative
        # - ban_terms should not be in positive
        # Use a substring-based filter as well to catch plurals like "swimsuits"/"bikinis".
        improved_negative = _filter_csv_by_terms(improved_negative, must_include_terms)
        improved_positive = _filter_csv_by_terms(improved_positive, ban_terms)
        # Also keep avoid_terms out of positive as a stronger guard.
        improved_positive = _filter_csv_by_terms(improved_positive, avoid_terms)
        
        # Aggressive filter: if we have change-intent criteria with must_include terms,
        # remove any contradictory clothing terms from positive that match ban_terms or common opposites.
        # This prevents the AI from adding "formal business attire" when we want "snowsuit" or "summer dress".
        if change_must_include:
            # Build a list of terms to aggressively remove from positive
            contradictory_terms = list(ban_terms)  # Start with explicit ban_terms
            # Add common business/formal clothing terms if we're changing to casual/outdoor wear
            change_terms_lower = [str(t).strip().lower() for t in change_must_include]
            # Check for casual/outdoor wear terms (not just the original list)
            # Be specific: "summer dress" is casual, but "dress" alone could be formal
            is_casual_wear = any(
                "snowsuit" in t or "swimsuit" in t or "bikini" in t or "hawaiian" in t or "coconut" in t or
                "summer dress" in t or ("summer" in t and "dress" in t) or "casual" in t or "outdoor" in t
                for t in change_terms_lower
            )
            if is_casual_wear:
                contradictory_terms.extend([
                    "formal business attire", "business suit", "business attire", "formal attire",
                    "suit", "blazer", "dress shirt", "tie", "professional attire", "corporate attire",
                    "black blazer", "dark blue blazer", "gray blazer", "blue blazer", "navy blazer",
                    "dark blazer", "light blazer", "blazers", "suit jacket", "jacket"
                ])
            # Remove these from positive (run this filter multiple times to catch all variations)
            improved_positive = _filter_csv_by_terms(improved_positive, contradictory_terms)
            # Run again to catch any remaining variations
            improved_positive = _filter_csv_by_terms(improved_positive, contradictory_terms)
            self.logger.info(f"  Removed contradictory clothing terms (casual wear mode): {len(contradictory_terms)} terms")

        # Avoid dangling "no" tokens (these can confuse the image model and encourage drift).
        def _clean_tail(s: str) -> str:
            ss = (s or "").strip()
            if ss.lower().endswith(", no"):
                ss = ss[:-4].strip(" ,\n")
            elif ss.lower().endswith(" no"):
                # only remove if "no" is the last token
                parts = ss.split()
                if parts and parts[-1].lower() == "no":
                    ss = " ".join(parts[:-1]).strip(" ,\n")
            return ss

        def _strip_no_phrases_from_positive(pos: str) -> str:
            """Keep positive prompts positive.

            We frequently see LLMs emit lots of 'no X' fragments which weaken conditioning and
            lead to unstable results. Negations belong in the negative prompt.
            """
            p = (pos or "").strip()
            if not p:
                return p
            # Remove common broken fragments
            p = re.sub(r"\bno\s+on\s+body\b", "", p, flags=re.IGNORECASE)
            # Remove other unhelpful fragments we've seen the improver emit
            p = re.sub(r"\bon\s+the\s+body\b", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\bon\s+body\b", "", p, flags=re.IGNORECASE)
            p = re.sub(r"\btint\b", "", p, flags=re.IGNORECASE)
            # Remove 'no <word>' phrases
            p = re.sub(r"\bno\s+[a-zA-Z_]+\b", "", p, flags=re.IGNORECASE)
            # Remove standalone 'no'
            p = re.sub(r"\bno\b", "", p, flags=re.IGNORECASE)
            # Normalise separators/spaces
            p = re.sub(r"\s{2,}", " ", p)
            p = re.sub(r"\s*,\s*", ", ", p)
            p = re.sub(r"(,\s*){2,}", ", ", p)
            return p.strip(" ,\n")

        def _dedupe_csv(s: str) -> str:
            parts = [p.strip() for p in (s or "").split(",")]
            out = []
            seen = set()
            for p in parts:
                if not p:
                    continue
                k = p.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(p)
            return ", ".join(out).strip()

        improved_positive = _clean_tail(improved_positive)
        improved_negative = _clean_tail(improved_negative)
        improved_positive = _strip_no_phrases_from_positive(improved_positive)
        # Strip leading negation words from negative tokens (we want raw terms, not "no X" phrases).
        improved_negative = re.sub(r"(?i)\\b(no|not|without)\\s+", "", improved_negative).strip(" ,\n")
        improved_positive = _dedupe_csv(improved_positive)
        improved_negative = _dedupe_csv(improved_negative)

        # If we're preserving the background, aggressively remove common scene/background words
        # that are not present in the original description.
        try:
            od = (original_description or "").lower()
        except Exception:
            od = ""
        preserve_background = any(
            isinstance(c, dict) and (
                str(c.get("field") or "").lower() in ("background_identical", "background")
                or "identical background" in ", ".join(_listify(c.get("must_include"))).lower()
            )
            for c in criteria_defs
        )
        if preserve_background and od:
            env_words = [
                "forest", "jungle", "beach", "desert", "mountain", "snow", "street", "city",
                "bedroom", "kitchen", "bathroom", "studio", "office", "conference", "park",
            ]
            def _filter_env(s: str) -> str:
                parts = [p.strip() for p in (s or "").split(",")]
                out = []
                for p in parts:
                    if not p:
                        continue
                    pl = p.lower()
                    if any(w in pl for w in env_words) and not any(w in od for w in env_words if w in pl):
                        # If token mentions an environment keyword not found in original description, drop it.
                        # (e.g. "forest backdrop" when original is an office.)
                        continue
                    out.append(p)
                return ", ".join(out).strip(" ,\n")
            improved_positive = _filter_env(improved_positive)
            improved_negative = _filter_env(improved_negative)

        # Final canonicalisation to keep prompts tag-like.
        improved_positive = _canonicalise_csv_tags(improved_positive)
        improved_negative = _canonicalise_csv_tags(improved_negative)

        # Keep negatives focused: if we have explicit ban_terms, don't keep extra fluff.
        if ban_terms:
            improved_negative = _canonicalise_csv_tags(", ".join(ban_terms + _listify(improved_negative.split(","))))

        # FINAL safety check: ensure all change-intent must_include terms are in the positive prompt
        # This runs AFTER all filtering/canonicalisation to ensure they're never removed
        for t in change_must_include:
            t_str = str(t).strip()
            if t_str:
                t_lower = t_str.lower()
                # Extract key words from the must_include term
                t_words = [w for w in t_lower.split() if len(w) > 3 and w not in ("this", "lady", "wearing", "the", "that")]
                prompt_lower = improved_positive.lower()
                all_words_present = all(w in prompt_lower for w in t_words) if t_words else False
                full_phrase_present = t_lower in prompt_lower
                
                if not (full_phrase_present or (len(t_words) > 1 and all_words_present)):
                    # Force add it at the front for maximum weight (remove any existing instance first)
                    improved_positive = _filter_csv_by_terms(improved_positive, [t_str])
                    improved_positive = f"{t_str}, {improved_positive}".strip(" ,\n")
                    self.logger.warning(f"  FINAL force-add: missing must_include term added to front of positive prompt: {t_str}")

        # Generate diff info for logging
        def _tokenise_prompt(s: str) -> List[str]:
            """Rough tokeniser for prompt diffs: split on commas, normalise whitespace."""
            if not s:
                return []
            parts = [p.strip() for p in str(s).split(",")]
            out = []
            seen = set()
            for p in parts:
                if not p:
                    continue
                k = p.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(p)
            return out

        def _diff_tokens(before: str, after: str) -> Dict[str, List[str]]:
            b = _tokenise_prompt(before)
            a = _tokenise_prompt(after)
            bs = {x.lower() for x in b}
            as_ = {x.lower() for x in a}
            added = [x for x in a if x.lower() not in bs]
            removed = [x for x in b if x.lower() not in as_]
            return {"added": added, "removed": removed}
        
        pos_diff = _diff_tokens(current_positive, improved_positive)
        neg_diff = _diff_tokens(current_negative, improved_negative)
        
        diff_info = {
            "must_include_terms": must_include_terms,
            "ban_terms": ban_terms,
            "avoid_terms": avoid_terms,
            "pos_diff": pos_diff,
            "neg_diff": neg_diff,
        }

        return improved_positive, improved_negative, diff_info
