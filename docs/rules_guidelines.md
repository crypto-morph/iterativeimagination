## Rules.yaml acceptance_criteria tagging guidelines

These tags are used to drive prompt shaping and parameter tuning in a data-driven way.

### intent

- **change**: The image must change in a specific way (e.g., change clothing style, add/remove an object, change expression).
- **preserve**: The image must stay the same for this aspect (e.g., identity, pose, background, proportions).

If a criterion is phrased like "Is X identical to the original?", the intent is almost always **preserve**.
If it is phrased like "Has X been changed to ...?", the intent is almost always **change**.

### edit_strength

How hard it is to satisfy this criterion without breaking other constraints:

- **low**: Minor tweaks; should not push denoise/cfg much.
- **medium**: Typical; may require moderate denoise/cfg and stronger prompt emphasis.
- **high**: Difficult constraint; likely needs stronger denoise/cfg and strong prompt steering.

Heuristics:
- identity/face preservation is often **high**
- pose/background/proportions preservation is often **medium**
- large semantic changes (clothing/props) are often **high**

### must_include

A list of short prompt phrases that materially help satisfy the criterion.

Good:
- concise, specific descriptors: "formal business attire", "same pose", "identical background"
- include multiple synonyms only when helpful (not noise)

Avoid:
- full sentences
- contradictions with other criteria
- huge lists (prefer 3–8 strong phrases)

### ban_terms

Hard bans. Terms that must not appear in the result (and should usually be placed in the negative prompt).

Use when:
- the change criterion is frequently "stuck" because the old attribute persists
- a term is a strong attractor in the base image (e.g., "bikini" in a beach photo)

Keep it short and specific (3–12 terms). Prefer concrete nouns/adjectives.

### avoid_terms

Soft discouragement. Terms that you want to reduce, but not as strictly as ban_terms.

Use when:
- the term is sometimes acceptable, but tends to pull the result away from the target
- you want to steer style without overconstraining the model

### General rules

- Keep tags **task-specific** and **image-model-friendly** (phrases a Stable Diffusion prompt can use).
- Prefer **non-ambiguous** wording: "suit jacket" > "smart outfit".
- Do not add sexual content, minors, or disallowed content.
- Do not change field names/questions when suggesting tags; only adjust the tags.
