# Iterative Imagination (ComfyScripts)

This tool uses **AI vision (AIVis)** and **AI image generation (AIGen via ComfyUI)** to iteratively transform an image until a project’s `rules.yaml` acceptance criteria are met (or a maximum iteration count is reached).

## Goals

- Make **controlled, local edits** to an existing image (usually via mask + inpaint) while preserving everything else.
- Provide a **rules-driven, repeatable** loop: generate → evaluate → adjust → repeat.
- Make projects **generic**: any image can be refined if it has a suitable `rules.yaml`.
- Provide a **single place to edit projects**: viewer UI for masks and rules.

## Non-goals (for now)

- Fully automatic multi-pass “run all masks sequentially” as the default run mode (there is support, but it is not the only mode).
- Perfect, universal prompt quality without any project-specific rules tuning.

## Terms

- **AIVis**: the vision component (provider and model configurable; OpenRouter or local Ollama).
- **AIGen**: the generation component (ComfyUI + checkpoint + workflow).
- **Mask**: a per-pixel edit region used for local edits via inpainting.
- **Scope**: either “global/unmasked” or “mask: <name>”, used to decide which criteria are active.

## Project structure

Projects live under `projects/` and are self-contained:

- `projects/<name>/config/rules.yaml`: acceptance criteria, mask membership, and base prompts
- `projects/<name>/config/AIGen.yaml`: ComfyUI workflow choice, model, parameters, masking config, prompts
- `projects/<name>/config/AIVis.yaml`: vision provider and model selection (including fallbacks)
- `projects/<name>/config/prompts.yaml` (optional): per-project AIVis prompt overrides
- `projects/<name>/input/input.png`: the original input image (never overwritten by the runner)
- `projects/<name>/input/progress.png` (optional): runner-managed evolving input used for iterative refinement
- `projects/<name>/input/mask.png` (optional): legacy single mask
- `projects/<name>/input/masks/<mask>.png` (optional): named masks for multi-mask projects
- `projects/<name>/input/mask.anchor.json` and `projects/<name>/input/masks/<mask>.anchor.json` (optional): anchor points for mask suggestion
- `projects/<name>/working/`:
  - `AIGen.yaml`: the working generation config updated each iteration
  - `checkpoint.json`: resume state
  - `original_description.txt` (optional): cached description of the current input image
  - `<run_id>/`:
    - `images/iteration_<N>.png`
    - `questions/iteration_<N>_questions.json`
    - `evaluation/iteration_<N>_evaluation.json`
    - `comparison/iteration_<N>_comparison.json`
    - `metadata/iteration_<N>_metadata.json`
    - `human/ranking.json` (optional)
- `projects/<name>/logs/app.log`
- `projects/<name>/output/output.png` and `output_metadata.json`

Defaults live under `defaults/` and are copied into new projects.

## Configuration formats

### `rules.yaml`

Top-level sections:

- `project`: metadata and global knobs (e.g. `max_iterations`, optional `lock_seed`)
- `acceptance_criteria`: criteria evaluated by AIVis
- `questions`: question prompts answered by AIVis (batched)
- `masking`: **membership model** mapping mask name → which criteria are active for that mask
- `prompts`: **base prompts** (global and per-mask) generated from the original description and active criteria

Example schema:

```yaml
project:
  name: string
  max_iterations: int
  lock_seed: bool  # optional

acceptance_criteria:
  - field: string
    question: string
    type: boolean|string|integer|number|array
    min: 0
    max: 1
    intent: change|preserve
    edit_strength: low|medium|high
    must_include: [string]
    ban_terms: [string]
    avoid_terms: [string]

masking:
  masks:
    - name: default
      active_criteria: [field1, field2]
    - name: left
      active_criteria: [fieldA, fieldB]

prompts:
  global:
    positive: string
    negative: string
  masks:
    left:
      positive: string
      negative: string

questions:
  - field: string
    question: string
    type: string|integer|number|boolean|array
```

### `AIGen.yaml`

Key sections:

- `workflow_file`: workflow file path (JSON or workflow PNG)
- `model.ckpt_name`: base checkpoint
- `model.ckpt_name_inpaint` (optional): inpaint-tuned checkpoint used when masking/inpainting is active
- `masking.enabled`: master on/off switch
- `masking.active_mask`: selects a named mask scope
- `masking.masks`: optional mapping of mask names to files
- `parameters`: sampler parameters (denoise, cfg, steps, seed, etc.)
- `prompts`: the current prompts used by the workflow

Example schema:

```yaml
workflow_file: defaults/workflow/img2img_no_mask_api.json
model:
  ckpt_name: realisticVisionV60B1_v51VAE.safetensors
  ckpt_name_inpaint: realisticVisionV60B1_v51VAE-inpainting.safetensors
masking:
  enabled: false
  active_mask: left
  masks:
    - name: left
      file: input/masks/left.png
parameters:
  denoise: 0.4
  cfg: 6.0
  steps: 25
  seed: null
prompts:
  positive: ""
  negative: ""
```

### `AIVis.yaml`

Select provider and models:

```yaml
provider: "openrouter" | "ollama"
model: string
fallback_provider: "openrouter" | "ollama"
fallback_model: string
# api_key: ""  # optional, otherwise OPENROUTER_API_KEY
```

## Workflows and workflow updating

Workflows are loaded from JSON, and workflow PNGs are supported by extracting embedded workflow JSON metadata.

The runner updates the workflow by:

- Setting the checkpoint (and using `ckpt_name_inpaint` when inpainting is active)
- Setting `KSampler` parameters
- Setting prompts by **graph traversal** to the `CLIPTextEncode` nodes connected to the `KSampler` positive and negative inputs (no hard-coded node IDs)
- Wiring the input image (`input.png` or `progress.png`)
- Wiring the mask (for inpaint workflows)
- Wiring a control image (for ControlNet workflows) when `input/control.png` exists

## Masking and inpainting behaviour

- Masking is enabled by `AIGen.yaml -> masking.enabled`.
- If a mask is present and the current workflow is not an inpaint workflow, the runner will switch to an inpaint workflow candidate.
- If the mask is detected as all-white, the runner treats it as “no mask” and runs without inpainting.

## Base prompts (rules-driven)

Base prompts are generated from:

- the cached original description of the current input image
- the active criteria for a scope (global or mask membership)

These are stored in `rules.yaml -> prompts`.

Runner behaviour:

- If `working/AIGen.yaml` prompts are empty, the runner initialises them from `rules.yaml` prompts:
  - Prefer `prompts.masks[active_mask]`
  - Else fall back to `prompts.global`

Viewer behaviour:

- Rules UI has a **Generate base prompts** button which writes the prompts into `config/rules.yaml`.

## AIVis evaluation and scoring

Per iteration:

1. Answer all `questions` in a single batched request.
2. Evaluate `acceptance_criteria` and produce per-field results.
3. Compare original vs generated image for similarity and diffs.

Important:

- `overall_score` is computed deterministically from the criteria results. Model-provided `overall_score` is not trusted.

## Iteration loop (high-level)

1. Load project config and input image (prefer `input/progress.png` if it exists).
2. Generate image via ComfyUI.
3. Evaluate with AIVis (questions, acceptance criteria, image comparison).
4. If score is 100%, stop and write:
   - `output/output.png`
   - `input/progress.png` (best image so far)
5. Otherwise, update working parameters and prompts and continue until `max_iterations`.

## Viewer (project UI)

The viewer provides:

- Run browsing and iteration inspection
- Mask editor supporting multiple masks
- Rules UI (mask-aware membership model)
- Mask suggestion (text-to-mask) using ComfyUI GroundingDINO + SAM2, with optional focus and anchor point

## CLI (`iterativectl`)

`iterativectl` provides convenience commands, including:

- `iterativectl run --project <name> ...`
- `iterativectl variants --project <name> [--mask] [--force]` for A/B testing workflow variants
- `iterativectl viewer start|stop|status`

## Security and repository hygiene

- Restricted projects should not be committed.
- Generated artefacts under `projects/*/working/`, `projects/*/logs/`, `projects/*/output/` are not committed.
- User input masks, anchors, and rules backups are treated as local project data and are ignored by default.

