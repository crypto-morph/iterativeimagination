# Iterative Imagination

This tool will use AI vision (qwen3-vl:4b) and AI Image Generation (ComfyUI) to generate images from a source image with particular attributes

## Legacy

The tooling is an iteration of tooling built to do both tasks separately.  This new project can be self contained and properly software engineered - but can learn from the previous iterations

ComfyScripts/archive - initial experiments
ComfyScripts/flexible - computer vision experiment
ComfyScripts/archive2 - ComfyUI interator

## What this tool should do

The inputs are a rules file and an image - default input.png
The output will be a new image which conforms to the rule file - default output.png
The rules.yaml will dictate what the AI is aiming to do with the picture
The AIGen.yaml will allow AI to control the ComfyUI system
The workflow ComfyUI will use is based on workflow/img2img_no_mask_api.json - but with parameterisation - so that AIGen.yaml controls it. 

## Terms

The vision component - initially selected as qwen3-vl:4b will be referred to as "AIVis".
The generation component - initially selected as realisticVisionV60B1_v51VAE.safetensors + ComfyUI will be referred to as "AIGen". 

## Where will it store things?

Projects are stored in a `projects/` directory, with each project being self-contained:

**Project Structure** (`projects/{projectname}/`):
   - `config/rules.yaml` - Project-specific rules and acceptance criteria
   - `input/input.png` - Input image for the project
   - `working/` - Generated images and metadata for each iteration
     - `AIGen.yaml` - Current ComfyUI settings (updated each iteration by AI)
     - `iteration_{N}.png` - Generated images
     - `iteration_{N}_questions.json` - Question answers
     - `iteration_{N}_evaluation.json` - Acceptance criteria evaluation
     - `iteration_{N}_comparison.json` - Image comparison results
     - `iteration_{N}_metadata.json` - Complete iteration data
   - `logs/app.log` - Application logs
   - `output/` - Final output image and metadata
     - `output.png` - Final generated image
     - `output_metadata.json` - Final metadata

**Defaults** (`defaults/`):
   - `config/rules.yaml` - Template rules file
   - `config/AIGen.yaml` - Template AIGen configuration
   - `input/example.jpg` - Example input image
   - `workflow/img2img_no_mask_api.json` - Workflow template
   - `working/` - Empty directory (for AIGen.yaml updates during iterations)

**Creating a New Project**: Copy `defaults/` to `projects/{projectname}/` and customize:
   - Edit `projects/{projectname}/config/rules.yaml` for project-specific rules
   - Place input image in `projects/{projectname}/input/input.png`

## Rules

The rules.yaml file will contain:
   project.*  (project metadata)
   acceptance_criteria.* (questions which govern when a picture passes the test)
   questions.* (questions the AI will answer to help it come to the correct conclusion)

**Example**: The default `rules.yaml` demonstrates changing clothing style to formal business attire while preserving person identity, pose, and background. Additional example rules files are available in the `rules/` directory.

### Data Formats

**rules.yaml Structure**:
```yaml
project:
  name: string          # Used for project directory name
  max_iterations: int   # Maximum number of iterations to attempt

acceptance_criteria:
  - field: string      # Unique identifier
    question: string    # Question to evaluate
    type: boolean       # Type of answer expected
    min: 0              # Minimum value (for boolean: 0 = false, 1 = true)
    max: 1              # Maximum value

questions:
  - field: string      # Unique identifier
    question: string    # Question to ask AIVis
    type: string|integer|number|boolean|array
    min: number        # Optional minimum
    max: number        # Optional maximum
    enum: [list]       # Optional allowed values
    items: [list]      # For array types, item schema
```

**AIGen.yaml Structure**:
```yaml
workflow_file: string           # Relative path to workflow JSON
model:
  ckpt_name: string            # Model filename
comfyui:
  host: string                  # Default: "localhost"
  port: int                     # Default: 8188
parameters:
  denoise: float                # 0.0-1.0
  cfg: float                    # 1.0-20.0
  steps: int                    # Typically 20-30
  seed: int|null                # null = random
  sampler_name: string          # e.g. "dpmpp_2m"
  scheduler: string             # e.g. "karras"
prompts:
  positive: string              # Positive prompt text
  negative: string              # Negative prompt text
```

**Iteration Metadata JSON**:
```json
{
  "iteration": 1,
  "timestamp": 1234567890.0,
  "image_path": "working/iteration_1.png",
  "parameters_used": {
    "denoise": 0.5,
    "cfg": 7.0,
    "steps": 25,
    "seed": 12345,
    "sampler_name": "dpmpp_2m",
    "scheduler": "karras"
  },
  "questions": {
    "person_clothing_description": "formal business suit with tie",
    "clothing_formal_elements": "suit jacket, dress shirt, tie",
    "facial_features_similarity": 0.92,
    ...
  },
  "evaluation": {
    "overall_score": 85,
    "criteria_results": {
      "clothing_style_changed": true,
      "person_facial_features": true,
      ...
    }
  },
  "comparison": {
    "similarity_score": 0.70,
    "differences": ["clothing changed to formal attire", "background unchanged"],
    "analysis": "..."
  }
}
```

## Technical Implementation Details

### ComfyUI Integration
- **API Endpoint**: `http://{host}:{port}/prompt` (POST) to queue workflows
- **WebSocket**: `ws://{host}:{port}/ws?clientId={uuid}` for progress tracking
- **History API**: `http://{host}:{port}/history/{prompt_id}` to get execution results
- **Image Download**: `http://{host}:{port}/view?filename={filename}&subfolder={subfolder}&type=output`
- **Workflow Loading**: Load JSON from `workflow_file` path in AIGen.yaml
- **Workflow Updates Required**:
  - Update CheckpointLoaderSimple node with `model.ckpt_name`
  - Update KSampler node with parameters: `denoise`, `cfg`, `steps`, `seed`, `sampler_name`, `scheduler`
  - Update CLIPTextEncode nodes (node "9" = positive, node "10" = negative) with `prompts.positive` and `prompts.negative`
  - Update LoadImage node with input image path
- **Image Output**: Save to `{projectname}/working/iteration_{N}.png` with metadata JSON

### AIVis Integration (qwen3-vl:4b)
- **API Endpoint**: `http://localhost:11434/api/generate` (POST)
- **Image Format**: Base64-encoded PNG (resize to max 1024px for performance)
- **Timeout**: 180 seconds per request
- **Retry Logic**: 2 retry attempts with 5 second delay
- **Response Format**: JSON (request with `"format": "json"`)
- **JSON Parsing**: Handle markdown code blocks (```json ... ```) in responses

### AIVis Evaluation Process
1. **Answer Questions**: For each question in `rules.yaml`, send image + question to AIVis
   - Format: Structured prompt requesting JSON response matching question schema
   - Store answers in `{projectname}/working/iteration_{N}_questions.json`
2. **Evaluate Acceptance Criteria**: For each criterion, determine pass/fail
   - Use question answers to inform evaluation
   - Calculate score: `(passed_criteria / total_criteria) * 100`
   - Store evaluation in `{projectname}/working/iteration_{N}_evaluation.json`
3. **Compare Images**: Compare generated vs original image
   - Calculate similarity score (0.0 = different, 1.0 = identical)
   - Identify differences (clothing, pose, features, background, proportions)
   - Store comparison in `{projectname}/working/iteration_{N}_comparison.json`

### Iteration Logic
1. **Initial Setup**:
   - Load project from `projects/{projectname}/config/rules.yaml`
   - Load input image from `projects/{projectname}/input/input.png`
   - Create project directory structure if needed (working/, logs/, output/)
   - Copy initial `AIGen.yaml` from `projects/{projectname}/config/AIGen.yaml` to `projects/{projectname}/working/AIGen.yaml` (or use from defaults if not present)
   - Describe original image once (cache for all iterations)

2. **Each Iteration**:
   - Load `AIGen.yaml` from `projects/{projectname}/working/`
   - Load workflow template from path in AIGen.yaml (relative to project root or absolute path)
   - Update workflow with AIGen.yaml parameters
   - Submit to ComfyUI API
   - Wait for completion (WebSocket + polling fallback)
   - Download generated image
   - Save to `projects/{projectname}/working/iteration_{N}.png`
   - Answer all questions from `projects/{projectname}/config/rules.yaml`
   - Evaluate against acceptance criteria
   - Compare with original image
   - Calculate overall score

3. **Decision Making**:
   - **Score = 100%**: Stop, copy image to `projects/{projectname}/output/output.png`, save metadata
   - **Score < 100% AND iterations < max_iterations**:
     - If similarity > 0.85: Images too similar → increase `denoise` (+0.05) and `cfg` (+1.0)
     - If similarity < 0.3: Images too different → decrease `denoise` (-0.05)
     - Based on failed criteria, generate improved prompts using AIVis text model
     - Update `projects/{projectname}/working/AIGen.yaml` with new parameters and prompts
     - Continue to next iteration
   - **Iterations >= max_iterations**: Stop, use best scoring iteration

### Prompt Improvement
- Use AIVis text model (qwen3-vl:4b can handle text) or separate text model
- Input: Current prompt, evaluation results, failed criteria, comparison analysis
- Output: Improved positive/negative prompts
- Update `AIGen.yaml` prompts section

### Logging
- Application logs: `projects/{projectname}/logs/app.log` (all operations, errors, decisions)
- Iteration metadata: Each iteration saves JSON files in `projects/{projectname}/working/`:
  - `iteration_{N}.png` - Generated image
  - `iteration_{N}_questions.json` - Question answers
  - `iteration_{N}_evaluation.json` - Acceptance criteria evaluation
  - `iteration_{N}_comparison.json` - Image comparison results
  - `iteration_{N}_metadata.json` - Complete iteration data (parameters, scores, etc.)

### Error Handling
- ComfyUI connection failures: Retry 3 times, then fail iteration
- AIVis timeouts: Retry 2 times with 5s delay, then use fallback evaluation
- JSON parsing failures: Extract JSON from markdown, fallback to text parsing
- Workflow execution failures: Log error, skip iteration, continue if iterations remain

### File Structure Example
```
ComfyScripts/
├── defaults/                     # Template structure (copy to create new project)
│   ├── config/
│   │   ├── rules.yaml            # Template rules
│   │   └── AIGen.yaml            # Template AIGen config
│   ├── input/
│   │   └── example.jpg           # Example input image
│   ├── workflow/
│   │   └── img2img_no_mask_api.json  # Workflow template
│   └── working/                  # Empty (for AIGen.yaml updates)
└── projects/
    └── example2/                 # Example project (copied from defaults)
        ├── config/
        │   ├── rules.yaml        # Project-specific rules
        │   └── AIGen.yaml        # (copied from defaults, updated during iterations)
        ├── input/
        │   └── input.png         # Project input image
        ├── working/
        │   ├── AIGen.yaml        # Current config (updated each iteration)
        │   ├── iteration_1.png
        │   ├── iteration_1_questions.json
        │   ├── iteration_1_evaluation.json
        │   ├── iteration_1_comparison.json
        │   ├── iteration_1_metadata.json
        │   └── ...
        ├── logs/
        │   └── app.log           # Application logs
        └── output/
            ├── output.png        # Final image
            └── output_metadata.json  # Final metadata
```

### Command Line Interface
```bash
python iterative_imagination.py [options]

Options:
  --project NAME       Project name (uses projects/{NAME}/config/rules.yaml)
  --rules PATH         Path to rules.yaml (alternative to --project)
  --input PATH         Path to input image (if not using project structure)
  --dry-run           Validate configs but don't run
  --verbose           Enable debug logging
```

### Dependencies
- Python 3.8+
- `requests` - HTTP client for ComfyUI and Ollama APIs
- `websocket-client` - WebSocket client for ComfyUI progress
- `Pillow` - Image processing (resize, format conversion)
- `PyYAML` - YAML parsing
- `pathlib` - Path handling (standard library)

### Environment Requirements
- ComfyUI server running on localhost:8188 (or configured host/port)
- Ollama running on localhost:11434 with qwen3-vl:4b model installed
- Sufficient disk space for project directories and generated images

## Workflow

1) The application will take the original image and pass it through ComfyUI with a preset Workflow which is configurable by AIGen.yaml
2) AIGen will produce a new image
3) AIVis will answer "questions" about the image to help it think correctly about the answers to the next section
4) AIVis will "mark" the image based on the acceptance_criteria
5) AIVis will also check the generated image against the original and decide how different the image is.
6) The mark will govern what the AI will do next:
   * If the mark is 100% - the process will stop and the image will be output
   * If the input and output images are too similar or too different - the model will be shown the AIGen.yaml and will suggest improvements - which will be applied.
   * Based on the criteria the image hit and missed - AIVis can also change the positive and negative prompts in AIGen.yaml  
   * If the images are not hitting all the acceptance_criteria then the system iterates again up to the max iterations set in the project file.


