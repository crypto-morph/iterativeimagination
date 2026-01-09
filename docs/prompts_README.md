# LLM Prompts Configuration

All LLM prompts used by the Iterative Imagination system are now stored in `prompts.yaml` files, making them easy to edit without touching Python code.

## Location

The system looks for prompts in this order:
1. `projects/{project_name}/config/prompts.yaml` (project-specific prompts)
2. `defaults/prompts.yaml` (default prompts for all projects)
3. Built-in fallback prompts (if neither file exists)

## Editing Prompts

Simply edit the `prompts.yaml` file with your preferred text editor. The file uses YAML format with these sections:

- **ask_question** - Prompt for answering questions about images
- **evaluate_acceptance_criteria** - Prompt for evaluating images against criteria
- **compare_images** - Prompt for comparing two images
- **describe_image** - Prompt for describing an image
- **improve_prompts** - Prompt for improving Stable Diffusion prompts

## Placeholders

Prompts use `{placeholders}` that are automatically filled in by the code:

### ask_question
- `{question}` - The question text
- `{type_info}` - Question type (string, boolean, number, etc.)
- `{enum_info}` - Allowed values (if applicable)
- `{min_max_info}` - Value range (if applicable)

### evaluate_acceptance_criteria
- `{original_description}` - Description of the original image
- `{questions_summary}` - Summary of question answers
- `{criteria_text}` - List of acceptance criteria

### improve_prompts
- `{current_positive}` - Current positive prompt
- `{current_negative}` - Current negative prompt
- `{overall_score}` - Evaluation score (0-100)
- `{failed_criteria}` - List of failed criteria
- `{similarity_score}` - Image similarity score (0.0-1.0)
- `{differences}` - List of differences found
- `{analysis}` - Analysis text
- `{rules_text}` - Rules to achieve

### compare_images and describe_image
- No placeholders (static prompts)

## JSON Formatting

When prompts include JSON examples, use double braces `{{` and `}}` to escape them:
```yaml
{{
    "field": "value"
}}
```

## Example: Project-Specific Prompts

To customise prompts for a specific project:

1. Copy `defaults/prompts.yaml` to `projects/{project_name}/config/prompts.yaml`
2. Edit the prompts as needed
3. The project will use your custom prompts automatically

## Tips

- Keep prompts clear and specific
- Include examples in the prompt when possible
- Test changes with a small iteration first
- Use YAML's `|` (pipe) syntax for multi-line prompts (already done in the file)
