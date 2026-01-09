# OpenRouter Setup Guide

OpenRouter provides access to vision models via API, which can be faster and more reliable than local Ollama for vision tasks.

## Setup

1. **Get an API Key**:
   - Sign up at https://openrouter.ai/
   - Go to API Keys section
   - Create a new key
   - Copy the key

2. **Set Environment Variable** (recommended):
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

   Or add to your shell profile (`~/.bashrc` or `~/.zshrc`):
   ```bash
   echo 'export OPENROUTER_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Configure in AIGen.yaml** (alternative):
   Edit `projects/{project_name}/config/AIGen.yaml`:
   ```yaml
   aivis:
     provider: "openrouter"
     model: "qwen/qwen-2-vl-7b-instruct:free"
     api_key: "your-api-key-here"  # Optional if using env var
     max_concurrent: 2
   ```

## Available Vision Models

Check https://openrouter.ai/models for current vision models and pricing. Common options:
- `qwen/qwen-2.5-vl-7b-instruct:free` - Qwen 2.5 vision model (FREE tier)
- `qwen/qwen3-vl-8b-instruct` - Qwen 3 vision model (check pricing)
- `qwen/qwen2.5-vl-32b-instruct` - Qwen 2.5 vision model (check pricing)
- `google/gemini-flash-1.5` - Google Gemini Flash (check pricing)

**Note**: Model names and pricing change frequently. Always check https://openrouter.ai/models for the latest available models and their pricing.

## Benefits

- **Faster**: Cloud-based, no local processing delays
- **No blocking**: Multiple concurrent requests supported
- **No local resources**: Doesn't use your GPU/CPU
- **More reliable**: Professional infrastructure

## Usage

Once configured, the system will automatically use OpenRouter instead of Ollama. No code changes needed!
