# Ollama Setup for Image Comparison

## Overview

The smart sweep script can use Ollama vision models to semantically compare images, providing better analysis than pixel-based comparison.

## Setup

1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

3. **Pull a vision model**:
   ```bash
   ollama pull llava
   ```
   
   Other vision models you can try:
   - `llava` - Good general purpose vision model
   - `llava:13b` - Larger, more capable version
   - `bakllava` - Alternative vision model

4. **Test the connection**:
   ```bash
   python3 ollama_image_comparer.py <image1> <image2>
   ```

## Configuration

In `smart_sweep_config.json`, enable Ollama:

```json
{
  "ollama": {
    "enabled": true,
    "model": "llava",
    "base_url": "http://localhost:11434"
  }
}
```

## How It Works

When enabled, the script uses Ollama to:
1. **Compare images semantically** - Understands what changed, not just pixel differences
2. **Identify meaningful changes** - Distinguishes between "same person, different clothing" vs "completely different image"
3. **Provide analysis** - Describes what differences it sees

## Benefits Over Pixel Comparison

- **Semantic understanding**: Knows that "same person without shirt" is a meaningful change
- **Ignores noise**: Doesn't flag minor pixel variations as significant
- **Context aware**: Understands the task (removing clothing) and can assess success
- **Better range detection**: Finds parameter ranges that actually produce desired changes

## Performance

- Ollama comparisons are slower than pixel-based (~2-5 seconds per comparison)
- The script samples fewer comparisons when using Ollama (2×2 instead of 3×3)
- Still much faster than generating all combinations

## Troubleshooting

**"Cannot connect to Ollama"**
- Make sure `ollama serve` is running
- Check the `base_url` in config matches your Ollama setup

**"Model not found"**
- Run `ollama pull llava` to download the model
- Check the model name in config matches an installed model

**Slow comparisons**
- Use a smaller model: `llava:7b` instead of `llava:13b`
- Reduce the number of comparisons in the code (already optimized)
