# Enhanced Parameter Sweep Script

## What the Workflow Does

This is an **image-to-image (img2img) workflow** that:

1. **Loads an image** - `ComfyUI_00278_.png`
2. **Encodes to latent space** using VAE
3. **Samples with prompt** - modifies the image based on text prompts
4. **Decodes and saves** the result

**Purpose**: Modify images based on text prompts while preserving the overall structure and composition. The denoise parameter controls how much the image changes.

## Enhanced Script Features

The script now supports varying **multiple parameters**:

### Currently Supported Parameters:

1. **denoise** (0.0-1.0): How much to modify the image
   - Range: `{"min": 0.5, "max": 1.0, "step": 0.1}`
   - Or list: `[0.5, 0.7, 0.9]`

2. **cfg** or **cfg_scale** (typically 1.0-20.0): How closely to follow prompt
   - Range: `{"min": 5.0, "max": 10.0, "step": 1.0}`
   - Or list: `[5.0, 7.0, 10.0]`

3. **steps** (integer): Number of sampling steps
   - List: `[20, 25, 30]`
   - More steps = better quality but slower

4. **sampler_name** (string): Which sampler algorithm to use
   - List: `["heun", "dpmpp_2m", "euler_ancestral"]`
   - Fast: `euler`, `dpm_fast`, `lcm`
   - Quality: `dpmpp_2m`, `dpmpp_3m_sde`, `dpmpp_sde`
   - Balanced: `heun`, `euler_ancestral`

5. **scheduler** (string): Noise schedule algorithm
   - List: `["karras", "normal", "exponential"]`
   - Options: `karras`, `exponential`, `normal`, `simple`, `ddim_uniform`

6. **seed** (integer): Random seed for reproducibility
   - Range: `{"min": 1, "max": 100, "step": 1}`
   - Or list: `[42, 123, 456]`

## Example Configurations

### Quick Test (Fewer Combinations)
```json
{
  "parameters": {
    "denoise": [0.7, 0.9],
    "cfg": [7.0, 9.0],
    "steps": [20],
    "sampler_name": ["heun"],
    "scheduler": ["karras"]
  }
}
```
**Total: 2 × 2 × 1 × 1 × 1 = 4 combinations**

### Comprehensive Sweep
```json
{
  "parameters": {
    "denoise": {"min": 0.5, "max": 1.0, "step": 0.1},
    "cfg": {"min": 5.0, "max": 10.0, "step": 1.0},
    "steps": [20, 25, 30],
    "sampler_name": ["heun", "dpmpp_2m", "euler_ancestral"],
    "scheduler": ["karras", "normal"]
  }
}
```
**Total: 6 × 6 × 3 × 3 × 2 = 648 combinations** (will take a very long time!)

### Balanced Sweep (Recommended)
```json
{
  "parameters": {
    "denoise": [0.6, 0.8, 1.0],
    "cfg": [6.0, 8.0, 10.0],
    "steps": [20, 25],
    "sampler_name": ["heun", "dpmpp_2m"],
    "scheduler": ["karras"]
  }
}
```
**Total: 3 × 3 × 2 × 2 × 1 = 36 combinations**

## Optimization Suggestions

### For Speed:
- Use fewer steps (15-20)
- Use faster samplers: `euler`, `dpm_fast`, `lcm`
- Reduce parameter ranges

### For Quality:
- Use more steps (30-40)
- Use quality samplers: `dpmpp_2m_sde`, `dpmpp_3m_sde`
- Test different schedulers

### For Testing:
- Start with extreme values (min/max) to see the range
- Then fill in middle values
- Focus on parameters that matter most (denoise, cfg, steps)

## Usage

```bash
cd /home/tclark/ComfyUI/ComfyScripts

# Use enhanced config
python3 comfyui_sweep.py sweep_config_enhanced.json

# Or use original simple config
python3 comfyui_sweep.py sweep_config.json
```

## Output

Files are saved with descriptive names including all varied parameters:
- `denoise_0.70_cfg_8.00_steps_25_sampler_heun_scheduler_karras_001.png`

The summary JSON includes all parameter combinations and results.
