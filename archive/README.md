# ComfyUI Parameter Sweep Script

A Python script that automates parameter sweeps for ComfyUI workflows by varying denoise strength and CFG scale values.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `sweep_config.json` to specify:

- **workflow_file**: Path to your ComfyUI workflow JSON file
- **comfyui**: Host and port where ComfyUI is running
- **parameters**: 
  - **denoise**: Range (min, max, step) or list of values
  - **cfg_scale**: Range (min, max, step) or list of values
- **output_dir**: Directory where results will be saved

### Range-based Configuration

```json
{
  "parameters": {
    "denoise": {
      "min": 0.5,
      "max": 1.0,
      "step": 0.1
    },
    "cfg_scale": {
      "min": 5.0,
      "max": 10.0,
      "step": 1.0
    }
  }
}
```

### List-based Configuration

```json
{
  "parameters": {
    "denoise": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "cfg_scale": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  }
}
```

## Usage

```bash
python comfyui_sweep.py sweep_config.json
```

The script will:
1. Load your workflow JSON
2. Generate all combinations of denoise and CFG values
3. Submit each combination to ComfyUI via the API
4. Save results with descriptive filenames like `denoise_0.5_cfg_7.0_001.png`
5. Generate a summary report in `sweep_summary.json`

## Output

Results are saved in the specified `output_dir` with filenames indicating the parameters used:
- `denoise_0.50_cfg_7.00_001.png`
- `denoise_0.60_cfg_7.00_002.png`
- etc.

A summary file `sweep_summary.json` contains metadata about all generated images.

## Notes

- The script automatically detects denoise and CFG parameters in your workflow
- It uses ComfyUI's WebSocket API for real-time progress tracking
- Make sure ComfyUI is running and accessible at the specified host/port
- If ComfyUI is running on the Windows host from WSL, use the host IP address (e.g., `172.20.224.1`) instead of `localhost`

