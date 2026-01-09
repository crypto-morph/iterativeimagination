# Smart Parameter Sweep - Two-Phase Approach

## Overview

This script uses a **two-phase approach** to efficiently find optimal parameter combinations:

1. **Phase 1: Coarse Scan** - Quickly tests parameter extremes and middle values
2. **Phase 2: Fine Sweep** - Focuses detailed testing on promising ranges found in Phase 1

## How It Works

### Phase 1: Coarse Scan

For each parameter, tests only **3 values**:
- Minimum value
- Middle value  
- Maximum value

**Example:** If `denoise` ranges from 0.3 to 0.7, it tests: `[0.3, 0.5, 0.7]`

This gives you a quick overview of how each parameter affects the output.

### Phase 2: Fine Sweep

After Phase 1, the script:
1. **Analyzes image differences** between parameter values
2. **Identifies promising ranges** where parameters have the most impact
3. **Generates detailed values** only in those promising ranges

**Example:** If Phase 1 shows that `denoise` values 0.4-0.6 produce the most variation, Phase 2 will test `[0.4, 0.45, 0.5, 0.55, 0.6]` instead of the full range.

## Benefits

✅ **Faster**: Tests fewer combinations overall  
✅ **Smarter**: Focuses on parameter ranges that actually matter  
✅ **Avoids duplicates**: Skips ranges that produce nearly identical images  
✅ **Triangulation**: Helps you find the "sweet spot" for each parameter

## Usage

```bash
cd /home/tclark/ComfyUI/ComfyScripts
python3 smart_sweep.py smart_sweep_config.json
```

## Output Structure

```
smart_sweep_results/
├── coarse_scan/              # Phase 1 results
│   ├── *.png                 # Coarse scan images
│   └── coarse_scan_summary.json
└── fine_sweep/               # Phase 2 results
    ├── *.png                 # Fine sweep images
    └── fine_sweep_summary.json
```

## Configuration

The config file (`smart_sweep_config.json`) defines:
- Parameter ranges (min, max, step)
- The workflow to use
- Output directory

**Note:** The `step` value is used in Phase 2 for fine sweeps. Phase 1 always uses 3 values (min, mid, max) regardless of step.

## Example Workflow

1. **Run Phase 1** (coarse scan):
   - Tests ~27 combinations (3 values × 3 values × 3 values...)
   - Takes ~10-15 minutes
   - Shows you which parameters matter most

2. **Script analyzes results**:
   - Compares images to find where parameters have most impact
   - Identifies promising ranges

3. **Run Phase 2** (fine sweep):
   - Tests detailed values only in promising ranges
   - Might test ~50-100 combinations (focused)
   - Takes ~30-60 minutes
   - Helps you triangulate optimal values

## Tips

- **Review Phase 1 results** before Phase 2 runs - you can manually adjust ranges if needed
- **Check the summary JSON files** - they contain all parameter combinations and file paths
- **Use image comparison tools** to visually compare results
- **Adjust step sizes** in config if you want finer or coarser Phase 2 sweeps

## How It Finds Promising Ranges

The script calculates **image differences** (using mean squared error) between adjacent parameter values. Ranges with **higher differences** indicate parameters that have more impact, so those ranges get more detailed testing in Phase 2.

This means:
- If `denoise` 0.3→0.4 produces big visual changes → Phase 2 tests 0.3-0.4 in detail
- If `cfg` 5.0→6.0 produces tiny changes → Phase 2 might skip that range or test it less
