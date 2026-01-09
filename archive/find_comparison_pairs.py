#!/usr/bin/env python3
"""
Find image pairs for comparison - shows images with same params except one difference.
"""

from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """Parse parameter values from filename."""
    parts = filename.replace('.png', '').split('_')
    params = {}
    i = 0
    while i < len(parts):
        if parts[i] == 'denoise' and i + 1 < len(parts):
            try:
                params['denoise'] = float(parts[i + 1])
            except ValueError:
                pass
            i += 2
        elif parts[i] == 'cfg' and i + 1 < len(parts):
            try:
                params['cfg'] = float(parts[i + 1])
            except ValueError:
                pass
            i += 2
        elif parts[i] == 'steps' and i + 1 < len(parts):
            try:
                params['steps'] = int(float(parts[i + 1]))
            except ValueError:
                pass
            i += 2
        elif parts[i] == 'sampler' and i + 1 < len(parts) and parts[i + 1] == 'name' and i + 2 < len(parts):
            params['sampler_name'] = parts[i + 2]
            i += 3
        elif parts[i] == 'scheduler' and i + 1 < len(parts):
            params['scheduler'] = parts[i + 1]
            i += 2
        else:
            i += 1
    return params

def main():
    coarse_dir = Path("smart_sweep_results/coarse_scan")
    
    if not coarse_dir.exists():
        print(f"Directory not found: {coarse_dir}")
        return
    
    # Parse all images
    results = []
    for img_file in coarse_dir.glob("*.png"):
        params = parse_filename(img_file.name)
        params['filepath'] = str(img_file)
        params['filename'] = img_file.name
        results.append(params)
    
    print("="*60)
    print("FINDING COMPARISON PAIRS")
    print("="*60)
    print()
    
    # Group by all params except one
    for param_to_vary in ['denoise', 'cfg', 'steps', 'sampler_name', 'scheduler']:
        print(f"\n📊 Pairs varying {param_to_vary}:")
        print("-" * 60)
        
        # Group by all other params
        groups = defaultdict(list)
        for r in results:
            # Create key from all params except the one we're varying
            key_parts = []
            for p in ['denoise', 'cfg', 'steps', 'sampler_name', 'scheduler']:
                if p != param_to_vary and p in r:
                    key_parts.append(f"{p}={r[p]}")
            key = ", ".join(sorted(key_parts))
            groups[key].append(r)
        
        # Find groups with multiple values for the varied param
        found_pairs = False
        for key, group in groups.items():
            if len(group) > 1:
                # Check if they have different values for param_to_vary
                param_values = [r.get(param_to_vary) for r in group if param_to_vary in r]
                unique_values = set(param_values)
                if len(unique_values) > 1:
                    found_pairs = True
                    print(f"\n  {key}:")
                    for r in sorted(group, key=lambda x: x.get(param_to_vary, 0)):
                        param_val = r.get(param_to_vary, '?')
                        filename = Path(r.get('filepath', r.get('filename', ''))).name
                        print(f"    {param_to_vary}={param_val}: {filename}")
        
        if not found_pairs:
            print(f"  No pairs found (all images have same {param_to_vary} value)")
    
    print("\n" + "="*60)
    print("QUICK COMPARISON COMMANDS:")
    print("="*60)
    
    # Generate some example commands
    examples = []
    for param_to_vary in ['cfg', 'steps', 'sampler_name', 'scheduler']:
        groups = defaultdict(list)
        for r in results:
            key_parts = []
            for p in ['denoise', 'cfg', 'steps', 'sampler_name', 'scheduler']:
                if p != param_to_vary and p in r:
                    key_parts.append(f"{p}={r[p]}")
            key = ", ".join(sorted(key_parts))
            groups[key].append(r)
        
        for key, group in groups.items():
            if len(group) >= 2:
                sorted_group = sorted(group, key=lambda x: str(x.get(param_to_vary, '')))
                unique_vals = set(r.get(param_to_vary) for r in sorted_group)
                if len(unique_vals) > 1:
                    r1 = sorted_group[0]
                    r2 = sorted_group[1]
                    file1 = r1.get('filepath', r1.get('filename', ''))
                    file2 = r2.get('filepath', r2.get('filename', ''))
                    if file1 and file2:
                        examples.append((param_to_vary, file1, file2))
                        break
    
    for param, file1, file2 in examples[:5]:
        print(f"\npython3 quick_ollama_test.py \\")
        print(f"  {file1} \\")
        print(f"  {file2}")
        print(f"# Comparing {param} values")

if __name__ == '__main__':
    main()
