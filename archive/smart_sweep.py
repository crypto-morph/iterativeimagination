#!/usr/bin/env python3
"""
Smart Parameter Sweep - Two Phase Approach

Phase 1: Coarse scan to find promising parameter ranges
Phase 2: Fine sweep in promising ranges to triangulate optimal values
"""

import json
import os
import sys
import time
import uuid
import websocket
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from PIL import Image
import numpy as np

# Try to import Ollama comparer
try:
    from ollama_image_comparer import OllamaImageComparer
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaImageComparer = None


class SmartSweep:
    """Two-phase parameter sweep with intelligent range detection."""
    
    def __init__(self, config_path: str):
        """Initialise the smart sweep with configuration."""
        self.config = self._load_config(config_path)
        workflow_path = self.config['workflow_file']
        
        workflow_data = self._load_workflow(workflow_path)
        if 'nodes' in workflow_data:
            raise ValueError("Workflow must be in API format")
        
        self.prompt = workflow_data
        self.output_dir = Path(self.config.get('output_dir', 'smart_sweep_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coarse_dir = self.output_dir / 'coarse_scan'
        self.fine_dir = self.output_dir / 'fine_sweep'
        self.coarse_dir.mkdir(parents=True, exist_ok=True)
        self.fine_dir.mkdir(parents=True, exist_ok=True)
        
        # ComfyUI connection
        self.host = self.config['comfyui']['host']
        self.port = self.config['comfyui']['port']
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}/ws?clientId={uuid.uuid4()}"
        
        # Parameter configuration
        self.param_configs = self.config['parameters']
        
        # Results tracking
        self.coarse_results = []
        self.fine_results = []
        
        # Ollama comparer (optional)
        self.ollama_comparer = None
        if OLLAMA_AVAILABLE:
            ollama_config = self.config.get('ollama', {})
            if ollama_config.get('enabled', False):
                try:
                    self.ollama_comparer = OllamaImageComparer(
                        model=ollama_config.get('model', 'llava'),
                        base_url=ollama_config.get('base_url', 'http://localhost:11434')
                    )
                    if self.ollama_comparer.test_connection():
                        print("✓ Ollama vision model available for semantic comparison")
                    else:
                        print("⚠ Ollama not accessible, falling back to pixel comparison")
                        self.ollama_comparer = None
                except Exception as e:
                    print(f"⚠ Could not initialise Ollama: {e}")
                    self.ollama_comparer = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Load workflow JSON file (API format)."""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_coarse_values(self, param_name: str, param_config: Dict) -> List:
        """Generate coarse scan values (min, max, middle)."""
        if isinstance(param_config, list):
            # For lists, take first, last, and middle
            if len(param_config) <= 3:
                return param_config
            mid = len(param_config) // 2
            return [param_config[0], param_config[mid], param_config[-1]]
        
        if 'min' in param_config and 'max' in param_config:
            min_val = param_config['min']
            max_val = param_config['max']
            mid_val = (min_val + max_val) / 2
            
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                if param_config.get('step', 1.0) >= 1.0 and isinstance(min_val, int):
                    return [int(min_val), int(mid_val), int(max_val)]
                else:
                    return [round(min_val, 3), round(mid_val, 3), round(max_val, 3)]
        
        raise ValueError(f"Invalid parameter configuration for {param_name}: {param_config}")
    
    def _generate_fine_values(self, param_name: str, param_config: Any, min_val: Any, max_val: Any) -> List:
        """Generate fine sweep values in a specific range."""
        if isinstance(param_config, list):
            # For lists, filter to range
            values = [v for v in param_config if min_val <= v <= max_val]
            if not values:
                # If no values in range, add min and max
                values = [min_val, max_val]
            return sorted(set(values))
        
        if isinstance(param_config, dict) and 'min' in param_config and 'max' in param_config:
            step = param_config.get('step', 1.0)
            values = []
            current = min_val
            while current <= max_val:
                if isinstance(current, int) and step >= 1.0:
                    values.append(int(current))
                else:
                    values.append(round(current, 3))
                current += step
            return values
        
        raise ValueError(f"Invalid parameter configuration for {param_name}: {param_config}")
    
    def _find_parameter_nodes(self) -> Dict[str, Optional[str]]:
        """Find nodes containing various parameters."""
        nodes = {}
        
        for node_id, node_data in self.prompt.items():
            if not isinstance(node_data, dict):
                continue
            
            node_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            if node_type in ['KSampler', 'KSamplerAdvanced']:
                if 'denoise' in inputs:
                    nodes['denoise'] = node_id
                if 'cfg' in inputs:
                    nodes['cfg'] = node_id
                if 'steps' in inputs:
                    nodes['steps'] = node_id
                if 'sampler_name' in inputs:
                    nodes['sampler_name'] = node_id
                if 'scheduler' in inputs:
                    nodes['scheduler'] = node_id
                if 'seed' in inputs:
                    nodes['seed'] = node_id
        
        return nodes
    
    def _update_prompt_parameters(self, **kwargs) -> Dict[str, Any]:
        """Create a copy of the prompt with updated parameters."""
        import random
        prompt_copy = json.loads(json.dumps(self.prompt))
        param_nodes = self._find_parameter_nodes()
        
        for param_name, param_value in kwargs.items():
            node_id = param_nodes.get(param_name)
            if node_id and node_id in prompt_copy:
                if 'inputs' in prompt_copy[node_id]:
                    inputs = prompt_copy[node_id]['inputs']
                    
                    if param_name == 'cfg':
                        if 'cfg' in inputs:
                            inputs['cfg'] = param_value
                        elif 'cfg_scale' in inputs:
                            inputs['cfg_scale'] = param_value
                    else:
                        if param_name in inputs:
                            inputs[param_name] = param_value
        
        # Always vary seed
        if 'seed' in param_nodes:
            seed_node_id = param_nodes['seed']
            if seed_node_id and seed_node_id in prompt_copy:
                if 'inputs' in prompt_copy[seed_node_id]:
                    prompt_copy[seed_node_id]['inputs']['seed'] = random.randint(0, 2**32 - 1)
        
        return prompt_copy
    
    def _queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt via ComfyUI REST API."""
        data = json.dumps({"prompt": prompt}).encode('utf-8')
        response = requests.post(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()['prompt_id']
    
    def _get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt ID."""
        response = requests.get(f"{self.base_url}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()
    
    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Download an image from ComfyUI."""
        data = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        response = requests.get(f"{self.base_url}/view", params=data)
        response.raise_for_status()
        return response.content
    
    def _run_workflow(self, params: Dict[str, Any], output_dir: Path, index: int) -> Optional[str]:
        """Run a single workflow and return image path."""
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"  [{index}] {param_str}")
        
        updated_prompt = self._update_prompt_parameters(**params)
        
        try:
            prompt_id = self._queue_prompt(updated_prompt)
        except Exception as e:
            print(f"    Error: {e}")
            return None
        
        # WebSocket tracking (simplified)
        ws = websocket.WebSocket()
        try:
            ws.connect(self.ws_url)
            ws.settimeout(1.0)
            
            max_wait = 300
            start_time = time.time()
            completed = False
            
            while time.time() - start_time < max_wait:
                try:
                    message = ws.recv()
                    if message:
                        data = json.loads(message)
                        if data.get('type') == 'executing':
                            executing_data = data.get('data', {})
                            if executing_data.get('prompt_id') == prompt_id:
                                if executing_data.get('node') is None:
                                    completed = True
                                    break
                except websocket.WebSocketTimeoutException:
                    elapsed = time.time() - start_time
                    if elapsed > 5 and int(elapsed) % 3 == 0:
                        try:
                            history = self._get_history(prompt_id)
                            if prompt_id in history:
                                status = history[prompt_id].get('status', {})
                                if status.get('completed'):
                                    completed = True
                                    break
                        except:
                            pass
                    continue
                except Exception as e:
                    break
            
            ws.close()
            
            if not completed:
                return None
            
            time.sleep(0.5)
            
            # Get image
            for attempt in range(5):
                try:
                    history = self._get_history(prompt_id)
                    if prompt_id not in history:
                        if attempt < 4:
                            time.sleep(0.5)
                            continue
                        return None
                    
                    outputs = history[prompt_id].get('outputs', {})
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            images = node_output['images']
                            if len(images) > 0:
                                image = images[0]
                                image_data = self._get_image(
                                    image['filename'],
                                    image.get('subfolder', ''),
                                    image.get('type', 'output')
                                )
                                
                                # Save image
                                filename_parts = [f"{k}_{v}" for k, v in params.items()]
                                filename = "_".join(filename_parts) + f"_{index:03d}.png"
                                filepath = output_dir / filename
                                with open(filepath, 'wb') as f:
                                    f.write(image_data)
                                
                                return str(filepath)
                    
                    if attempt < 4:
                        time.sleep(0.5)
                        continue
                except Exception as e:
                    if attempt < 4:
                        time.sleep(0.5)
                        continue
                    return None
            
            return None
        
        except Exception as e:
            print(f"    Error: {e}")
            return None
        finally:
            if ws.connected:
                ws.close()
    
    def _calculate_image_difference(self, img1_path: str, img2_path: str, 
                                    params1: Dict = None, params2: Dict = None) -> float:
        """
        Calculate difference between two images (0-1 scale).
        Uses Ollama for semantic comparison if available, otherwise pixel-based.
        """
        # Try Ollama first if available
        if self.ollama_comparer:
            try:
                comparison = self.ollama_comparer.compare_images(
                    img1_path, img2_path, params1, params2
                )
                # Convert similarity score to difference (1 - similarity)
                similarity = comparison.get('similarity_score', 0.5)
                difference = 1.0 - similarity
                
                # If meaningful change detected, boost difference slightly
                if comparison.get('meaningful_change', False):
                    difference = min(1.0, difference * 1.2)
                
                return difference
            except Exception as e:
                print(f"    Ollama comparison failed, using pixel diff: {e}")
                # Fall through to pixel-based comparison
        
        # Fallback to pixel-based comparison
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Resize to same size if needed
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            
            # Calculate mean squared error
            mse = np.mean((arr1 - arr2) ** 2)
            # Normalize to 0-1 scale (max MSE for 0-255 range is 255^2)
            normalized_diff = mse / (255.0 ** 2)
            
            return normalized_diff
        except Exception as e:
            print(f"    Error calculating difference: {e}")
            return 1.0  # Assume maximum difference on error
    
    def _analyze_coarse_results(self) -> Dict[str, Tuple[Any, Any]]:
        """Analyze coarse scan results to find promising ranges."""
        print("\n" + "="*60)
        print("Analyzing coarse scan results...")
        print("="*60)
        
        if len(self.coarse_results) == 0:
            print("No coarse results to analyze!")
            return {}
        
        # Group results by parameter
        param_ranges = {}
        
        for param_name in self.param_configs.keys():
            print(f"\nAnalyzing {param_name}...")
            
            # Get unique values for this parameter
            param_values = sorted(set(r[param_name] for r in self.coarse_results if param_name in r))
            
            if len(param_values) < 2:
                # Not enough variation, use full range
                if isinstance(self.param_configs[param_name], dict):
                    param_ranges[param_name] = (
                        self.param_configs[param_name]['min'],
                        self.param_configs[param_name]['max']
                    )
                else:
                    param_ranges[param_name] = (param_values[0], param_values[-1])
                print(f"  Using full range: {param_ranges[param_name]}")
                continue
            
            # Calculate differences between adjacent values
            differences = []
            for i in range(len(param_values) - 1):
                val1 = param_values[i]
                val2 = param_values[i + 1]
                
                # Find results with these values (keeping other params similar)
                results1 = [r for r in self.coarse_results if r.get(param_name) == val1]
                results2 = [r for r in self.coarse_results if r.get(param_name) == val2]
                
                if results1 and results2:
                    # Compare images
                    avg_diff = 0.0
                    count = 0
                    meaningful_changes = []
                    
                    for r1 in results1[:2]:  # Sample a few (reduced for Ollama speed)
                        for r2 in results2[:2]:
                            if r1.get('filepath') and r2.get('filepath'):
                                # Create param dicts for context
                                params1 = {param_name: val1}
                                params2 = {param_name: val2}
                                
                                diff = self._calculate_image_difference(
                                    r1['filepath'], r2['filepath'], params1, params2
                                )
                                avg_diff += diff
                                count += 1
                                
                                # If using Ollama, check for meaningful changes
                                if self.ollama_comparer:
                                    try:
                                        comp = self.ollama_comparer.compare_images(
                                            r1['filepath'], r2['filepath'], params1, params2
                                        )
                                        meaningful_changes.append(comp.get('meaningful_change', False))
                                    except:
                                        pass
                    
                    if count > 0:
                        avg_diff /= count
                        has_meaningful = any(meaningful_changes) if meaningful_changes else True
                        differences.append((val1, val2, avg_diff, has_meaningful))
                        
                        method = "Ollama" if self.ollama_comparer else "pixel"
                        print(f"  {val1} -> {val2}: diff={avg_diff:.4f} ({method})" + 
                              (f", meaningful={has_meaningful}" if self.ollama_comparer else ""))
            
            # Find range with most variation (highest differences)
            if differences:
                # Sort by difference (highest first), but prefer meaningful changes if using Ollama
                if self.ollama_comparer:
                    # Prioritize ranges with meaningful changes
                    meaningful_diffs = [d for d in differences if len(d) > 3 and d[3]]
                    if meaningful_diffs:
                        meaningful_diffs.sort(key=lambda x: x[2], reverse=True)
                        best_range = meaningful_diffs[0]
                    else:
                        differences.sort(key=lambda x: x[2], reverse=True)
                        best_range = differences[0]
                else:
                    differences.sort(key=lambda x: x[2], reverse=True)
                    best_range = differences[0]
                
                param_ranges[param_name] = (best_range[0], best_range[1])
                print(f"  → Promising range: {best_range[0]} to {best_range[1]} (diff: {best_range[2]:.4f})")
            else:
                # Fallback to full range
                if isinstance(self.param_configs[param_name], dict):
                    param_ranges[param_name] = (
                        self.param_configs[param_name]['min'],
                        self.param_configs[param_name]['max']
                    )
                else:
                    param_ranges[param_name] = (param_values[0], param_values[-1])
                print(f"  → Using full range: {param_ranges[param_name]}")
        
        return param_ranges
    
    def run_coarse_scan(self):
        """Phase 1: Coarse scan of parameter space."""
        print("="*60)
        print("PHASE 1: COARSE SCAN")
        print("="*60)
        print("Testing parameter extremes and middle values...")
        print()
        
        # Generate coarse values for each parameter
        coarse_params = {}
        for param_name, param_config in self.param_configs.items():
            coarse_params[param_name] = self._generate_coarse_values(param_name, param_config)
            print(f"{param_name}: {coarse_params[param_name]}")
        
        # Generate all combinations
        param_names = list(coarse_params.keys())
        param_value_lists = [coarse_params[name] for name in param_names]
        combinations = list(product(*param_value_lists))
        total = len(combinations)
        
        print(f"\nTotal coarse combinations: {total}")
        print()
        
        for index, combination in enumerate(combinations, 1):
            params = dict(zip(param_names, combination))
            filepath = self._run_workflow(params, self.coarse_dir, index)
            
            if filepath:
                result = params.copy()
                result['filepath'] = filepath
                result['index'] = index
                self.coarse_results.append(result)
                print(f"    ✓ Saved")
            else:
                print(f"    ✗ Failed")
            print()
        
        # Save coarse results
        summary_path = self.coarse_dir / 'coarse_scan_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total': total,
                'successful': len(self.coarse_results),
                'results': self.coarse_results
            }, f, indent=2)
        
        print(f"Coarse scan complete: {len(self.coarse_results)}/{total} successful")
        print(f"Results saved to: {summary_path}")
    
    def run_fine_sweep(self, param_ranges: Dict[str, Tuple[Any, Any]]):
        """Phase 2: Fine sweep in promising ranges."""
        print("\n" + "="*60)
        print("PHASE 2: FINE SWEEP")
        print("="*60)
        print("Focusing on promising parameter ranges...")
        print()
        
        # Generate fine values for each parameter
        fine_params = {}
        for param_name, param_config in self.param_configs.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                fine_params[param_name] = self._generate_fine_values(
                    param_name, param_config, min_val, max_val
                )
            else:
                # Use original config
                if isinstance(param_config, list):
                    fine_params[param_name] = param_config
                else:
                    fine_params[param_name] = self._generate_fine_values(
                        param_name, param_config,
                        param_config['min'], param_config['max']
                    )
            print(f"{param_name}: {len(fine_params[param_name])} values")
        
        # Generate all combinations
        param_names = list(fine_params.keys())
        param_value_lists = [fine_params[name] for name in param_names]
        combinations = list(product(*param_value_lists))
        total = len(combinations)
        
        print(f"\nTotal fine combinations: {total}")
        print()
        
        for index, combination in enumerate(combinations, 1):
            params = dict(zip(param_names, combination))
            filepath = self._run_workflow(params, self.fine_dir, index)
            
            if filepath:
                result = params.copy()
                result['filepath'] = filepath
                result['index'] = index
                self.fine_results.append(result)
                print(f"    ✓ Saved")
            else:
                print(f"    ✗ Failed")
            print()
        
        # Save fine results
        summary_path = self.fine_dir / 'fine_sweep_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total': total,
                'successful': len(self.fine_results),
                'param_ranges': param_ranges,
                'results': self.fine_results
            }, f, indent=2)
        
        print(f"Fine sweep complete: {len(self.fine_results)}/{total} successful")
        print(f"Results saved to: {summary_path}")
    
    def run(self):
        """Run the complete two-phase sweep."""
        # Phase 1: Coarse scan
        self.run_coarse_scan()
        
        # Analyze results
        param_ranges = self._analyze_coarse_results()
        
        # Phase 2: Fine sweep
        if param_ranges:
            self.run_fine_sweep(param_ranges)
        else:
            print("\nSkipping fine sweep - no promising ranges found")
        
        print("\n" + "="*60)
        print("SMART SWEEP COMPLETE")
        print("="*60)
        print(f"Coarse scan: {len(self.coarse_results)} images")
        print(f"Fine sweep: {len(self.fine_results)} images")
        print(f"Total: {len(self.coarse_results) + len(self.fine_results)} images")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python smart_sweep.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        sweep = SmartSweep(config_path)
        sweep.run()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
