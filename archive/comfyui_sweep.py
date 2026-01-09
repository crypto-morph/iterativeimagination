#!/usr/bin/env python3
"""
ComfyUI Parameter Sweep Script

Automates parameter sweeps for ComfyUI workflows by varying denoise strength
and CFG scale values, submitting jobs via the ComfyUI API.
"""

import json
import os
import random
import sys
import time
import uuid
import websocket
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from itertools import product


class ComfyUISweep:
    """Main class for running parameter sweeps on ComfyUI workflows."""
    
    def __init__(self, config_path: str):
        """Initialise the sweep with configuration."""
        self.config = self._load_config(config_path)
        workflow_path = self.config['workflow_file']
        
        # Check if it's API format (dict with numeric string keys) or workflow format (has 'nodes' key)
        workflow_data = self._load_workflow(workflow_path)
        if 'nodes' in workflow_data:
            # It's workflow format - need to convert or use API format
            # For now, assume user will export to API format
            raise ValueError(
                "Workflow file appears to be in workflow format, not API format.\n"
                "Please export it to API format:\n"
                "1. Open ComfyUI: http://localhost:8188\n"
                "2. Load the workflow\n"
                "3. Click File -> Export (API)\n"
                "4. Save it and update workflow_file path in config"
            )
        
        self.prompt = workflow_data  # API format uses 'prompt' terminology
        self.output_dir = Path(self.config.get('output_dir', 'sweep_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ComfyUI connection details
        self.host = self.config['comfyui']['host']
        self.port = self.config['comfyui']['port']
        self.base_url = f"http://{self.host}:{self.port}"
        self.ws_url = f"ws://{self.host}:{self.port}/ws?clientId={uuid.uuid4()}"
        
        # Parameter ranges - support multiple parameter types
        self.parameters = {}
        param_config = self.config.get('parameters', {})
        
        # Standard parameters
        if 'denoise' in param_config:
            self.parameters['denoise'] = self._generate_values(param_config['denoise'])
        if 'cfg_scale' in param_config or 'cfg' in param_config:
            key = 'cfg_scale' if 'cfg_scale' in param_config else 'cfg'
            self.parameters['cfg'] = self._generate_values(param_config[key])
        
        # Additional parameters
        if 'steps' in param_config:
            self.parameters['steps'] = self._generate_values(param_config['steps'], int_values=True)
        if 'sampler_name' in param_config:
            self.parameters['sampler_name'] = param_config['sampler_name'] if isinstance(param_config['sampler_name'], list) else [param_config['sampler_name']]
        if 'scheduler' in param_config:
            self.parameters['scheduler'] = param_config['scheduler'] if isinstance(param_config['scheduler'], list) else [param_config['scheduler']]
        if 'seed' in param_config:
            self.parameters['seed'] = self._generate_values(param_config['seed'], int_values=True)
        
        # Track results
        self.results = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Load workflow JSON file (API format)."""
        with open(workflow_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_values(self, param_config: Union[Dict[str, Any], List], int_values: bool = False) -> List:
        """Generate list of values from config (range or list)."""
        if isinstance(param_config, list):
            return param_config
        
        if 'min' in param_config and 'max' in param_config:
            min_val = param_config['min']
            max_val = param_config['max']
            step = param_config.get('step', 1.0)
            values = []
            current = min_val
            while current <= max_val:
                if int_values:
                    values.append(int(round(current)))
                else:
                    values.append(round(current, 6))
                current += step
            return values
        
        raise ValueError(f"Invalid parameter configuration: {param_config}")
    
    def _find_parameter_nodes(self) -> Dict[str, Optional[str]]:
        """Find nodes containing various parameters in API format."""
        nodes = {}
        
        # API format: {node_id: {class_type: ..., inputs: {...}}}
        for node_id, node_data in self.prompt.items():
            if not isinstance(node_data, dict):
                continue
            
            node_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            # Check for KSampler node (contains most sampling parameters)
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
                break
        
        return nodes
    
    def _update_prompt_parameters(self, **kwargs) -> Dict[str, Any]:
        """Create a copy of the prompt with updated parameters."""
        prompt_copy = json.loads(json.dumps(self.prompt))
        
        param_nodes = self._find_parameter_nodes()
        
        # Update each parameter
        for param_name, param_value in kwargs.items():
            node_id = param_nodes.get(param_name)
            if node_id and node_id in prompt_copy:
                if 'inputs' in prompt_copy[node_id]:
                    inputs = prompt_copy[node_id]['inputs']
                    
                    # Handle different parameter names
                    if param_name == 'cfg':
                        if 'cfg' in inputs:
                            inputs['cfg'] = param_value
                        elif 'cfg_scale' in inputs:
                            inputs['cfg_scale'] = param_value
                        elif 'guidance_scale' in inputs:
                            inputs['guidance_scale'] = param_value
                    else:
                        # Direct parameter name match
                        if param_name in inputs:
                            inputs[param_name] = param_value
        
        # Always vary the seed to ensure different results (unless explicitly set)
        # This prevents identical-looking images when only CFG/denoise change
        if 'seed' not in kwargs and 'seed' in param_nodes:
            seed_node_id = param_nodes['seed']
            if seed_node_id and seed_node_id in prompt_copy:
                if 'inputs' in prompt_copy[seed_node_id]:
                    # Generate a new random seed for each run
                    prompt_copy[seed_node_id]['inputs']['seed'] = random.randint(0, 2**32 - 1)
        
        return prompt_copy
    
    def _queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt via ComfyUI REST API."""
        # Prompt is already in API format: {node_id: {class_type: ..., inputs: ...}}
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
    
    def _save_result(self, image_data: bytes, params: Dict[str, Any], index: int) -> str:
        """Save result image with descriptive filename."""
        # Build filename from parameters
        parts = []
        for key in ['denoise', 'cfg', 'steps', 'sampler_name', 'scheduler', 'seed']:
            if key in params:
                value = params[key]
                if isinstance(value, float):
                    parts.append(f"{key}_{value:.2f}")
                elif isinstance(value, int):
                    parts.append(f"{key}_{value}")
                else:
                    # String value - use first few chars
                    parts.append(f"{key}_{str(value)[:8]}")
        
        filename = "_".join(parts) + f"_{index:03d}.png"
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return str(filepath)
    
    def _run_workflow(self, **params) -> Optional[bytes]:
        """Run a single workflow with given parameters and return image data."""
        # Format parameter display
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"Running: {param_str}")
        
        # Update prompt with parameters
        updated_prompt = self._update_prompt_parameters(**params)
        
        # Debug: Verify parameters were updated (only for first run)
        if len(self.results) == 0:
            param_nodes = self._find_parameter_nodes()
            ksampler_id = param_nodes.get('denoise') or param_nodes.get('cfg')
            if ksampler_id and ksampler_id in updated_prompt:
                inputs = updated_prompt[ksampler_id].get('inputs', {})
                print(f"  Debug - KSampler inputs: cfg={inputs.get('cfg')}, denoise={inputs.get('denoise')}, seed={inputs.get('seed')}")
        
        # Queue the prompt
        try:
            prompt_id = self._queue_prompt(updated_prompt)
        except Exception as e:
            print(f"Error queueing prompt: {e}")
            return None
        
        # Connect via WebSocket to track progress
        ws = websocket.WebSocket()
        try:
            ws.connect(self.ws_url)
            
            # Wait for completion
            max_wait = 300  # 5 minutes timeout
            start_time = time.time()
            completed = False
            image_info = None
            
            # Set a timeout on the websocket to avoid blocking indefinitely
            ws.settimeout(1.0)  # 1 second timeout for recv()
            
            while time.time() - start_time < max_wait:
                try:
                    message = ws.recv()
                    if message:
                        data = json.loads(message)
                        
                        # Check for execution progress
                        if data.get('type') == 'execution_cached':
                            cached_data = data.get('data', {})
                            if cached_data.get('prompt_id') == prompt_id:
                                cached_nodes = cached_data.get('nodes', [])
                                print(f"  ⚠️  CACHED execution for prompt {prompt_id} (nodes: {cached_nodes})")
                                print(f"     This means ComfyUI reused previous results - images may be identical!")
                        
                        elif data.get('type') == 'executing':
                            executing_data = data.get('data', {})
                            # Check if this is for our prompt
                            if executing_data.get('prompt_id') == prompt_id:
                                if executing_data.get('node') is None:
                                    # Execution finished - now fetch history to get image info
                                    completed = True
                                    break
                        
                        elif data.get('type') == 'progress':
                            progress_data = data.get('data', {})
                            value = progress_data.get('value', 0)
                            max_val = progress_data.get('max', 100)
                            if max_val > 0:
                                percent = (value / max_val) * 100
                                print(f"  Progress: {percent:.1f}%", end='\r')
                        
                        elif data.get('type') == 'executed':
                            # Sometimes executed messages come, but we'll use history API instead
                            pass
                
                except websocket.WebSocketTimeoutException:
                    # Timeout is normal - check if execution completed via history
                    # Poll history every few seconds to see if it's done
                    elapsed = time.time() - start_time
                    if elapsed > 5 and int(elapsed) % 3 == 0:  # Check every 3 seconds after 5 seconds
                        try:
                            history = self._get_history(prompt_id)
                            if prompt_id in history:
                                prompt_history = history[prompt_id]
                                status = prompt_history.get('status', {})
                                if status.get('completed'):
                                    completed = True
                                    break
                        except:
                            pass  # History might not be available yet
                    continue
                except Exception as e:
                    print(f"  WebSocket error: {e}")
                    break
            
            ws.close()
            
            if not completed:
                print(f"  Timeout waiting for completion")
                return None
            
            # Small delay to ensure history is written
            time.sleep(0.5)
            
            # Fetch history to get image information
            # Try a few times in case history isn't immediately available
            image_info = None
            for attempt in range(5):
                try:
                    history = self._get_history(prompt_id)
                    if prompt_id not in history:
                        if attempt < 4:
                            time.sleep(0.5)
                            continue
                        print(f"  Prompt ID not found in history after {attempt + 1} attempts")
                        return None
                    
                    prompt_history = history[prompt_id]
                    outputs = prompt_history.get('outputs', {})
                    
                    # Find SaveImage node output (or any node with images)
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output:
                            images = node_output['images']
                            if len(images) > 0:
                                # Get the first image
                                image = images[0]
                                filename = image['filename']
                                subfolder = image.get('subfolder', '')
                                folder_type = image.get('type', 'output')
                                image_info = (filename, subfolder, folder_type)
                                break
                    
                    if image_info:
                        break
                    elif attempt < 4:
                        time.sleep(0.5)
                        continue
                        
                except Exception as e:
                    if attempt < 4:
                        time.sleep(0.5)
                        continue
                    print(f"  Error fetching history: {e}")
                    return None
            
            if not image_info:
                print(f"  No image output found in history (checked {attempt + 1} times)")
                # Debug: show what we found
                try:
                    history = self._get_history(prompt_id)
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        print(f"  Debug: Found outputs for nodes: {list(outputs.keys())}")
                        for node_id, node_output in outputs.items():
                            print(f"    Node {node_id}: {list(node_output.keys())}")
                except:
                    pass
                return None
            
            # Download the image
            filename, subfolder, folder_type = image_info
            try:
                image_data = self._get_image(filename, subfolder, folder_type)
            except Exception as e:
                print(f"  Error downloading image: {e}")
                return None
            
            return image_data
        
        except Exception as e:
            print(f"  Error connecting to WebSocket: {e}")
            return None
        finally:
            if ws.connected:
                ws.close()
    
    def run_sweep(self):
        """Run the complete parameter sweep."""
        print(f"Starting parameter sweep...")
        for param_name, param_values in self.parameters.items():
            print(f"{param_name}: {param_values}")
        
        # Generate all combinations
        param_names = list(self.parameters.keys())
        param_value_lists = [self.parameters[name] for name in param_names]
        combinations = list(product(*param_value_lists))
        total = len(combinations)
        
        print(f"Total combinations: {total}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        for index, combination in enumerate(combinations, 1):
            print(f"[{index}/{total}] ", end="")
            
            # Create parameter dict
            params = dict(zip(param_names, combination))
            
            image_data = self._run_workflow(**params)
            
            if image_data:
                filepath = self._save_result(image_data, params, index)
                result = params.copy()
                result['filepath'] = filepath
                result['index'] = index
                self.results.append(result)
                print(f"  Saved: {filepath}")
            else:
                print(f"  Failed to generate image")
            
            print()
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate a summary report of the sweep."""
        summary_path = self.output_dir / 'sweep_summary.json'
        
        # Calculate total combinations
        total_combinations = 1
        for values in self.parameters.values():
            total_combinations *= len(values)
        
        summary = {
            'total_combinations': total_combinations,
            'successful': len(self.results),
            'failed': total_combinations - len(self.results),
            'parameters': {name: values for name, values in self.parameters.items()},
            'results': self.results
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSweep complete!")
        print(f"Successful: {summary['successful']}/{summary['total_combinations']}")
        print(f"Summary saved to: {summary_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python comfyui_sweep.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        sweep = ComfyUISweep(config_path)
        sweep.run_sweep()
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

