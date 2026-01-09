#!/usr/bin/env python3
"""
Test script that reads parameters from a config file.
Edit test_config.json to change parameters instead of editing code.
"""

import json
import sys
import time
import uuid
import websocket
import requests
import random
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_workflow(workflow_path: str) -> dict:
    """Load workflow JSON file."""
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_workflow_model(workflow: dict, ckpt_name: str, model_type: str = 'checkpoint') -> dict:
    """Update workflow checkpoint/model.
    
    Args:
        workflow: The workflow dictionary
        ckpt_name: Model filename
        model_type: 'checkpoint' (default) or 'diffusion'
    """
    workflow_copy = json.loads(json.dumps(workflow))
    
    if model_type == 'diffusion':
        # For diffusion models, try to find UNETLoader (loads from models/diffusion_models/)
        unet_id = None
        for node_id, node_data in workflow_copy.items():
            if isinstance(node_data, dict) and node_data.get('class_type') == 'UNETLoader':
                unet_id = node_id
                break
        
        if unet_id:
            workflow_copy[unet_id]['inputs']['unet_name'] = ckpt_name
            print(f"   Updated UNETLoader node {unet_id} with model: {ckpt_name}")
            return workflow_copy
        else:
            # CheckpointLoaderSimple won't work for diffusion_models directory
            print("❌ Error: Workflow doesn't have UNETLoader node required for diffusion models.")
            print("   Z-Image-Turbo and other diffusion models need a workflow with:")
            print("   - UNETLoader (for the diffusion model)")
            print("   - CLIPLoader (for text encoder)")
            print("   - VAELoader (for VAE)")
            print("   The current workflow uses CheckpointLoaderSimple which only looks in models/checkpoints/")
            print("   Try using: img2img_turbo_api.json workflow file")
            raise ValueError("Diffusion model requires UNETLoader workflow")
    
    # Find CheckpointLoaderSimple node (for checkpoint models)
    checkpoint_id = None
    for node_id, node_data in workflow_copy.items():
        if isinstance(node_data, dict) and node_data.get('class_type') == 'CheckpointLoaderSimple':
            checkpoint_id = node_id
            break
    
    if checkpoint_id:
        workflow_copy[checkpoint_id]['inputs']['ckpt_name'] = ckpt_name
    else:
        print("⚠️  Warning: No CheckpointLoaderSimple node found in workflow")
    
    return workflow_copy


def update_workflow_parameters(workflow: dict, **params) -> dict:
    """Update workflow parameters."""
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Find KSampler node
    ksampler_id = None
    for node_id, node_data in workflow_copy.items():
        if isinstance(node_data, dict) and node_data.get('class_type') == 'KSampler':
            ksampler_id = node_id
            break
    
    if not ksampler_id:
        raise ValueError("Could not find KSampler node in workflow")
    
    # Update parameters
    inputs = workflow_copy[ksampler_id]['inputs']
    if 'denoise' in params:
        inputs['denoise'] = params['denoise']
    if 'cfg' in params:
        inputs['cfg'] = params['cfg']
    if 'steps' in params:
        inputs['steps'] = params['steps']
    if 'seed' in params and params['seed'] is not None:
        inputs['seed'] = params['seed']
    if 'sampler_name' in params:
        inputs['sampler_name'] = params['sampler_name']
    if 'scheduler' in params:
        inputs['scheduler'] = params['scheduler']
    
    return workflow_copy


def update_workflow_prompts(workflow: dict, positive: str = None, negative: str = None) -> dict:
    """Update workflow prompts if overrides are enabled."""
    if not positive and not negative:
        return workflow
    
    workflow_copy = json.loads(json.dumps(workflow))
    
    # Find CLIPTextEncode nodes
    for node_id, node_data in workflow_copy.items():
        if isinstance(node_data, dict) and node_data.get('class_type') == 'CLIPTextEncode':
            inputs = node_data.get('inputs', {})
            meta_title = node_data.get('_meta', {}).get('title', '')
            
            # Node 9 is typically positive, node 10 is negative
            # Or check if it's connected to positive/negative inputs
            if node_id == "9" or (positive and 'negative' not in meta_title.lower()):
                if positive:
                    inputs['text'] = positive
            elif node_id == "10" or 'negative' in meta_title.lower():
                if negative:
                    inputs['text'] = negative
    
    return workflow_copy


def queue_prompt(prompt: dict, base_url: str) -> str:
    """Queue a prompt and return its ID."""
    data = {"prompt": prompt}
    response = requests.post(f"{base_url}/prompt", json=data)
    response.raise_for_status()
    return response.json()['prompt_id']


def get_history(prompt_id: str, base_url: str) -> dict:
    """Get execution history for a prompt."""
    response = requests.get(f"{base_url}/history/{prompt_id}")
    response.raise_for_status()
    return response.json()


def download_image(filename: str, subfolder: str, base_url: str, folder_type: str = "output") -> bytes:
    """Download an image from ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{base_url}/view", params=data)
    response.raise_for_status()
    return response.content


def extract_prompts(workflow: dict) -> tuple:
    """Extract positive and negative prompts from workflow."""
    positive_prompt = None
    negative_prompt = None
    
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict) and node_data.get('class_type') == 'CLIPTextEncode':
            inputs = node_data.get('inputs', {})
            text = inputs.get('text', '')
            meta_title = node_data.get('_meta', {}).get('title', '')
            
            if node_id == "9" or (not negative_prompt and positive_prompt is None):
                positive_prompt = text
            elif node_id == "10" or 'negative' in meta_title.lower():
                negative_prompt = text
    
    return positive_prompt, negative_prompt


def run_test(config_path: str):
    """Run a test with parameters from config file."""
    
    # Load config
    config = load_config(config_path)
    workflow_path = config['workflow_file']
    model_config = config.get('model', {})
    comfyui_config = config['comfyui']
    params_config = config['parameters']
    prompt_overrides = config.get('prompt_overrides', {})
    output_dir = config.get('output', {}).get('dir', 'test_results')
    
    base_url = f"http://{comfyui_config['host']}:{comfyui_config['port']}"
    ws_url = f"ws://{comfyui_config['host']}:{comfyui_config['port']}/ws?clientId={uuid.uuid4()}"
    
    print("="*60)
    print("Test Run (Config-Based)")
    print("="*60)
    
    # Extract parameter values
    denoise = params_config['denoise'].get('value') if isinstance(params_config['denoise'], dict) else params_config['denoise']
    cfg = params_config['cfg'].get('value') if isinstance(params_config['cfg'], dict) else params_config['cfg']
    steps = params_config['steps'].get('value') if isinstance(params_config['steps'], dict) else params_config['steps']
    seed = params_config.get('seed', {}).get('value') if isinstance(params_config.get('seed', {}), dict) else params_config.get('seed')
    sampler_name = params_config.get('sampler_name', {}).get('value') if isinstance(params_config.get('sampler_name', {}), dict) else params_config.get('sampler_name')
    scheduler = params_config.get('scheduler', {}).get('value') if isinstance(params_config.get('scheduler', {}), dict) else params_config.get('scheduler')
    
    # Use random seed if not specified
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    
    print(f"Parameters:")
    print(f"  denoise: {denoise}")
    print(f"  cfg: {cfg}")
    print(f"  steps: {steps}")
    print(f"  seed: {seed}")
    if sampler_name:
        print(f"  sampler_name: {sampler_name}")
    if scheduler:
        print(f"  scheduler: {scheduler}")
    print()
    
    # Load workflow
    print("Loading workflow...")
    workflow = load_workflow(workflow_path)
    
    # Update model if specified in config
    if model_config.get('ckpt_name'):
        ckpt_name = model_config['ckpt_name']
        model_type = model_config.get('type', 'checkpoint')  # Default to 'checkpoint' for backward compatibility
        print(f"Using model: {ckpt_name} (type: {model_type})")
        try:
            workflow = update_workflow_model(workflow, ckpt_name, model_type)
        except ValueError as e:
            print(f"\n❌ {e}")
            print("\nTo use Z-Image-Turbo, you need a workflow file that uses:")
            print("  - UNETLoader (loads from models/diffusion_models/)")
            print("  - CLIPLoader (for text encoder, e.g., qwen_3_4b.safetensors)")
            print("  - VAELoader (for VAE, e.g., ae.safetensors)")
            print("\nThe current workflow only supports checkpoint models.")
            return
    
    # Extract and display prompts
    positive_prompt, negative_prompt = extract_prompts(workflow)
    
    # Override prompts if enabled - read from files
    if prompt_overrides.get('enabled'):
        config_dir = Path(config_path).parent
        
        # Read positive prompt from file
        positive_file = prompt_overrides.get('positive_file', 'positive_prompt.txt')
        positive_path = config_dir / positive_file
        if positive_path.exists():
            with open(positive_path, 'r', encoding='utf-8') as f:
                # Read file and replace newlines with spaces, then strip
                positive_prompt = f.read().replace('\n', ' ').strip()
        else:
            print(f"⚠️  Warning: Positive prompt file not found: {positive_path}")
        
        # Read negative prompt from file
        negative_file = prompt_overrides.get('negative_file', 'negative_prompt.txt')
        negative_path = config_dir / negative_file
        if negative_path.exists():
            with open(negative_path, 'r', encoding='utf-8') as f:
                # Read file and replace newlines with spaces, then strip
                negative_prompt = f.read().replace('\n', ' ').strip()
        else:
            print(f"⚠️  Warning: Negative prompt file not found: {negative_path}")
        
        # Update workflow with prompts from files
        workflow = update_workflow_prompts(workflow, positive_prompt, negative_prompt)
    
    if positive_prompt or negative_prompt:
        print("Prompts:")
        if positive_prompt:
            print(f"  Positive: {positive_prompt}")
        if negative_prompt:
            print(f"  Negative: {negative_prompt}")
        print()
    
    # Update parameters
    print("Updating parameters...")
    params = {
        "denoise": denoise,
        "cfg": cfg,
        "steps": steps,
        "seed": seed
    }
    if sampler_name:
        params["sampler_name"] = sampler_name
    if scheduler:
        params["scheduler"] = scheduler
    
    updated_workflow = update_workflow_parameters(workflow, **params)
    
    # Queue prompt
    print("Queueing prompt...")
    try:
        prompt_id = queue_prompt(updated_workflow, base_url)
        print(f"Prompt ID: {prompt_id}")
    except Exception as e:
        print(f"❌ Error queueing prompt: {e}")
        return None
    
    # Connect via WebSocket to track progress
    print("Waiting for completion...")
    completed = False
    max_wait = 300
    start_time = time.time()
    ws_connected = False
    
    # Try WebSocket first
    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        ws.settimeout(1.0)
        ws_connected = True
        
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
                    
                    elif data.get('type') == 'progress':
                        progress_data = data.get('data', {})
                        value = progress_data.get('value', 0)
                        max_val = progress_data.get('max', 100)
                        if max_val > 0:
                            percent = (value / max_val) * 100
                            print(f"  Progress: {percent:.1f}%", end='\r')
                            
                            if percent >= 100.0:
                                print("\n  Progress: 100% - waiting for finalisation...")
                                time.sleep(2)
                                completed = True
                                break
            
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                print("\n  WebSocket disconnected, polling history instead...")
                ws_connected = False
                break
        
        if ws_connected:
            ws.close()
    
    except Exception as e:
        print(f"  WebSocket connection failed: {e}")
        print("  Falling back to polling history API...")
        ws_connected = False
    
    # Poll history if WebSocket didn't complete
    if not completed:
        print("  Polling for completion...")
        poll_interval = 1
        while time.time() - start_time < max_wait:
            try:
                history = get_history(prompt_id, base_url)
                if prompt_id in history:
                    execution = history[prompt_id]
                    status = execution.get('status', {})
                    outputs = execution.get('outputs', {})
                    if status.get('completed') or outputs:
                        completed = True
                        break
            except Exception:
                pass
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed > 10:
                print(f"\n  Been waiting {elapsed}s, checking if result is available...")
                try:
                    history = get_history(prompt_id, base_url)
                    if prompt_id in history:
                        execution = history[prompt_id]
                        outputs = execution.get('outputs', {})
                        if outputs:
                            completed = True
                            break
                except:
                    pass
                if elapsed > 15:
                    print("  Assuming completion after 15s...")
                    break
            else:
                print(f"  Waiting... ({elapsed}s)", end='\r')
    
    if not completed:
        print("\n⚠️  Didn't detect completion via WebSocket/history")
        print("  Attempting to fetch result anyway (may have completed)...")
    else:
        print("\n✓ Execution completed!")
    
    # Get result
    print("Fetching result...")
    image_info = None
    
    # Try history API first
    for attempt in range(8):
        try:
            time.sleep(1 if attempt > 0 else 0.5)
            history = get_history(prompt_id, base_url)
            
            if prompt_id in history:
                execution = history[prompt_id]
                outputs = execution.get('outputs', {})
                
                if outputs:
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output and len(node_output['images']) > 0:
                            image_info = node_output['images'][0]
                            break
                    
                    if image_info:
                        break
        except Exception:
            if attempt < 7:
                continue
    
    # Fallback: check output directory
    if not image_info:
        print("  History not available, using fallback: checking output directory...")
        try:
            output_dir_path = Path("/home/tclark/ComfyUI/output")
            if output_dir_path.exists():
                png_files = sorted(
                    output_dir_path.glob("ComfyUI_*.png"), 
                    key=lambda p: p.stat().st_mtime, 
                    reverse=True
                )
                if png_files:
                    most_recent = png_files[0]
                    filename = most_recent.name
                    image_info = {
                        'filename': filename,
                        'subfolder': '',
                        'type': 'output'
                    }
                    print(f"  Found most recent output: {filename}")
                else:
                    print("❌ No output files found in output directory")
                    return None
            else:
                print("❌ Output directory not found")
                return None
        except Exception as e:
            print(f"❌ Fallback failed: {e}")
            return None
    
    if not image_info:
        print("❌ No image found in outputs")
        return None
    
    # Download image
    print("Downloading image...")
    image_data = download_image(
        image_info['filename'],
        image_info.get('subfolder', ''),
        base_url,
        image_info.get('type', 'output')
    )
    
    # Save image
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    seed_str = f"_seed{seed}" if seed else ""
    sampler_str = f"_sampler{sampler_name}" if sampler_name else ""
    scheduler_str = f"_scheduler{scheduler}" if scheduler else ""
    filename = f"test_denoise_{denoise:.2f}_cfg_{cfg:.1f}_steps_{steps}{sampler_str}{scheduler_str}{seed_str}.png"
    filepath = output_path / filename
    
    with open(filepath, 'wb') as f:
        f.write(image_data)
    
    print(f"✓ Image saved to: {filepath}")
    return str(filepath)


if __name__ == '__main__':
    config_path = "/home/tclark/ComfyUI/ComfyScripts/test_config.json"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    result = run_test(config_path)
    
    if result:
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Test failed - check ComfyUI is running")
        print("="*60)
        sys.exit(1)
