#!/usr/bin/env python3
"""
One-off test script to test workflow with specific parameters.
"""

import json
import sys
import time
import uuid
import websocket
import requests
from pathlib import Path


def load_workflow(workflow_path: str) -> dict:
    """Load workflow JSON file."""
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    if 'seed' in params:
        inputs['seed'] = params['seed']
    
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


def run_test(workflow_path: str, denoise: float = 0.35, cfg: float = 9.0, 
             steps: int = 30, seed: int = None, host: str = "localhost", 
             port: int = 8188, output_dir: str = "test_results"):
    """Run a one-off test with specified parameters."""
    
    base_url = f"http://{host}:{port}"
    ws_url = f"ws://{host}:{port}/ws?clientId={uuid.uuid4()}"
    
    print("="*60)
    print("One-Off Test Run")
    print("="*60)
    print(f"Parameters:")
    print(f"  denoise: {denoise}")
    print(f"  cfg: {cfg}")
    print(f"  steps: {steps}")
    if seed:
        print(f"  seed: {seed}")
    print()
    
    # Load workflow
    print("Loading workflow...")
    workflow = load_workflow(workflow_path)
    
    # Extract and display prompts
    positive_prompt = None
    negative_prompt = None
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict):
            node_type = node_data.get('class_type', '')
            if node_type == 'CLIPTextEncode':
                inputs = node_data.get('inputs', {})
                text = inputs.get('text', '')
                # Check _meta title to identify positive vs negative
                meta_title = node_data.get('_meta', {}).get('title', '')
                if 'negative' in meta_title.lower() or '(negative)' in meta_title.lower():
                    negative_prompt = text
                else:
                    # Positive prompt (or check if it's connected to KSampler positive input)
                    # In the workflow, node 9 is positive, node 10 is negative
                    if node_id == "9" or (not negative_prompt and positive_prompt is None):
                        positive_prompt = text
                    elif node_id == "10":
                        negative_prompt = text
    
    if positive_prompt or negative_prompt:
        print("Prompts:")
        if positive_prompt:
            print(f"  Positive: {positive_prompt}")
        if negative_prompt:
            print(f"  Negative: {negative_prompt}")
        print()
    
    # Update parameters
    print("Updating parameters...")
    params = {"denoise": denoise, "cfg": cfg, "steps": steps}
    if seed:
        params["seed"] = seed
    updated_workflow = update_workflow_parameters(workflow, **params)
    
    # Queue prompt
    print("Queueing prompt...")
    try:
        prompt_id = queue_prompt(updated_workflow, base_url)
        print(f"Prompt ID: {prompt_id}")
    except Exception as e:
        print(f"❌ Error queueing prompt: {e}")
        return None
    
    # Connect via WebSocket to track progress (with fallback to polling)
    print("Waiting for completion...")
    completed = False
    max_wait = 300  # 5 minutes timeout
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
                            
                            # If we hit 100%, wait a moment then proceed to fetch result
                            if percent >= 100.0:
                                print("\n  Progress: 100% - waiting for finalisation...")
                                time.sleep(2)  # Give ComfyUI a moment to save
                                # Mark as completed and break out to fetch result
                                completed = True
                                break
            
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                # WebSocket disconnected, fall back to polling
                print("\n  WebSocket disconnected, polling history instead...")
                ws_connected = False
                break
        
        if ws_connected:
            ws.close()
    
    except Exception as e:
        print(f"  WebSocket connection failed: {e}")
        print("  Falling back to polling history API...")
        ws_connected = False
    
    # If WebSocket didn't complete, poll history API
    if not completed:
        print("  Polling for completion...")
        poll_interval = 1  # Check every 1 second (more aggressive)
        last_check = time.time()
        while time.time() - start_time < max_wait:
            try:
                history = get_history(prompt_id, base_url)
                if prompt_id in history:
                    execution = history[prompt_id]
                    # Check if completed via status or outputs
                    status = execution.get('status', {})
                    outputs = execution.get('outputs', {})
                    if status.get('completed') or outputs:
                        completed = True
                        break
            except Exception as e:
                # History might not be available yet - but if we've been waiting a while, try anyway
                if time.time() - start_time > 30:
                    print(f"\n  Warning: History API error: {e}")
            
            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            # After 10 seconds, assume it's done and try to fetch anyway
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
                # If still no result after 15 seconds, break and try to fetch anyway
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
    
    # Get result - try multiple times with delays
    print("Fetching result...")
    image_info = None
    
    # First, try to get from history API
    for attempt in range(8):
        try:
            time.sleep(1 if attempt > 0 else 0.5)
            history = get_history(prompt_id, base_url)
            
            if prompt_id in history:
                execution = history[prompt_id]
                outputs = execution.get('outputs', {})
                
                if outputs:
                    # Find the SaveImage node output
                    for node_id, node_output in outputs.items():
                        if 'images' in node_output and len(node_output['images']) > 0:
                            image_info = node_output['images'][0]
                            break
                    
                    if image_info:
                        break
        except Exception:
            if attempt < 7:
                continue
    
    # Fallback: Get most recent image from output directory if history fails
    if not image_info:
        print("  History not available, using fallback: checking output directory...")
        try:
            output_dir_path = Path("/home/tclark/ComfyUI/output")
            if output_dir_path.exists():
                # Get most recent PNG file (should be our result)
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
    filename = f"test_denoise_{denoise:.2f}_cfg_{cfg:.1f}_steps_{steps}{seed_str}.png"
    filepath = output_path / filename
    
    with open(filepath, 'wb') as f:
        f.write(image_data)
    
    print(f"✓ Image saved to: {filepath}")
    return str(filepath)


if __name__ == '__main__':
    workflow_path = "/home/tclark/ComfyUI/ComfyScripts/img2img_no_mask_api.json"
    
    # Recommended values for "identical but naked" - Very low denoise to preserve identity
    denoise = 0.35  # Very low to preserve person's identity - img2img may not be ideal for this
    cfg = 9.0       # Moderate CFG
    steps = 30
    
    # Use a different seed each time to avoid cache (or use random)
    import random
    seed = random.randint(1, 2**31 - 1)  # Random seed to force new generation
    # Or use a specific seed: seed = 123456789
    
    result = run_test(
        workflow_path=workflow_path,
        denoise=denoise,
        cfg=cfg,
        steps=steps,
        seed=seed,
        host="localhost",
        port=8188
    )
    
    if result:
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Test failed - check ComfyUI is running")
        print("="*60)
        sys.exit(1)
