#!/usr/bin/env python3
"""
Simple script to fetch the most recent result from ComfyUI.
"""

import json
import requests
import sys
from pathlib import Path
from datetime import datetime


def get_history(base_url: str, prompt_id: str = None):
    """Get history - either specific prompt or all recent."""
    if prompt_id:
        response = requests.get(f"{base_url}/history/{prompt_id}")
    else:
        # Get all history
        response = requests.get(f"{base_url}/history")
    response.raise_for_status()
    return response.json()


def download_image(filename: str, subfolder: str, base_url: str, folder_type: str = "output") -> bytes:
    """Download an image from ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{base_url}/view", params=data)
    response.raise_for_status()
    return response.content


def main():
    base_url = "http://localhost:8188"
    prompt_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("Fetching history...")
    try:
        if prompt_id:
            history = get_history(base_url, prompt_id)
            if prompt_id not in history:
                print(f"❌ Prompt {prompt_id} not found in history")
                return
            prompts = {prompt_id: history[prompt_id]}
        else:
            history = get_history(base_url)
            # Get most recent prompt
            if not history:
                print("❌ No history found")
                return
            # Sort by prompt ID (they're usually sequential/timestamped)
            prompts = dict(sorted(history.items(), reverse=True)[:1])
            prompt_id = list(prompts.keys())[0]
            print(f"Found most recent prompt: {prompt_id}")
    except Exception as e:
        print(f"❌ Error fetching history: {e}")
        return
    
    # Get outputs
    execution = prompts[prompt_id]
    outputs = execution.get('outputs', {})
    
    if not outputs:
        print("❌ No outputs found")
        return
    
    print(f"Found outputs for {len(outputs)} node(s)")
    
    # Find images
    images_found = []
    for node_id, node_output in outputs.items():
        if 'images' in node_output:
            for img in node_output['images']:
                images_found.append((node_id, img))
    
    if not images_found:
        print("❌ No images found in outputs")
        return
    
    print(f"Found {len(images_found)} image(s)")
    
    # Download all images
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, (node_id, img_info) in enumerate(images_found):
        filename = img_info['filename']
        subfolder = img_info.get('subfolder', '')
        folder_type = img_info.get('type', 'output')
        
        print(f"\nDownloading image {i+1}: {filename}...")
        try:
            image_data = download_image(filename, subfolder, base_url, folder_type)
            
            # Save with descriptive name
            save_filename = f"result_{prompt_id[:8]}_node{node_id}_{filename}"
            filepath = output_dir / save_filename
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            print(f"✓ Saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
