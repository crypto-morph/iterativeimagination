#!/usr/bin/env python3
"""
List available models and update config with selected model.
"""

import json
import sys
from pathlib import Path


def find_models_dir(base_name: str):
    """Find a ComfyUI models directory (checkpoints, diffusion_models, etc.)."""
    # Common locations
    possible_paths = [
        Path(f"/home/tclark/ComfyUI/models/{base_name}"),
        Path(f"models/{base_name}"),
        Path(f"../models/{base_name}"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Try to find it relative to script location
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models" / base_name
    if models_dir.exists():
        return models_dir
    
    return None


def list_models():
    """List all available models from checkpoints and diffusion_models directories."""
    checkpoints_dir = find_models_dir("checkpoints")
    diffusion_models_dir = find_models_dir("diffusion_models")
    
    all_models = []
    
    # Check checkpoints directory
    if checkpoints_dir:
        for ext in ['*.safetensors', '*.ckpt']:
            for model_file in checkpoints_dir.glob(ext):
                all_models.append({
                    'path': model_file,
                    'name': model_file.name,
                    'type': 'checkpoint',
                    'dir': 'checkpoints'
                })
    
    # Check diffusion_models directory
    if diffusion_models_dir:
        for ext in ['*.safetensors', '*.ckpt']:
            for model_file in diffusion_models_dir.glob(ext):
                all_models.append({
                    'path': model_file,
                    'name': model_file.name,
                    'type': 'diffusion',
                    'dir': 'diffusion_models'
                })
    
    if not all_models:
        print("❌ No model files found")
        if not checkpoints_dir and not diffusion_models_dir:
            print("   Could not find ComfyUI models directories")
        return None, None
    
    # Sort by name
    all_models.sort(key=lambda x: x['name'].lower())
    
    print(f"\nFound {len(all_models)} model(s):\n")
    
    for i, model_info in enumerate(all_models, 1):
        size_gb = model_info['path'].stat().st_size / (1024 * 1024 * 1024)
        type_label = "📦 checkpoint" if model_info['type'] == 'checkpoint' else "⚡ diffusion"
        print(f"  {i:2d}. {type_label} {model_info['name']} ({size_gb:.2f} GB)")
    
    return all_models, None


def select_model():
    """Interactive model selection."""
    all_models, _ = list_models()
    
    if not all_models:
        return None, None
    
    print()
    while True:
        try:
            choice = input(f"Select model (1-{len(all_models)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None, None
            
            index = int(choice) - 1
            if 0 <= index < len(all_models):
                selected = all_models[index]
                print(f"\n✓ Selected: {selected['name']} ({selected['type']} model)")
                return selected['name'], selected['type']
            else:
                print(f"❌ Please enter a number between 1 and {len(all_models)}")
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nCancelled")
            return None, None


def update_config(config_path: str, model_name: str, model_type: str):
    """Update config file with selected model."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'model' not in config:
            config['model'] = {}
        
        config['model']['ckpt_name'] = model_name
        config['model']['type'] = model_type  # 'checkpoint' or 'diffusion'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Updated {config_path} with model: {model_name} (type: {model_type})")
        return True
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False


if __name__ == '__main__':
    config_path = "/home/tclark/ComfyUI/ComfyScripts/test_config.json"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print("="*60)
    print("Model Selector")
    print("="*60)
    
    selected_model, model_type = select_model()
    
    if selected_model:
        update_config(config_path, selected_model, model_type)
        print("\n" + "="*60)
        if model_type == 'diffusion':
            print("⚠️  Note: Diffusion models (like z_image_turbo) may need a different workflow file.")
            print("   The current workflow uses CheckpointLoaderSimple which may not work.")
            print("   You may need to create a workflow using DiffusionModelLoader instead.")
        print("Config updated! Run test_with_config.py to use the new model.")
        print("="*60)
    else:
        print("\nNo model selected.")
