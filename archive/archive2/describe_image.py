#!/usr/bin/env python3
"""
Describe a single image using Ollama vision model.
Usage: python describe_image.py <image_path>
"""

import sys
import json
from pathlib import Path

# Add archive directory to path so we can import from there
sys.path.insert(0, str(Path(__file__).parent / "archive"))
from ollama_image_comparer import OllamaImageComparer

def main():
    if len(sys.argv) < 2:
        print("Usage: python describe_image.py <image_path>")
        print("\nExample:")
        print("  python describe_image.py smart_sweep_results/coarse_scan/denoise_0.3_cfg_5.0_steps_20_sampler_name_heun_scheduler_normal_002.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if Ollama is available
    comparer = OllamaImageComparer(model="llava:7b")
    
    if not comparer.test_connection():
        print("❌ Cannot connect to Ollama!")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        sys.exit(1)
    
    print("✓ Ollama connection successful!")
    print()
    print(f"Describing image: {image_path}")
    print("Processing... (this may take 30-90 seconds for vision models)")
    print()
    
    result = comparer.describe_image(image_path, max_size=1024)
    
    print("="*60)
    print("IMAGE DESCRIPTION:")
    print("="*60)
    print()
    
    if 'description' in result:
        print("📝 Description:")
        print(f"   {result['description']}")
        print()
    
    if result.get('person_present'):
        print("👤 Person Present: Yes")
        if result.get('person_description'):
            print(f"   {result['person_description']}")
        print()
        
        if result.get('clothing_status'):
            print("👕 Clothing Status:")
            print(f"   {result['clothing_status']}")
            print()
    else:
        print("👤 Person Present: No")
        print()
    
    if result.get('background'):
        print("🖼️  Background:")
        print(f"   {result['background']}")
        print()
    
    if result.get('scene_elements'):
        print("🔍 Scene Elements:")
        for element in result['scene_elements']:
            print(f"   • {element}")
        print()
    
    if result.get('overall_assessment'):
        print("💭 Overall Assessment:")
        print(f"   {result['overall_assessment']}")
        print()
    
    print("="*60)
    print("Full JSON:")
    print("="*60)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
