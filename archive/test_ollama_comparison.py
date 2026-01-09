#!/usr/bin/env python3
"""
Quick test of Ollama image comparison using existing coarse scan images.
"""

import sys
import json
from pathlib import Path
from ollama_image_comparer import OllamaImageComparer

def main():
    # Check if Ollama is available
    comparer = OllamaImageComparer(model="llava:7b")
    
    if not comparer.test_connection():
        print("❌ Cannot connect to Ollama!")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        print("\nAnd you have a vision model installed:")
        print("  ollama pull llava")
        sys.exit(1)
    
    print("✓ Ollama connection successful!")
    print()
    
    # Find some test images from coarse scan
    coarse_dir = Path("smart_sweep_results/coarse_scan")
    
    if not coarse_dir.exists():
        print(f"❌ Coarse scan directory not found: {coarse_dir}")
        print("Run the smart sweep first to generate test images.")
        sys.exit(1)
    
    # Get a few images to compare
    images = list(coarse_dir.glob("*.png"))
    
    if len(images) < 2:
        print(f"❌ Need at least 2 images to compare, found {len(images)}")
        sys.exit(1)
    
    print(f"Found {len(images)} images in coarse scan")
    print()
    
    # Test 1: Compare two images with different denoise values (same other params)
    print("="*60)
    print("TEST 1: Comparing images with different denoise values")
    print("="*60)
    
    # Find images with same cfg/steps/sampler but different denoise
    denoise_images = {}
    for img in images:
        parts = img.stem.split('_')
        # Extract denoise value
        try:
            denoise_idx = parts.index('denoise')
            denoise_val = float(parts[denoise_idx + 1])
            
            # Create key from other params
            key_parts = []
            for i, part in enumerate(parts):
                if part in ['cfg', 'steps', 'sampler_name', 'scheduler']:
                    key_parts.append(f"{part}_{parts[i+1]}")
            
            key = "_".join(key_parts)
            
            if key not in denoise_images:
                denoise_images[key] = []
            denoise_images[key].append((denoise_val, img))
        except (ValueError, IndexError):
            continue
    
    # Find a key with multiple denoise values
    test_key = None
    for key, imgs in denoise_images.items():
        if len(imgs) >= 2:
            test_key = key
            break
    
    if test_key:
        imgs = sorted(denoise_images[test_key], key=lambda x: x[0])
        img1_path = str(imgs[0][1])
        img2_path = str(imgs[1][1])
        denoise1 = imgs[0][0]
        denoise2 = imgs[1][0]
        
        print(f"Image 1: {imgs[0][1].name} (denoise={denoise1})")
        print(f"Image 2: {imgs[1][1].name} (denoise={denoise2})")
        print()
        
        print("Comparing with Ollama...")
        result = comparer.compare_images(
            img1_path,
            img2_path,
            {"denoise": denoise1},
            {"denoise": denoise2}
        )
        
        print("\n" + "="*60)
        print("COMPARISON RESULT:")
        print("="*60)
        print(json.dumps(result, indent=2))
        print()
        
        if 'similarity_score' in result:
            similarity = result['similarity_score']
            print(f"Similarity Score: {similarity:.3f} (1.0 = identical, 0.0 = completely different)")
            print(f"Meaningful Change: {result.get('meaningful_change', 'unknown')}")
            print(f"Same Person: {result.get('same_person', 'unknown')}")
        
        if 'differences' in result and result['differences']:
            print(f"\nDifferences detected:")
            for diff in result['differences']:
                print(f"  - {diff}")
        
        if 'analysis' in result:
            print(f"\nAnalysis:")
            print(f"  {result['analysis']}")
    else:
        print("Could not find images with same params but different denoise")
        # Fallback: just compare first two images
        img1_path = str(images[0])
        img2_path = str(images[1])
        
        print(f"Image 1: {images[0].name}")
        print(f"Image 2: {images[1].name}")
        print()
        
        print("Comparing with Ollama...")
        result = comparer.compare_images(img1_path, img2_path)
        
        print("\n" + "="*60)
        print("COMPARISON RESULT:")
        print("="*60)
        print(json.dumps(result, indent=2))
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
