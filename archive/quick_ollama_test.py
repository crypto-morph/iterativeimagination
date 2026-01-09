#!/usr/bin/env python3
"""
Quick Ollama comparison test - compare two specific images.
Usage: python quick_ollama_test.py <image1> <image2>
"""

import sys
import json
from ollama_image_comparer import OllamaImageComparer

def main():
    if len(sys.argv) < 3:
        print("Usage: python quick_ollama_test.py <image1> <image2>")
        print("\nExample:")
        print("  python quick_ollama_test.py smart_sweep_results/coarse_scan/denoise_0.3_cfg_5.0_steps_20_sampler_name_heun_scheduler_normal_002.png \\")
        print("                              smart_sweep_results/coarse_scan/denoise_0.5_cfg_5.0_steps_20_sampler_name_heun_scheduler_normal_002.png")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
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
    print(f"Comparing:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print()
    print("Sending to Ollama... (this may take 30-90 seconds for vision models)")
    print("Images will be resized to max 1024px to speed up processing")
    print()
    
    result = comparer.compare_images(img1_path, img2_path, max_size=1024)
    
    print("="*60)
    print("COMPARISON RESULT:")
    print("="*60)
    print()
    
    # Show descriptions if available
    if result.get('image1_description'):
        print("📷 Image 1 Description:")
        print(f"   {result['image1_description']}")
        print()
    
    if result.get('image2_description'):
        print("📷 Image 2 Description:")
        print(f"   {result['image2_description']}")
        print()
    
    if 'similarity_score' in result:
        similarity = result['similarity_score']
        print(f"📊 Similarity Score: {similarity:.3f}")
        print(f"   (1.0 = identical, 0.0 = completely different)")
        print()
        print(f"👤 Same Person: {result.get('same_person', 'unknown')}")
        print(f"✨ Meaningful Change: {result.get('meaningful_change', 'unknown')}")
        print()
    
    if 'differences' in result and result['differences']:
        print("🔍 Differences detected:")
        for diff in result['differences']:
            print(f"   • {diff}")
        print()
    
    if 'analysis' in result:
        print("💭 Analysis:")
        print(f"   {result['analysis']}")
        print()
    
    print("="*60)
    print("Full JSON:")
    print("="*60)
    print(json.dumps(result, indent=2))
    print()

if __name__ == '__main__':
    main()
