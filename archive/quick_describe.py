#!/usr/bin/env python3
"""
Quick image description using Ollama - simplified version.
"""

import sys
from ollama_image_comparer import OllamaImageComparer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_describe.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    comparer = OllamaImageComparer(model="llava:7b")
    
    if not comparer.test_connection():
        print("❌ Cannot connect to Ollama!")
        sys.exit(1)
    
    print(f"Describing: {image_path}")
    print("Processing with Ollama...")
    
    result = comparer.describe_image(image_path, max_size=1024)
    
    print("\n" + "="*60)
    if result.get('description'):
        print(result['description'])
    elif result.get('raw_response'):
        print(result['raw_response'])
    else:
        print("No description available")
    print("="*60)
