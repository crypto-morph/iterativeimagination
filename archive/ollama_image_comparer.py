#!/usr/bin/env python3
"""
Ollama Image Comparer

Uses Ollama vision models to semantically compare images and identify
which parameter combinations produce meaningful differences.
"""

import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io


class OllamaImageComparer:
    """Compare images using Ollama vision models."""
    
    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434"):
        """
        Initialise the comparer.
        
        Args:
            model: Ollama model name (default: "llava" for vision)
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _image_to_base64_pil(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def describe_image(self, image_path: str, params: Dict = None, max_size: int = 1024) -> Dict:
        """
        Describe a single image using Ollama vision model.
        
        Args:
            image_path: Path to image file
            params: Optional parameter dict for context
            max_size: Maximum dimension to resize image (reduces processing time)
        
        Returns:
            Dict with description including:
            - description: Detailed description of the scene
            - person_present: Whether a person is visible
            - clothing_status: Description of clothing/nudity
            - scene_elements: List of key elements in the scene
        """
        try:
            # Load and optionally resize image to speed up processing
            img = Image.open(image_path).convert('RGB')
            if max(img.size) > max_size:
                # Maintain aspect ratio
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_b64 = self._image_to_base64_pil(img)
            
            # Build prompt
            param_info = ""
            if params:
                param_info = f"\n\nImage parameters: {params}"
            
            prompt = f"""Describe this image in detail.

Focus on:
1. What is the main subject? (person, object, scene)
2. If there's a person: describe their appearance, pose, clothing/nudity status
3. What is the background/setting?
4. What is the overall scene composition?
5. Any notable details or features?

{param_info}

Provide your description in JSON format:
{{
    "description": "detailed description of the scene",
    "person_present": <true/false>,
    "person_description": "description of person if present",
    "clothing_status": "description of clothing/nudity",
    "background": "description of background/setting",
    "scene_elements": ["element1", "element2", ...],
    "overall_assessment": "brief overall assessment"
}}"""
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "format": "json"
            }
            
            # Make request (increased timeout for vision models - can take 2-3 minutes)
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '')
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (might have markdown code blocks)
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif '```' in response_text:
                    json_start = response_text.find('```') + 3
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                description = json.loads(response_text)
                
                # Ensure all fields exist
                description.setdefault('description', response_text)
                description.setdefault('person_present', False)
                description.setdefault('person_description', '')
                description.setdefault('clothing_status', '')
                description.setdefault('background', '')
                description.setdefault('scene_elements', [])
                description.setdefault('overall_assessment', '')
                
                return description
            
            except json.JSONDecodeError:
                # Fallback: return raw response as description
                return {
                    "description": response_text,
                    "person_present": False,
                    "person_description": "",
                    "clothing_status": "",
                    "background": "",
                    "scene_elements": [],
                    "overall_assessment": "",
                    "raw_response": response_text
                }
        
        except Exception as e:
            print(f"    Error describing image: {e}")
            return {
                "description": f"Error: {str(e)}",
                "person_present": False,
                "person_description": "",
                "clothing_status": "",
                "background": "",
                "scene_elements": [],
                "overall_assessment": "",
                "error": str(e)
            }
    
    def compare_images(self, image1_path: str, image2_path: str, 
                      params1: Dict = None, params2: Dict = None, max_size: int = 1024) -> Dict:
        """
        Compare two images using Ollama vision model.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            params1: Optional parameters for first image
            params2: Optional parameters for second image
            max_size: Maximum dimension to resize images (reduces processing time)
        
        Returns:
            Dict with comparison results including:
            - similarity_score: 0-1 (higher = more similar)
            - differences: List of differences found
            - meaningful_change: Boolean indicating if change is meaningful
        """
        try:
            # Load images and optionally resize to speed up processing
            img1 = Image.open(image1_path).convert('RGB')
            img2 = Image.open(image2_path).convert('RGB')
            
            # Resize if too large (maintains aspect ratio)
            for img in [img1, img2]:
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    if img == img1:
                        img1 = img.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        img2 = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Resize to same size if needed (for comparison)
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            img1_b64 = self._image_to_base64_pil(img1)
            img2_b64 = self._image_to_base64_pil(img2)
            
            # Build prompt
            param_info = ""
            if params1 and params2:
                param_info = f"\n\nImage 1 parameters: {params1}\nImage 2 parameters: {params2}"
            
            prompt = f"""Compare these two images and analyze the differences.

First, describe what you see in each image, then compare them.

Focus on:
1. Describe Image 1: What is the scene? Is there a person? What are they wearing?
2. Describe Image 2: What is the scene? Is there a person? What are they wearing?
3. Are these images of the same person/subject?
4. What visual differences do you see? (clothing, pose, background, features, etc.)
5. Rate similarity on a scale of 0-1 (1 = identical, 0 = completely different)
6. Is the difference meaningful or just minor variations?

{param_info}

Provide your analysis in JSON format:
{{
    "image1_description": "description of first image",
    "image2_description": "description of second image",
    "similarity_score": <0-1>,
    "same_person": <true/false>,
    "differences": ["difference1", "difference2", ...],
    "meaningful_change": <true/false>,
    "analysis": "brief explanation comparing the two"
}}"""
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img1_b64, img2_b64],
                "stream": False,
                "format": "json"
            }
            
            # Make request (increased timeout for vision models - can take 2-3 minutes)
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '')
            
            # Try to parse JSON from response
            try:
                # Extract JSON from response (might have markdown code blocks)
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif '```' in response_text:
                    json_start = response_text.find('```') + 3
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                comparison = json.loads(response_text)
                
                # Ensure all fields exist
                comparison.setdefault('image1_description', '')
                comparison.setdefault('image2_description', '')
                comparison.setdefault('similarity_score', 0.5)
                comparison.setdefault('same_person', True)
                comparison.setdefault('differences', [])
                comparison.setdefault('meaningful_change', False)
                comparison.setdefault('analysis', response_text)
                
                return comparison
            
            except json.JSONDecodeError:
                # Fallback: try to extract similarity score from text
                return {
                    "image1_description": "",
                    "image2_description": "",
                    "similarity_score": 0.5,
                    "same_person": True,
                    "differences": [],
                    "meaningful_change": True,
                    "analysis": response_text,
                    "raw_response": response_text
                }
        
        except Exception as e:
            print(f"    Error comparing images: {e}")
            return {
                "image1_description": "",
                "image2_description": "",
                "similarity_score": 0.5,
                "same_person": True,
                "differences": [],
                "meaningful_change": False,
                "error": str(e)
            }
    
    def analyze_parameter_impact(self, results: List[Dict], param_name: str) -> Dict:
        """
        Analyze how a parameter affects image output.
        
        Args:
            results: List of result dicts with 'filepath' and parameter values
            param_name: Name of parameter to analyze
        
        Returns:
            Dict with analysis including:
            - impact_score: How much this parameter affects output (0-1)
            - promising_ranges: Ranges where parameter has most impact
            - recommendations: Suggested values to test
        """
        if len(results) < 2:
            return {
                "impact_score": 0.0,
                "promising_ranges": [],
                "recommendations": []
            }
        
        # Group results by parameter value
        param_groups = {}
        for result in results:
            param_value = result.get(param_name)
            if param_value is not None:
                if param_value not in param_groups:
                    param_groups[param_value] = []
                param_groups[param_value].append(result)
        
        if len(param_groups) < 2:
            return {
                "impact_score": 0.0,
                "promising_ranges": [],
                "recommendations": []
            }
        
        # Compare adjacent parameter values
        sorted_values = sorted(param_groups.keys())
        comparisons = []
        
        for i in range(len(sorted_values) - 1):
            val1 = sorted_values[i]
            val2 = sorted_values[i + 1]
            
            # Compare images from these two parameter values
            group1 = param_groups[val1]
            group2 = param_groups[val2]
            
            # Sample a few comparisons
            similarities = []
            meaningful_changes = []
            
            for r1 in group1[:2]:  # Sample up to 2 from each group
                for r2 in group2[:2]:
                    if r1.get('filepath') and r2.get('filepath'):
                        comp = self.compare_images(
                            r1['filepath'],
                            r2['filepath'],
                            {param_name: val1},
                            {param_name: val2},
                            max_size=1024  # Resize to speed up processing
                        )
                        similarities.append(comp.get('similarity_score', 0.5))
                        meaningful_changes.append(comp.get('meaningful_change', False))
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                has_meaningful_change = any(meaningful_changes)
                
                comparisons.append({
                    'range': (val1, val2),
                    'similarity': avg_similarity,
                    'meaningful_change': has_meaningful_change,
                    'impact': 1.0 - avg_similarity  # Higher impact = lower similarity
                })
        
        # Find promising ranges (where impact is high)
        if not comparisons:
            return {
                "impact_score": 0.0,
                "promising_ranges": [],
                "recommendations": []
            }
        
        # Sort by impact (highest first)
        comparisons.sort(key=lambda x: x['impact'], reverse=True)
        
        # Calculate overall impact score
        avg_impact = sum(c['impact'] for c in comparisons) / len(comparisons)
        
        # Find promising ranges (impact > threshold)
        threshold = 0.2  # 20% difference threshold
        promising_ranges = [
            c['range'] for c in comparisons 
            if c['impact'] > threshold and c['meaningful_change']
        ]
        
        # Generate recommendations
        recommendations = []
        if promising_ranges:
            # Recommend testing values in promising ranges
            for min_val, max_val in promising_ranges[:3]:  # Top 3 ranges
                mid_val = (min_val + max_val) / 2
                recommendations.append({
                    'range': (min_val, max_val),
                    'suggested_values': [min_val, mid_val, max_val],
                    'reason': 'High impact range'
                })
        
        return {
            "impact_score": avg_impact,
            "promising_ranges": promising_ranges,
            "recommendations": recommendations,
            "comparisons": comparisons
        }
    
    def test_connection(self) -> bool:
        """Test if Ollama is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


if __name__ == '__main__':
    # Test the comparer
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ollama_image_comparer.py <image1> <image2>")
        sys.exit(1)
    
    comparer = OllamaImageComparer()
    
    if not comparer.test_connection():
        print("Error: Cannot connect to Ollama. Is it running?")
        print("Start Ollama: ollama serve")
        print("Pull vision model: ollama pull llava")
        sys.exit(1)
    
    result = comparer.compare_images(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
