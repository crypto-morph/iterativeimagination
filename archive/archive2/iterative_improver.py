#!/usr/bin/env python3
"""
Iterative image improvement system with feedback loop.
Uses Ollama Vision to evaluate images and suggest improvements.
"""

import json
import sys
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import re

# Add archive to path for imports
sys.path.insert(0, str(Path(__file__).parent / "archive"))
from ollama_image_comparer import OllamaImageComparer


class IterativeImprover:
    """Iterative improvement system with Ollama feedback."""
    
    def __init__(self, config_path: str, original_image_path: str, rules_path: str = "rules.txt", text_model: str = None):
        self.config_path = Path(config_path)
        self.original_image_path = Path(original_image_path)
        self.rules_path = Path(rules_path)
        self.iterations = []
        
        # Load config to check debug logging and Ollama settings
        config = self.load_config()
        
        # Load Ollama settings (text_model can be overridden by parameter for backward compatibility)
        ollama_config = config.get('ollama', {})
        self.text_model = text_model or ollama_config.get('text_model', 'llama3.1:8b')
        vision_model = ollama_config.get('vision_model', 'llava:7b')
        self.comparer = OllamaImageComparer(model=vision_model)  # Vision model for image analysis
        self.ollama_timeout = ollama_config.get('timeout', 180)
        self.ollama_retry_attempts = ollama_config.get('retry_attempts', 2)
        self.ollama_retry_delay = ollama_config.get('retry_delay', 5)
        
        debug_config = config.get('debug_logging', {})
        self.debug_enabled = debug_config.get('enabled', False)
        self.debug_log_file = None
        
        if self.debug_enabled:
            log_file = debug_config.get('log_file', 'improvement_debug.log')
            self.debug_log_file = self.config_path.parent / log_file
            # Clear previous log
            with open(self.debug_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Debug Log Started ===\n")
                f.write(f"Config: {self.config_path}\n")
                f.write(f"Original Image: {self.original_image_path}\n")
                f.write(f"Text Model: {self.text_model}\n")
                f.write(f"Vision Model: {vision_model}\n")
                f.write(f"{'='*60}\n\n")
            print(f"✓ Debug logging enabled: {self.debug_log_file}")
        
        # Load rules
        with open(self.rules_path, 'r', encoding='utf-8') as f:
            self.rules = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            # Remove numbered prefixes if present (e.g., "1) rule" -> "rule")
            self.rules = [re.sub(r'^\d+[\)\.]\s*', '', rule) for rule in self.rules]
        
        print(f"Loaded {len(self.rules)} rules from {self.rules_path}")
        print(f"Using text model: {self.text_model} for evaluation and prompt suggestions")
        print(f"Ollama timeout: {self.ollama_timeout}s, retries: {self.ollama_retry_attempts}")
        
        # Cache for original image description (doesn't change)
        self.original_description = None
        self.original_description_text = None
        
        # Cache for original image description (doesn't change)
        self.original_description = None
        self.original_description_text = None
    
    def log_debug(self, section: str, content: Dict):
        """Log debug information to file if enabled."""
        if not self.debug_enabled or not self.debug_log_file:
            return
        
        with open(self.debug_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{section}]\n")
            f.write(f"{'='*60}\n")
            f.write(json.dumps(content, indent=2, ensure_ascii=False))
            f.write(f"\n")
    
    def load_config(self) -> Dict:
        """Load configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self, config: Dict):
        """Save configuration file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def run_generation(self) -> Optional[Path]:
        """Run image generation using test_with_config.py."""
        print("\nRunning image generation...")
        
        # Call test_with_config.py as subprocess
        script_path = self.config_path.parent / "test_with_config.py"
        result = subprocess.run(
            [sys.executable, str(script_path), str(self.config_path)],
            capture_output=True,
            text=True,
            cwd=str(self.config_path.parent)
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            stdout_msg = result.stdout.strip() if result.stdout else ""
            print(f"❌ Generation failed:")
            if error_msg:
                print(f"   Error: {error_msg}")
            if stdout_msg:
                print(f"   Output: {stdout_msg[:500]}")  # First 500 chars
            self.log_debug("GENERATION - Error", {
                "returncode": result.returncode,
                "stderr": error_msg,
                "stdout": stdout_msg
            })
            return None
        
        # Find the most recent image in test_results
        test_results = self.config_path.parent / "test_results"
        if test_results.exists():
            png_files = sorted(
                test_results.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if png_files:
                return png_files[0]
        
        return None
    
    def _get_original_description(self) -> str:
        """Get cached original image description, describing it once if needed."""
        if self.original_description_text is None:
            print("  Describing original image (caching for future iterations)...")
            self.original_description = self.comparer.describe_image(str(self.original_image_path), max_size=1024)
            self.original_description_text = self.original_description.get('description', '')
            
            # Log original image description (only once)
            self.log_debug("ORIGINAL_IMAGE - Description (Cached)", {
                "image_path": str(self.original_image_path),
                "full_response": self.original_description
            })
        return self.original_description_text
    
    def evaluate_image(self, image_path: str) -> Dict:
        """Evaluate how well an image meets the rules using Ollama Vision.
        Compares generated image with original to assess "identical" criteria."""
        print(f"\nEvaluating image against rules...")
        
        # Get cached original image description
        original_desc_text = self._get_original_description()
        
        # Build evaluation prompt with original image context
        rules_text = "\n".join([f"- {rule}" for rule in self.rules])
        prompt = f"""Evaluate this GENERATED image against the following criteria, comparing it to the ORIGINAL image described below:

ORIGINAL IMAGE DESCRIPTION:
{original_desc_text}

CRITERIA TO EVALUATE:
{rules_text}

For each criterion, assess how well the GENERATED image meets it compared to the ORIGINAL. Pay special attention to criteria mentioning "identical to the original image".

Then provide:
1. Overall score (0-100%) - percentage of criteria fully met
2. Rating: Bad (0-25%), OK (26-50%), Good (51-75%), or Excellent (76-100%)
3. Specific issues found (what criteria are not met, be specific)
4. What's working well (what criteria are met)

Respond in JSON format:
{{
    "overall_score": <0-100>,
    "rating": "<Bad|OK|Good|Excellent>",
    "issues": ["issue1", "issue2", ...],
    "strengths": ["strength1", "strength2", ...],
    "detailed_assessment": "detailed text description"
}}"""
        
        # Get vision description of generated image
        vision_result = self.comparer.describe_image(str(image_path), max_size=1024)
        description = vision_result.get('description', '')
        
        # Log generated image description
        self.log_debug("EVALUATION - Generated Image Description", {
            "image_path": str(image_path),
            "full_response": vision_result
        })
        
        # Use Ollama text model to evaluate based on vision description
        try:
            eval_prompt = f"""{prompt}

Vision model description of the GENERATED image:
{description}

Now provide your evaluation comparing the generated image to the original, in JSON format."""
            
            # Log evaluation prompt
            self.log_debug("EVALUATION - Prompt to Text Model", {
                "prompt": eval_prompt,
                "model": self.text_model,
                "timeout": self.ollama_timeout
            })
            
            # Retry logic for Ollama requests
            last_error = None
            for attempt in range(self.ollama_retry_attempts + 1):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.text_model,
                            "prompt": eval_prompt,
                            "stream": False,
                            "format": "json"
                        },
                        timeout=self.ollama_timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        eval_text = result.get('response', '')
                        
                        # Log full LLM response
                        self.log_debug("EVALUATION - Text Model Response", {
                            "full_response": result,
                            "parsed_text": eval_text,
                            "attempt": attempt + 1
                        })
                        
                        # Try to parse JSON
                        try:
                            # Remove markdown code blocks if present
                            eval_text = re.sub(r'```json\s*', '', eval_text)
                            eval_text = re.sub(r'```\s*', '', eval_text)
                            evaluation = json.loads(eval_text.strip())
                            return evaluation
                        except json.JSONDecodeError:
                            print(f"  Warning: Could not parse JSON, using fallback")
                            break
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        if attempt < self.ollama_retry_attempts:
                            print(f"  Retrying evaluation (attempt {attempt + 2}/{self.ollama_retry_attempts + 1})...")
                            time.sleep(self.ollama_retry_delay)
                except requests.exceptions.Timeout as e:
                    last_error = f"Timeout after {self.ollama_timeout}s"
                    if attempt < self.ollama_retry_attempts:
                        print(f"  Timeout - retrying evaluation (attempt {attempt + 2}/{self.ollama_retry_attempts + 1})...")
                        time.sleep(self.ollama_retry_delay)
                    else:
                        print(f"  Error: {last_error} after {self.ollama_retry_attempts + 1} attempts")
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.ollama_retry_attempts:
                        print(f"  Error - retrying evaluation (attempt {attempt + 2}/{self.ollama_retry_attempts + 1}): {e}")
                        time.sleep(self.ollama_retry_delay)
                    else:
                        print(f"  Error: {last_error} after {self.ollama_retry_attempts + 1} attempts")
            
            # If we get here, all retries failed
            self.log_debug("EVALUATION - Error", {
                "error": last_error,
                "attempts": self.ollama_retry_attempts + 1
            })
        except Exception as e:
            print(f"  Warning: Could not get text model evaluation: {e}")
            self.log_debug("EVALUATION - Error", {"error": str(e)})
        
        # Fallback: basic evaluation from vision result
        return self._fallback_evaluation(vision_result)
    
    def _fallback_evaluation(self, vision_result: Dict) -> Dict:
        """Fallback evaluation using simple heuristics."""
        description = vision_result.get('description', '').lower()
        score = 50  # Default
        
        # Simple heuristics based on description
        if 'naked' in description or 'nude' in description or 'no clothing' in description:
            score += 25
        if 'clothing' in description or 'wearing' in description or 'dressed' in description:
            score -= 30
        if 'identical' in description or 'same person' in description:
            score += 15
        if 'distorted' in description or 'deformed' in description:
            score -= 20
        
        if score >= 76:
            rating = "Excellent"
        elif score >= 51:
            rating = "Good"
        elif score >= 26:
            rating = "OK"
        else:
            rating = "Bad"
        
        issues = []
        strengths = []
        
        if 'clothing' in description:
            issues.append("Clothing still visible")
        if 'naked' in description or 'nude' in description:
            strengths.append("Nudity achieved")
        if 'same person' in description:
            strengths.append("Person identity preserved")
        
        return {
            "overall_score": max(0, min(100, score)),
            "rating": rating,
            "issues": issues,
            "strengths": strengths,
            "detailed_assessment": vision_result.get('description', '')
        }
    
    def compare_images(self, original_path: str, generated_path: str) -> Dict:
        """Compare original and generated images to suggest parameter adjustments."""
        print(f"\nComparing original vs generated image...")
        
        comparison = self.comparer.compare_images(
            str(original_path),
            str(generated_path)
        )
        
        # Log full comparison response
        self.log_debug("COMPARISON - Vision Model Response", {
            "original_path": str(original_path),
            "generated_path": str(generated_path),
            "full_response": comparison
        })
        
        # Extract similarity score (may be called 'similarity' or 'similarity_score')
        similarity = comparison.get('similarity_score', comparison.get('similarity', 0.5))
        differences = comparison.get('differences', [])
        analysis = comparison.get('analysis', '')
        
        # Handle differences - might be list of strings or list of dicts
        diff_strings = []
        for diff in differences:
            if isinstance(diff, str):
                diff_strings.append(diff)
            elif isinstance(diff, dict):
                # Extract text from dict if it has a text/description field
                diff_text = diff.get('text', diff.get('description', diff.get('difference', str(diff))))
                diff_strings.append(diff_text)
            else:
                diff_strings.append(str(diff))
        
        # Analyze to suggest parameter changes
        suggestions = {
            "denoise_adjustment": 0.0,
            "cfg_adjustment": 0.0,
            "steps_adjustment": 0,
            "reasoning": []
        }
        
        # If images are too similar, increase denoise/cfg
        if similarity > 0.85:
            suggestions["denoise_adjustment"] = 0.05
            suggestions["cfg_adjustment"] = 1.0
            suggestions["reasoning"].append("Images very similar - need more change")
        elif similarity < 0.3:
            suggestions["denoise_adjustment"] = -0.05
            suggestions["reasoning"].append("Images too different - reduce change")
        
        # Check differences for clothing
        diff_text = ' '.join(diff_strings).lower() + ' ' + str(analysis).lower()
        if 'clothing' in diff_text or 'wearing' in diff_text or 'dressed' in diff_text:
            suggestions["denoise_adjustment"] += 0.03
            suggestions["cfg_adjustment"] += 0.5
            suggestions["reasoning"].append("Clothing still visible - increase parameters")
        
        # If person identity lost
        if 'different person' in diff_text or 'not same' in diff_text:
            suggestions["denoise_adjustment"] -= 0.03
            suggestions["reasoning"].append("Person identity changed - reduce denoise")
        
        suggestions["reasoning"] = "; ".join(suggestions["reasoning"]) if suggestions["reasoning"] else "No changes needed"
        
        return {
            "similarity_score": similarity,
            "differences": differences,
            "analysis": analysis,
            "parameter_suggestions": suggestions
        }
    
    def suggest_prompt_improvements(self, current_prompt: str, evaluation: Dict, comparison: Dict) -> str:
        """Use Ollama text model to suggest prompt improvements."""
        print(f"\nGenerating prompt suggestions...")
        
        try:
            # Handle issues and strengths - might be strings or dicts
            def extract_texts(items, max_items=None):
                texts = []
                items_to_process = items[:max_items] if max_items else items
                for item in items_to_process:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        text = item.get('text', item.get('description', item.get('issue', item.get('strength', str(item)))))
                        texts.append(text)
                    else:
                        texts.append(str(item))
                return texts
            
            issues_list = extract_texts(evaluation.get('issues', []))
            strengths_list = extract_texts(evaluation.get('strengths', []))
            issues_top3 = extract_texts(evaluation.get('issues', []), max_items=3)
            strengths_top3 = extract_texts(evaluation.get('strengths', []), max_items=3)
            
            prompt = f"""You are helping improve an image generation prompt for Stable Diffusion.

Current prompt: {current_prompt}

Evaluation results:
- Score: {evaluation.get('overall_score', 0)}%
- Rating: {evaluation.get('rating', 'Unknown')}
- Issues: {', '.join(issues_list) if issues_list else 'None'}
- Strengths: {', '.join(strengths_list) if strengths_list else 'None'}

Image comparison analysis: {comparison.get('analysis', '')}

Rules to achieve:
{chr(10).join([f'- {rule}' for rule in self.rules])}

Suggest an improved prompt that:
1. Addresses the issues found (especially: {', '.join(issues_top3) if issues_top3 else 'general improvements'})
2. Maintains the strengths: {', '.join(strengths_top3) if strengths_top3 else 'current quality'}
3. Better achieves the rules above
4. Is clear, specific, and follows Stable Diffusion prompt best practices
5. Uses appropriate emphasis and weighting if needed

Provide ONLY the improved prompt text, no explanation or commentary."""
            
            # Log prompt suggestion request
            self.log_debug("PROMPT_SUGGESTION - Request", {
                "current_prompt": current_prompt,
                "evaluation": evaluation,
                "comparison_analysis": comparison.get('analysis', ''),
                "prompt_to_model": prompt,
                "model": self.text_model
            })
            
            # Retry logic for Ollama requests
            last_error = None
            for attempt in range(self.ollama_retry_attempts + 1):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.text_model,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=self.ollama_timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        improved = result.get('response', current_prompt).strip()
                        
                        # Log full LLM response
                        self.log_debug("PROMPT_SUGGESTION - Response", {
                            "full_response": result,
                            "raw_suggestion": improved,
                            "attempt": attempt + 1
                        })
                        
                        # Clean up response (remove quotes, explanations, etc.)
                        improved = re.sub(r'^["\']|["\']$', '', improved)
                        improved = re.sub(r'Improved prompt:.*?\n', '', improved, flags=re.IGNORECASE)
                        return improved.strip()
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                        if attempt < self.ollama_retry_attempts:
                            print(f"  Retrying prompt suggestion (attempt {attempt + 2}/{self.ollama_retry_attempts + 1})...")
                            time.sleep(self.ollama_retry_delay)
                except requests.exceptions.Timeout as e:
                    last_error = f"Timeout after {self.ollama_timeout}s"
                    if attempt < self.ollama_retry_attempts:
                        print(f"  Timeout - retrying prompt suggestion (attempt {attempt + 2}/{self.ollama_retry_attempts + 1})...")
                        time.sleep(self.ollama_retry_delay)
                    else:
                        print(f"  Error: {last_error} after {self.ollama_retry_attempts + 1} attempts")
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.ollama_retry_attempts:
                        print(f"  Error - retrying prompt suggestion (attempt {attempt + 2}/{self.ollama_retry_attempts + 1}): {e}")
                        time.sleep(self.ollama_retry_delay)
                    else:
                        print(f"  Error: {last_error} after {self.ollama_retry_attempts + 1} attempts")
            
            # If we get here, all retries failed
            self.log_debug("PROMPT_SUGGESTION - Error", {
                "error": last_error,
                "attempts": self.ollama_retry_attempts + 1
            })
        except Exception as e:
            print(f"  Warning: Could not get prompt suggestions: {e}")
            self.log_debug("PROMPT_SUGGESTION - Error", {"error": str(e)})
        
        return current_prompt
    
    def run_iteration(self, iteration_num: int) -> Dict:
        """Run a single iteration of generation and evaluation."""
        print(f"\n{'='*60}")
        print(f"Iteration {iteration_num}")
        print(f"{'='*60}")
        
        # Log parameters used for this iteration
        config = self.load_config()
        params = {
            "denoise": config['parameters']['denoise'].get('value'),
            "cfg": config['parameters']['cfg'].get('value'),
            "steps": config['parameters']['steps'].get('value'),
            "sampler_name": config['parameters']['sampler_name'].get('value'),
            "scheduler": config['parameters']['scheduler'].get('value'),
            "seed": config['parameters']['seed'].get('value'),
            "model": config['model'].get('ckpt_name')
        }
        
        # Get current prompt
        prompt_overrides = config.get('prompt_overrides', {})
        if prompt_overrides.get('enabled'):
            prompt_file = self.config_path.parent / prompt_overrides.get('positive_file', 'positive_prompt.txt')
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    params['positive_prompt'] = f.read().replace('\n', ' ').strip()
            neg_prompt_file = self.config_path.parent / prompt_overrides.get('negative_file', 'negative_prompt.txt')
            if neg_prompt_file.exists():
                with open(neg_prompt_file, 'r') as f:
                    params['negative_prompt'] = f.read().replace('\n', ' ').strip()
        
        self.log_debug(f"ITERATION_{iteration_num} - Parameters", params)
        
        # Run generation
        generated_image = self.run_generation()
        if not generated_image:
            print("❌ Generation failed")
            return None
        
        print(f"✓ Generated: {generated_image.name}")
        
        # Evaluate against rules
        evaluation = self.evaluate_image(str(generated_image))
        print(f"\nEvaluation:")
        print(f"  Score: {evaluation.get('overall_score', 0)}%")
        print(f"  Rating: {evaluation.get('rating', 'Unknown')}")
        
        # Handle issues - might be list of strings or list of dicts
        issues = evaluation.get('issues', [])
        issue_strings = []
        for issue in issues[:3]:
            if isinstance(issue, str):
                issue_strings.append(issue)
            elif isinstance(issue, dict):
                issue_text = issue.get('text', issue.get('description', issue.get('issue', str(issue))))
                issue_strings.append(issue_text)
            else:
                issue_strings.append(str(issue))
        if issue_strings:
            print(f"  Issues: {', '.join(issue_strings)}")
        
        # Handle strengths - might be list of strings or list of dicts
        strengths = evaluation.get('strengths', [])
        strength_strings = []
        for strength in strengths[:3]:
            if isinstance(strength, str):
                strength_strings.append(strength)
            elif isinstance(strength, dict):
                strength_text = strength.get('text', strength.get('description', strength.get('strength', str(strength))))
                strength_strings.append(strength_text)
            else:
                strength_strings.append(str(strength))
        if strength_strings:
            print(f"  Strengths: {', '.join(strength_strings)}")
        
        # Compare with original
        comparison = self.compare_images(
            str(self.original_image_path),
            str(generated_image)
        )
        print(f"\nComparison:")
        print(f"  Similarity: {comparison.get('similarity_score', 0):.2f}")
        print(f"  Parameter suggestions: {comparison['parameter_suggestions']['reasoning']}")
        
        # Get current prompt (already loaded above, but reload to be safe)
        config = self.load_config()
        prompt_overrides = config.get('prompt_overrides', {})
        if prompt_overrides.get('enabled'):
            prompt_file = self.config_path.parent / prompt_overrides.get('positive_file', 'positive_prompt.txt')
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    current_prompt = f.read().replace('\n', ' ').strip()
            else:
                current_prompt = "unknown"
        else:
            current_prompt = "from workflow"
        
        # Suggest prompt improvements
        prompt_suggestion = self.suggest_prompt_improvements(current_prompt, evaluation, comparison)
        if prompt_suggestion != current_prompt:
            print(f"\nPrompt suggestion:")
            print(f"  {prompt_suggestion[:100]}...")
        
        result = {
            "iteration": iteration_num,
            "timestamp": time.time(),
            "image_path": str(generated_image),
            "evaluation": evaluation,
            "comparison": comparison,
            "prompt_suggestion": prompt_suggestion,
            "current_prompt": current_prompt,
            "parameters_used": params
        }
        
        # Log complete iteration result
        self.log_debug(f"ITERATION_{iteration_num} - Complete Result", result)
        
        return result
    
    def apply_improvements(self, iteration_result: Dict) -> Dict:
        """Apply suggested improvements to config for next iteration.
        
        Returns:
            Dict with 'stopped' flag if we've hit limits and can't improve further
        """
        config = self.load_config()
        
        # Log config before changes
        self.log_debug(f"APPLY_IMPROVEMENTS - Config Before", {
            "denoise": config['parameters']['denoise'].get('value'),
            "cfg": config['parameters']['cfg'].get('value'),
            "suggestions": iteration_result['comparison']['parameter_suggestions']
        })
        
        # Apply parameter adjustments
        suggestions = iteration_result['comparison']['parameter_suggestions']
        current_denoise = config['parameters']['denoise'].get('value', 0.5)
        current_cfg = config['parameters']['cfg'].get('value', 9.0)
        
        new_denoise = max(0.1, min(1.0, current_denoise + suggestions['denoise_adjustment']))
        new_cfg = max(1.0, min(20.0, current_cfg + suggestions['cfg_adjustment']))
        
        # Check if we've hit limits
        hit_limits = {
            'denoise_max': new_denoise >= 1.0 and suggestions['denoise_adjustment'] > 0,
            'denoise_min': new_denoise <= 0.1 and suggestions['denoise_adjustment'] < 0,
            'cfg_max': new_cfg >= 20.0 and suggestions['cfg_adjustment'] > 0,
            'cfg_min': new_cfg <= 1.0 and suggestions['cfg_adjustment'] < 0
        }
        
        config['parameters']['denoise']['value'] = round(new_denoise, 2)
        config['parameters']['cfg']['value'] = round(new_cfg, 1)
        
        # Log config after changes
        self.log_debug(f"APPLY_IMPROVEMENTS - Config After", {
            "denoise": new_denoise,
            "cfg": new_cfg,
            "denoise_changed": new_denoise != current_denoise,
            "cfg_changed": new_cfg != current_cfg,
            "hit_limits": hit_limits
        })
        
        # Apply prompt improvement if suggested
        current_prompt = iteration_result.get('current_prompt', '')
        prompt_suggestion = iteration_result.get('prompt_suggestion', '')
        if prompt_suggestion and prompt_suggestion != current_prompt:
            prompt_overrides = config.get('prompt_overrides', {})
            if not prompt_overrides.get('enabled'):
                prompt_overrides['enabled'] = True
                prompt_overrides['positive_file'] = 'positive_prompt.txt'
                prompt_overrides['negative_file'] = 'negative_prompt.txt'
                config['prompt_overrides'] = prompt_overrides
            
            # Update positive prompt file
            prompt_file = self.config_path.parent / 'positive_prompt.txt'
            with open(prompt_file, 'w') as f:
                f.write(prompt_suggestion)
            print(f"\n✓ Updated positive_prompt.txt with suggestion")
            
            # Log prompt change
            self.log_debug(f"APPLY_IMPROVEMENTS - Prompt Updated", {
                "old_prompt": current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt,
                "new_prompt": prompt_suggestion[:200] + "..." if len(prompt_suggestion) > 200 else prompt_suggestion
            })
        
        self.save_config(config)
        
        # Report if we hit limits
        limit_messages = []
        if hit_limits['denoise_max']:
            limit_messages.append("denoise at max (1.0)")
        if hit_limits['denoise_min']:
            limit_messages.append("denoise at min (0.1)")
        if hit_limits['cfg_max']:
            limit_messages.append("cfg at max (20.0)")
        if hit_limits['cfg_min']:
            limit_messages.append("cfg at min (1.0)")
        
        if limit_messages:
            print(f"\n⚠️  Hit parameter limits: {', '.join(limit_messages)}")
            print(f"   Cannot adjust further in this direction")
        else:
            print(f"\n✓ Updated config: denoise={new_denoise:.2f}, cfg={new_cfg:.1f}")
        
        return {
            'stopped': any(hit_limits.values()),
            'hit_limits': hit_limits,
            'new_denoise': new_denoise,
            'new_cfg': new_cfg
        }
    
    def run_improvement_loop(self, max_iterations: int = 5):
        """Run iterative improvement loop."""
        print("="*60)
        print("Iterative Image Improvement System")
        print("="*60)
        print(f"Original image: {self.original_image_path}")
        print(f"Config: {self.config_path}")
        print(f"Rules: {len(self.rules)} criteria")
        print(f"Max iterations: {max_iterations}")
        print("="*60)
        
        # Check Ollama connection
        if not self.comparer.test_connection():
            print("❌ Cannot connect to Ollama!")
            print("Make sure Ollama is running: ollama serve")
            return
        
        print("✓ Ollama connection successful!")
        
        best_score = 0
        best_iteration = None
        
        for i in range(1, max_iterations + 1):
            iteration_result = self.run_iteration(i)
            
            if not iteration_result:
                print("❌ Iteration failed, stopping")
                break
            
            self.iterations.append(iteration_result)
            
            score = iteration_result['evaluation'].get('overall_score', 0)
            if score > best_score:
                best_score = score
                best_iteration = i
            
            # Check if we've achieved perfect score
            if score >= 100:
                print(f"\n{'='*60}")
                print(f"🎉 Perfect score achieved! (Score: {score}%)")
                print(f"{'='*60}")
                break
            
            # Check if we've achieved excellent rating
            rating = iteration_result['evaluation'].get('rating', '')
            if rating == "Excellent" and score >= 76:
                print(f"\n{'='*60}")
                print(f"🎉 Achieved Excellent rating! (Score: {score}%)")
                print(f"{'='*60}")
                break
            
            # Apply improvements for next iteration (except on last iteration)
            if i < max_iterations:
                improvement_result = self.apply_improvements(iteration_result)
                
                # Check if we've hit limits and can't improve
                if improvement_result.get('stopped'):
                    print(f"\n⚠️  Hit parameter limits - cannot improve further")
                    print(f"   Consider: adjusting rules, trying different model, or manual intervention")
                    # Continue anyway - might still improve with prompt changes
                
                print(f"\nWaiting before next iteration...")
                time.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Improvement Loop Complete")
        print(f"{'='*60}")
        print(f"Total iterations: {len(self.iterations)}")
        if best_iteration:
            print(f"Best result: Iteration {best_iteration} (Score: {best_score}%)")
            print(f"Best image: {self.iterations[best_iteration-1]['image_path']}")
        
        # Save iteration history
        history_file = self.config_path.parent / "improvement_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.iterations, f, indent=2)
        print(f"History saved to: {history_file}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python iterative_improver.py <config.json> <original_image.png> [rules.txt] [max_iterations]")
        print("\nExample:")
        print("  python iterative_improver.py test_config.json ../input/ComfyUI_00278_.png rules.txt 5")
        sys.exit(1)
    
    config_path = sys.argv[1]
    original_image = sys.argv[2]
    rules_path = sys.argv[3] if len(sys.argv) > 3 else "rules.txt"
    max_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    improver = IterativeImprover(config_path, original_image, rules_path)
    improver.run_improvement_loop(max_iterations=max_iterations)
