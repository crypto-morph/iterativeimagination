#!/usr/bin/env python3
"""
AIVis Client - Ollama Vision Model Integration

Handles communication with Ollama vision models (qwen3-vl:4b) for image analysis,
question answering, evaluation, and prompt improvement.
"""

import base64
import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests
import yaml
from PIL import Image
import io


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and cannot be retried."""
    def __init__(self, message: str, reset_time: Optional[int] = None):
        super().__init__(message)
        self.reset_time = reset_time


class AIVisClient:
    """Client for interacting with vision models via Ollama or OpenRouter API."""
    
    def __init__(self, model: str = "qwen3-vl:4b", base_url: str = None, 
                 prompts_path: Path = None, max_concurrent: int = 1,
                 provider: str = "ollama", api_key: str = None,
                 fallback_provider: Optional[str] = None, fallback_model: Optional[str] = None):
        """
        Initialize AIVis client.
        
        Args:
            model: Model name (e.g., "qwen3-vl:4b" for Ollama, "qwen/qwen-2-vl-7b-instruct:free" for OpenRouter)
            base_url: Base URL for Ollama (default: http://localhost:11434)
            prompts_path: Path to prompts.yaml file
            max_concurrent: Maximum concurrent requests (default: 1)
            provider: "ollama" or "openrouter" (default: "ollama")
            api_key: API key for OpenRouter (can also use OPENROUTER_API_KEY env var)
        """
        import os
        
        self.model = model
        self.provider = provider.lower()
        self.timeout = 180
        self.retry_attempts = 2
        self.retry_delay = 5
        
        # Setup provider-specific configuration
        if self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter.")
            # OpenRouter can handle more concurrent requests
            max_concurrent = max(max_concurrent, 2)
        else:  # ollama
            self.base_url = base_url or "http://localhost:11434"
            self.api_key = None
        
        # Fallback provider for rate limit situations
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self._using_fallback = False
        
        # Track last request metadata
        self._last_request_metadata = None
        
        # Semaphore to limit concurrent requests
        self._request_semaphore = threading.Semaphore(max_concurrent)
        
        # Load prompts from YAML file
        self.prompts = self._load_prompts(prompts_path)
    
    def get_last_request_metadata(self) -> Optional[Dict]:
        """Get metadata from the last request (provider, model, route, etc.)."""
        return self._last_request_metadata
    
    def _load_prompts(self, prompts_path: Path = None) -> Dict[str, str]:
        """Load prompts from YAML file.

        We support project-specific prompt overrides. If a project file is missing
        newly-added prompt keys, we merge from repo defaults so prompts are never empty.
        """
        # __file__ is in src/, so go up one level to repo root
        repo_root = Path(__file__).parent.parent

        # Built-in baseline (ultimate fallback)
        baseline: Dict[str, str] = {
                "ask_question": "Answer this question about the image: {question}\n\nQuestion type: {type_info}{enum_info}{min_max_info}\n\nProvide your answer in JSON format matching the question type:\n{{\n    \"answer\": <your_answer_here>\n}}\n\nFor boolean questions, use true/false.\nFor number/integer questions, provide a numeric value.\nFor string questions, provide a text description.\nFor array questions, provide a JSON array.",
                "evaluate_acceptance_criteria": "Evaluate this GENERATED image against acceptance criteria, comparing it to the ORIGINAL image.\n\nORIGINAL IMAGE DESCRIPTION:\n{original_description}\n\nQUESTION ANSWERS (for context):\n{questions_summary}\n\nACCEPTANCE CRITERIA:\n{criteria_text}\n\nFor each criterion, determine if it passes (true/false or within min/max range).\nThen calculate overall score: (passed_criteria / total_criteria) * 100\n\nProvide your evaluation in JSON format:\n{{\n    \"overall_score\": <0-100>,\n    \"criteria_results\": {{\n        \"<field1>\": <true/false or value>,\n        \"<field2>\": <true/false or value>,\n        ...\n    }}\n}}",
                "compare_images": "Compare these two images and analyze the differences.\n\nFocus on:\n1. Similarity score (0.0 = completely different, 1.0 = identical)\n2. What differences exist? (clothing, pose, features, background, proportions)\n3. Overall analysis\n\nProvide your comparison in JSON format:\n{{\n    \"similarity_score\": <0.0-1.0>,\n    \"differences\": [\"difference1\", \"difference2\", ...],\n    \"analysis\": \"detailed analysis text\"\n}}",
                "describe_image": "Describe this image in detail. Focus on:\n1. Main subject (person, object, scene)\n2. If person: appearance, pose, clothing\n3. Background/setting\n4. Overall composition and notable details\n\nProvide a detailed description.",
                "improve_prompts": "You are helping improve image generation prompts for Stable Diffusion.\n\nCurrent positive prompt: {current_positive}\nCurrent negative prompt: {current_negative}\n\nEvaluation results:\n- Overall score: {overall_score}%\n- Failed criteria: {failed_criteria}\n\nImage comparison:\n- Similarity: {similarity_score}\n- Differences: {differences}\n- Analysis: {analysis}\n\nRules to achieve:\n{rules_text}\n\nSuggest improved positive and negative prompts that:\n1. Address the failed criteria\n2. Maintain what's working well\n3. Better achieve the rules\n4. Follow Stable Diffusion prompt best practices\n\nProvide your suggestions in JSON format:\n{{\n    \"positive\": \"<improved positive prompt>\",\n    \"negative\": \"<improved negative prompt>\"\n}}"

        }

        # Merge repo defaults/prompts.yaml then repo_root/prompts.yaml (if present)
        # so new prompt keys are available even if a project override file is stale.
        merged: Dict[str, str] = dict(baseline)

        defaults_path = repo_root / "defaults" / "prompts.yaml"
        if defaults_path.exists():
            with open(defaults_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    merged.update(loaded)

        canonical_path = repo_root / "prompts.yaml"
        if canonical_path.exists():
            with open(canonical_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    merged.update(loaded)

        # If no explicit prompts_path, default to repo_root/prompts.yaml or defaults
        if prompts_path is None:
            prompts_path = canonical_path if canonical_path.exists() else defaults_path

        # Finally, merge explicit prompts file overrides (project-specific)
        try:
            if prompts_path and Path(prompts_path).exists():
                with open(prompts_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                    if isinstance(loaded, dict):
                        merged.update(loaded)
        except Exception:
            # If prompt loading fails, fall back to merged baseline
            pass

        return merged
    
    def _image_to_base64(self, image_path: str, max_size: int = 1024) -> str:
        """Convert image to base64, optionally resizing."""
        img = Image.open(image_path).convert('RGB')
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            if json_end > json_start:
                response_text = response_text[json_start:json_end].strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON object from text
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            raise
    
    def _make_request_with_retry(self, payload: Dict, parse_json: bool = False, log_progress: bool = False) -> Tuple[Any, Dict]:
        """Make request with retry logic, using semaphore to limit concurrency.
        
        Returns:
            Tuple of (result, metadata) where metadata contains provider, model, and route info.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Track metadata for this request
        request_metadata = {
            "provider": self.provider,
            "model": self.model,
            "using_fallback": self._using_fallback,
            "attempts": 0
        }
        
        # Acquire semaphore to limit concurrent requests
        self._request_semaphore.acquire()
        try:
            last_error = None
            for attempt in range(self.retry_attempts + 1):
                try:
                    # Update payload model to match current provider (in case we switched)
                    payload['model'] = self.model
                    
                    # Update metadata to reflect current state
                    request_metadata["provider"] = self.provider
                    request_metadata["model"] = self.model
                    request_metadata["using_fallback"] = self._using_fallback
                    request_metadata["attempts"] = attempt + 1
                    
                    if log_progress:
                        logger.debug(f"Making {self.provider} request (attempt {attempt + 1}/{self.retry_attempts + 1})...")
                        logger.debug(f"  Model: {payload.get('model')}")
                        logger.debug(f"  Timeout: {self.timeout}s")
                    
                    # Prepare request based on provider
                    if self.provider == "openrouter":
                        # OpenRouter API format
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "HTTP-Referer": "https://github.com/ComfyScripts/iterative-imagination",
                            "X-Title": "Iterative Imagination",
                            "Content-Type": "application/json"
                        }
                        api_url = f"{self.base_url}/chat/completions"
                        # Convert to OpenRouter format
                        openrouter_payload = self._convert_to_openrouter_format(payload)
                        if log_progress:
                            logger.debug(f"  OpenRouter URL: {api_url}")
                            logger.debug(f"  Model: {openrouter_payload.get('model')}")
                            logger.debug(f"  Messages count: {len(openrouter_payload.get('messages', []))}")
                            # Log content structure (but not full base64 images)
                            if openrouter_payload.get('messages'):
                                msg = openrouter_payload['messages'][0]
                                if isinstance(msg.get('content'), list):
                                    content_types = [c.get('type') for c in msg['content']]
                                    logger.debug(f"  Content types: {content_types}")
                                    logger.debug(f"  Text length: {sum(len(c.get('text', '')) for c in msg['content'] if c.get('type') == 'text')}")
                                    logger.debug(f"  Image count: {sum(1 for c in msg['content'] if c.get('type') == 'image_url')}")
                        response = requests.post(api_url, json=openrouter_payload, headers=headers, timeout=self.timeout)
                    else:  # ollama
                        api_url = f"{self.base_url}/api/generate"
                        response = requests.post(api_url, json=payload, timeout=self.timeout)
                    
                    # Log error details for 400 responses
                    if response.status_code == 400:
                        try:
                            error_body = response.json()
                            logger.error(f"  OpenRouter 400 Error: {error_body}")
                        except:
                            logger.error(f"  OpenRouter 400 Error (raw): {response.text[:500]}")
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract response text based on provider
                    if self.provider == "openrouter":
                        response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:  # ollama
                        response_text = result.get('response', '')
                    
                    # Check for empty response
                    if not response_text or not response_text.strip():
                        error_msg = f"Empty response from {self.provider} (model: {self.model})"
                        logger.error(f"  {error_msg}")
                        logger.error(f"  Full response: {result}")
                        if attempt < self.retry_attempts:
                            logger.warning(f"  Retrying in {self.retry_delay}s...")
                            time.sleep(self.retry_delay)
                            continue
                        raise Exception(error_msg)
                    
                    if log_progress:
                        logger.debug(f"  Response received ({len(response_text)} chars)")
                        if len(response_text) < 100:
                            logger.debug(f"  Response preview: {response_text[:100]}")
                    
                    if parse_json:
                        try:
                            parsed_result = self._parse_json_response(response_text)
                            request_metadata["success"] = True
                            return parsed_result, request_metadata
                        except json.JSONDecodeError as e:
                            error_msg = f"Failed to parse JSON from {self.provider} response: {e}"
                            logger.error(f"  {error_msg}")
                            logger.error(f"  Response text (first 500 chars): {response_text[:500]}")
                            if attempt < self.retry_attempts:
                                logger.warning(f"  Retrying in {self.retry_delay}s...")
                                time.sleep(self.retry_delay)
                                continue
                            request_metadata["success"] = False
                            request_metadata["error"] = error_msg
                            raise Exception(error_msg)
                    request_metadata["success"] = True
                    return response_text, request_metadata
                
                except requests.exceptions.Timeout:
                    last_error = f"Timeout after {self.timeout}s"
                    if log_progress:
                        logger.warning(f"  Request timed out (attempt {attempt + 1})")
                    if attempt < self.retry_attempts:
                        if log_progress:
                            logger.debug(f"  Retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                except requests.exceptions.HTTPError as e:
                    # Handle rate limit (429) errors
                    if e.response.status_code == 429:
                        reset_time = None
                        wait_seconds = None
                        
                        # Try to get rate limit reset time from headers
                        if hasattr(e.response, 'headers'):
                            reset_header = e.response.headers.get('X-RateLimit-Reset')
                            if reset_header:
                                try:
                                    reset_time = int(reset_header) / 1000  # Convert from milliseconds
                                    wait_seconds = max(0, reset_time - time.time())
                                except (ValueError, TypeError):
                                    pass
                        
                        # Log rate limit error
                        if hasattr(e.response, 'json'):
                            try:
                                error_body = e.response.json()
                                logger.error(f"  Rate limit exceeded: {error_body}")
                            except:
                                logger.error(f"  Rate limit exceeded (429)")
                        
                        # If we have a fallback provider and haven't switched yet, switch to it
                        if self.fallback_provider and not self._using_fallback:
                            logger.warning(f"  Rate limited on {self.provider}. Switching to fallback: {self.fallback_provider}")
                            old_model = payload.get('model')
                            self._switch_to_fallback()
                            # Update payload model to match new provider
                            payload['model'] = self.model
                            logger.debug(f"  Updated payload model from '{old_model}' to '{self.model}'")
                            # Retry immediately with fallback
                            continue
                        
                        # If wait time is reasonable (< 5 minutes), wait and retry
                        if wait_seconds and wait_seconds < 300:
                            logger.warning(f"  Rate limit will reset in {wait_seconds:.0f} seconds. Waiting...")
                            time.sleep(min(wait_seconds + 5, 300))  # Wait with small buffer
                            continue
                        
                        # Otherwise, raise RateLimitError
                        raise RateLimitError(
                            f"Rate limit exceeded on {self.provider}. Reset time: {reset_time}",
                            reset_time=int(reset_time) if reset_time else None
                        )
                    
                    # Log detailed error for other HTTP errors
                    if hasattr(e.response, 'json'):
                        try:
                            error_body = e.response.json()
                            logger.error(f"  HTTP {e.response.status_code} Error: {error_body}")
                        except:
                            logger.error(f"  HTTP {e.response.status_code} Error (raw): {e.response.text[:500]}")
                    last_error = str(e)
                    if log_progress:
                        logger.warning(f"  Request error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts:
                        if log_progress:
                            logger.debug(f"  Retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                except Exception as e:
                    last_error = str(e)
                    if log_progress:
                        logger.warning(f"  Request error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry_attempts:
                        if log_progress:
                            logger.debug(f"  Retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
            
            request_metadata["success"] = False
            request_metadata["error"] = last_error
            raise Exception(f"Request failed after {self.retry_attempts + 1} attempts: {last_error}")
        finally:
            # Always release semaphore, even on error
            self._request_semaphore.release()
    
    def _switch_to_fallback(self):
        """Switch to fallback provider when rate limited."""
        if not self.fallback_provider:
            return
        
        import os
        old_provider = self.provider
        self.provider = self.fallback_provider.lower()
        self._using_fallback = True
        
        if self.fallback_provider == "ollama":
            self.base_url = "http://localhost:11434"
            self.api_key = None
            if self.fallback_model:
                self.model = self.fallback_model
        elif self.fallback_provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
            if self.fallback_model:
                self.model = self.fallback_model
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Switched from {old_provider} to {self.fallback_provider} (model: {self.model})")
    
    def _convert_to_openrouter_format(self, ollama_payload: Dict) -> Dict:
        """Convert Ollama API format to OpenRouter API format."""
        # OpenRouter uses chat completions format
        messages = []
        
        # Extract images and prompt
        images = ollama_payload.get('images', [])
        prompt = ollama_payload.get('prompt', '')
        
        # Build content array (text + images)
        # For vision models, images should come first, then text
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
        # Add text prompt after images
        if prompt:
            content.append({"type": "text", "text": prompt})
        
        # If no content, add empty text
        if not content:
            content = [{"type": "text", "text": ""}]
        
        messages.append({"role": "user", "content": content})
        
        result = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        # Note: Some vision models don't support response_format parameter
        # We rely on the prompt to request JSON format instead
        # If the model supports it, we could add:
        # if ollama_payload.get('format') == 'json':
        #     result["response_format"] = {"type": "json_object"}
        
        return result
    
    def ask_question(self, image_path: str, question: str, question_schema: Dict) -> Any:
        """Ask a question about an image and return structured answer."""
        image_b64 = self._image_to_base64(image_path)
        
        # Build prompt with schema information
        type_info = question_schema.get('type', 'string')
        enum_info = ""
        if 'enum' in question_schema:
            enum_info = f"\nAllowed values: {question_schema['enum']}"
        
        min_max_info = ""
        if 'min' in question_schema and 'max' in question_schema:
            min_max_info = f"\nRange: {question_schema['min']} to {question_schema['max']}"
        
        prompt = self.prompts['ask_question'].format(
            question=question,
            type_info=type_info,
            enum_info=enum_info,
            min_max_info=min_max_info
        )
        
        # Prepare payload based on provider
        if self.provider == "openrouter":
            # OpenRouter format will be converted in _make_request_with_retry
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json"
            }
        else:  # ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json"
            }
        
        try:
            parsed, metadata = self._make_request_with_retry(payload, parse_json=True)
            answer = parsed.get('answer', None)
            # Store metadata for later retrieval
            self._last_request_metadata = metadata
            return answer
        except Exception as e:
            # Store error metadata
            self._last_request_metadata = {
                "provider": self.provider,
                "model": self.model,
                "using_fallback": self._using_fallback,
                "success": False,
                "error": str(e)
            }
            # Return default based on type
            qtype = question_schema.get('type', 'string')
            if qtype == 'boolean':
                return False
            elif qtype in ['number', 'integer']:
                return question_schema.get('min', 0)
            elif qtype == 'array':
                return []
            return ""
    
    def ask_multiple_questions(self, image_path: str, questions: List[Dict], original_description: str = None) -> Tuple[Dict, Dict]:
        """Ask multiple questions about an image in a single request. Returns (answers_dict, metadata)."""
        image_b64 = self._image_to_base64(image_path)
        
        # Build questions list with schema information
        questions_list = []
        for i, q_def in enumerate(questions, 1):
            field = q_def['field']
            question_text = q_def['question']
            qtype = q_def.get('type', 'string')
            
            # Build type constraints
            type_info = f"Type: {qtype}"
            enum_info = ""
            if 'enum' in q_def:
                enum_info = f", Allowed values: {q_def['enum']}"
            
            min_max_info = ""
            if 'min' in q_def and 'max' in q_def:
                min_max_info = f", Range: {q_def['min']} to {q_def['max']}"
            
            questions_list.append(
                f"{i}. Field: {field}\n   Question: {question_text}\n   {type_info}{enum_info}{min_max_info}"
            )
        
        questions_text = "\n\n".join(questions_list)
        
        # Add original image context if provided
        original_image_context = ""
        if original_description:
            original_image_context = f"\n\nORIGINAL IMAGE DESCRIPTION (for comparison questions):\n{original_description}"
        
        # Build prompt
        prompt = self.prompts.get('ask_multiple_questions', self.prompts.get('ask_question', '')).format(
            questions_list=questions_text,
            original_image_context=original_image_context
        )
        
        # Prepare payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json"
        }
        
        try:
            parsed, metadata = self._make_request_with_retry(payload, parse_json=True)
            self._last_request_metadata = metadata
            
            # Extract answers - the response should be a dict with field names as keys
            answers = {}
            for q_def in questions:
                field = q_def['field']
                # Try to get answer from parsed response
                if isinstance(parsed, dict):
                    answer = parsed.get(field)
                    if answer is None:
                        # Fallback: try lowercase field name
                        answer = parsed.get(field.lower())
                    if answer is None:
                        # Use default based on type
                        qtype = q_def.get('type', 'string')
                        answer = False if qtype == 'boolean' else (q_def.get('min', 0) if qtype in ['number', 'integer'] else ([] if qtype == 'array' else ""))
                else:
                    # Unexpected format, use defaults
                    qtype = q_def.get('type', 'string')
                    answer = False if qtype == 'boolean' else (q_def.get('min', 0) if qtype in ['number', 'integer'] else ([] if qtype == 'array' else ""))
                
                answers[field] = answer
            
            return answers, metadata
        except Exception as e:
            # Store error metadata
            metadata = {
                "provider": self.provider,
                "model": self.model,
                "using_fallback": self._using_fallback,
                "success": False,
                "error": str(e),
                "attempts": 1
            }
            self._last_request_metadata = metadata
            
            # Return defaults for all questions
            answers = {}
            for q_def in questions:
                field = q_def['field']
                qtype = q_def.get('type', 'string')
                answers[field] = False if qtype == 'boolean' else (q_def.get('min', 0) if qtype in ['number', 'integer'] else ([] if qtype == 'array' else ""))
            
            return answers, metadata
    
    def evaluate_acceptance_criteria(self, image_path: str, original_description: str, 
                                     criteria: List[Dict], question_answers: Dict) -> Dict:
        """Evaluate image against acceptance criteria."""
        image_b64 = self._image_to_base64(image_path)
        
        # Ensure original_description is not None
        if original_description is None:
            original_description = "Original image description unavailable."
        
        criteria_text = "\n".join([
            f"- {c['field']}: {c['question']} (type: {c['type']}, min: {c['min']}, max: {c['max']})"
            for c in criteria
        ])
        
        questions_summary = "\n".join([
            f"- {k}: {v}"
            for k, v in question_answers.items()
        ])
        
        prompt = self.prompts['evaluate_acceptance_criteria'].format(
            original_description=original_description,
            questions_summary=questions_summary,
            criteria_text=criteria_text
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json"
        }
        
        try:
            result, metadata = self._make_request_with_retry(payload, parse_json=True)
            # Add metadata to result
            result["_metadata"] = metadata
            return result
        except Exception as e:
            # Fallback evaluation
            return {
                "overall_score": 0,
                "criteria_results": {c['field']: False for c in criteria},
                "_metadata": {
                    "provider": self.provider,
                    "model": self.model,
                    "using_fallback": self._using_fallback,
                    "success": False,
                    "error": str(e)
                }
            }
    
    def compare_images(self, original_path: str, generated_path: str) -> Dict:
        """Compare original and generated images."""
        img1_b64 = self._image_to_base64(original_path)
        img2_b64 = self._image_to_base64(generated_path)
        
        prompt = self.prompts['compare_images']
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img1_b64, img2_b64],
            "stream": False,
            "format": "json"
        }
        
        try:
            result, metadata = self._make_request_with_retry(payload, parse_json=True)
            # Add metadata to result
            result["_metadata"] = metadata
            return result
        except Exception as e:
            # Fallback comparison
            return {
                "similarity_score": 0.5,
                "differences": [],
                "analysis": f"Comparison failed: {str(e)}",
                "_metadata": {
                    "provider": self.provider,
                    "model": self.model,
                    "using_fallback": self._using_fallback,
                    "success": False,
                    "error": str(e)
                }
            }
    
    def describe_image(self, image_path: str) -> str:
        """Describe an image (cached for original image)."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Converting image to base64: {image_path}")
        image_b64 = self._image_to_base64(image_path)
        logger.debug(f"Image converted, base64 length: {len(image_b64)}")
        
        prompt = self.prompts['describe_image']
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }
        
        logger.info(f"Requesting image description from {self.model} (this may take 30-180 seconds)...")
        logger.info(f"  Image size: {len(image_b64)} chars base64")
        try:
            result, metadata = self._make_request_with_retry(payload, parse_json=False, log_progress=True)
            logger.info(f"  Description received ({len(result)} chars)")
            logger.info(f"  Provider: {metadata.get('provider')}, Model: {metadata.get('model')}, Fallback: {metadata.get('using_fallback')}")
            # Store metadata for later retrieval
            self._last_request_metadata = metadata
            return result
        except Exception as e:
            logger.error(f"Failed to describe image: {e}")
            logger.error(f"  This can happen if:")
            logger.error(f"    - Ollama is busy with another request")
            logger.error(f"    - Vision model is slow (try a smaller image)")
            logger.error(f"    - Network timeout (increase timeout in code)")
            # Store error metadata
            self._last_request_metadata = {
                "provider": self.provider,
                "model": self.model,
                "using_fallback": self._using_fallback,
                "success": False,
                "error": str(e)
            }
            return f"Failed to describe image: {str(e)}"

    def generate_base_prompts(self, original_description: str, scope: str, active_criteria_text: str) -> Tuple[Dict, Dict]:
        """Generate a clean starting positive/negative prompt from description + active criteria tags.

        Returns:
            (result, metadata) where result is a dict with keys: "positive", "negative"
        """
        original_description = original_description or "Original image description unavailable."
        scope = scope or "global"
        active_criteria_text = active_criteria_text or ""

        if "generate_base_prompts" not in self.prompts:
            raise ValueError("Missing prompt template: generate_base_prompts")

        prompt = self.prompts["generate_base_prompts"].format(
            original_description=original_description,
            scope=scope,
            active_criteria_text=active_criteria_text,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        parsed, metadata = self._make_request_with_retry(payload, parse_json=True)
        self._last_request_metadata = metadata
        return parsed, metadata
    
    def improve_prompts(self, current_positive: str, current_negative: str,
                       evaluation: Dict, comparison: Dict, failed_criteria: List[str],
                       rules_text: str) -> Tuple[str, str]:
        """Generate improved prompts based on evaluation and comparison."""
        # Format differences and analysis for prompt
        differences = ', '.join(comparison.get('differences', [])[:5])
        analysis = comparison.get('analysis', '')[:200]
        # Some providers may return numeric fields as strings; normalise for templating.
        raw_sim = comparison.get('similarity_score', 0.5)
        try:
            sim_val = float(raw_sim)
        except (TypeError, ValueError):
            sim_val = 0.5
        # Pass both a float (works with {similarity_score:.2f}) and a preformatted string (works with {similarity_score_str})
        similarity_score = sim_val
        similarity_score_str = f"{sim_val:.2f}"
        
        prompt = self.prompts['improve_prompts'].format(
            current_positive=current_positive,
            current_negative=current_negative,
            overall_score=evaluation.get('overall_score', 0),
            failed_criteria=', '.join(failed_criteria),
            similarity_score=similarity_score,
            similarity_score_str=similarity_score_str,
            differences=differences,
            analysis=analysis,
            rules_text=rules_text
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            parsed, metadata = self._make_request_with_retry(payload, parse_json=True)
            # Store metadata for later retrieval
            self._last_request_metadata = metadata
            return (
                parsed.get('positive', current_positive),
                parsed.get('negative', current_negative)
            )
        except Exception:
            return current_positive, current_negative
    
    def test_connection(self) -> bool:
        """Test if the vision service is accessible."""
        try:
            if self.provider == "openrouter":
                # Test OpenRouter connection
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(f"{self.base_url}/models", headers=headers, timeout=5)
                return response.status_code == 200
            else:  # ollama
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
        except:
            return False

    def generate_text(self, prompt: str) -> Tuple[str, Dict]:
        """Run a text-only prompt against the configured provider/model.

        Useful for non-vision tasks (e.g., rules.yaml checking) while reusing routing,
        retry, and metadata capture.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        result, metadata = self._make_request_with_retry(payload, parse_json=False)
        self._last_request_metadata = metadata
        return str(result), metadata

    def generate_json(self, prompt: str) -> Tuple[Dict, Dict]:
        """Run a text-only prompt and parse a JSON object from the response."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        parsed, metadata = self._make_request_with_retry(payload, parse_json=True)
        self._last_request_metadata = metadata
        # Ensure dict output
        if isinstance(parsed, dict):
            return parsed, metadata
        return {"result": parsed}, metadata

    def suggest_rules_tags(self, rules_yaml_text: str, guidelines_text: str = "") -> Tuple[Dict, Dict]:
        """Suggest improvements to acceptance_criteria tags in rules.yaml.

        This is a text-only call; it does not require images.
        """
        prompt = self.prompts.get("suggest_rules_tags", "").format(
            rules_yaml=rules_yaml_text,
            guidelines=guidelines_text
        )
        return self.generate_json(prompt)
