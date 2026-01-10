#!/usr/bin/env python3
"""
ComfyUI API Client

Handles communication with ComfyUI server for workflow execution and image retrieval.
"""

import json
import time
import uuid
from typing import Dict
import requests
import websocket


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""
    
    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws?clientId={uuid.uuid4()}"
    
    def queue_prompt(self, workflow: Dict) -> str:
        """Queue a workflow prompt and return prompt ID."""
        data = {"prompt": workflow}
        response = requests.post(f"{self.base_url}/prompt", json=data, timeout=30)
        response.raise_for_status()
        return response.json()['prompt_id']
    
    def get_history(self, prompt_id: str) -> Dict:
        """Get execution history for a prompt ID.
        
        ComfyUI history API returns:
        - /history/{prompt_id} - returns {prompt_id: {status, outputs, ...}} or {} if not found
        - /history - returns all history {prompt_id: {...}, ...}
        """
        # Try specific prompt_id first
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            data = response.json()
            # If we get a dict with our prompt_id, return it
            if isinstance(data, dict) and prompt_id in data:
                return data
            # If we get an empty dict or different format, try full history
            if not data or not isinstance(data, dict):
                response = requests.get(f"{self.base_url}/history", timeout=30)
                response.raise_for_status()
                all_history = response.json()
                if isinstance(all_history, dict) and prompt_id in all_history:
                    return {prompt_id: all_history[prompt_id]}
            return data
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Prompt not found yet, return empty dict
                return {}
            raise
    
    def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an image from ComfyUI."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        response = requests.get(f"{self.base_url}/view", params=params, timeout=60)
        response.raise_for_status()
        return response.content
    
    def wait_for_completion(self, prompt_id: str, max_wait: int = 300) -> bool:
        """Wait for workflow completion using WebSocket and polling fallback."""
        completed = False
        start_time = time.time()
        ws_connected = False
        
        # Try WebSocket first
        try:
            ws = websocket.WebSocket()
            ws.connect(self.ws_url)
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
                                if percent >= 100.0:
                                    time.sleep(2)
                                    completed = True
                                    break
                
                except websocket.WebSocketTimeoutException:
                    continue
                except Exception:
                    ws_connected = False
                    break
            
            if ws_connected:
                ws.close()
        
        except Exception:
            pass
        
        # Poll history if WebSocket didn't complete
        if not completed:
            poll_interval = 1
            while time.time() - start_time < max_wait:
                try:
                    history = self.get_history(prompt_id)
                    if prompt_id in history:
                        execution = history[prompt_id]
                        outputs = execution.get('outputs', {})
                        if outputs:
                            completed = True
                            break
                except Exception:
                    pass
                
                time.sleep(poll_interval)
                if time.time() - start_time > max_wait:
                    break
        
        return completed
    
    def test_connection(self) -> bool:
        """Test if ComfyUI is accessible."""
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
