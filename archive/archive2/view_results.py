#!/usr/bin/env python3
"""
Simple web server to view test results in a browser.
Serves the test_results directory on http://0.0.0.0:9121
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 9121
DIRECTORY = Path(__file__).parent / "test_results"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        # Add CORS headers to allow viewing from anywhere
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        super().end_headers()

def main():
    # Create directory if it doesn't exist
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    if not DIRECTORY.exists():
        print(f"❌ Directory does not exist: {DIRECTORY}")
        return
    
    with socketserver.TCPServer(("0.0.0.0", PORT), MyHTTPRequestHandler) as httpd:
        print("="*60)
        print("Test Results Viewer")
        print("="*60)
        print(f"Serving directory: {DIRECTORY}")
        print(f"Access at: http://localhost:9121")
        print(f"Or from network: http://0.0.0.0:9121")
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

if __name__ == '__main__':
    main()
