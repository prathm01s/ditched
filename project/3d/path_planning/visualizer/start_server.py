"""
Simple HTTP server for serving the visualization files.

This script starts a local web server to serve the Babylon.js visualization.
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

PORT = 8000


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler to set CORS headers."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()


def start_server(port: int = PORT):
    """
    Start the HTTP server.
    
    Args:
        port: Port number to use (default: 8000)
    """
    # Change to the visualizer directory
    visualizer_dir = Path(__file__).parent
    os.chdir(visualizer_dir)
    
    Handler = MyHTTPRequestHandler
    
    # Try to find an available port
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
            break
        except OSError as e:
            if attempt < max_attempts - 1:
                print(f"Port {port} is in use, trying port {port + 1}...")
                port += 1
            else:
                print(f"Error: Could not find an available port after {max_attempts} attempts.")
                print(f"Last error: {e}")
                print(f"\nPlease:")
                print(f"  1. Close any other servers using port {PORT} or higher")
                print(f"  2. Or specify a different port: python start_server.py <port>")
                return
    
    url = f"http://localhost:{port}/index.html"
    print(f"Server started at {url}")
    print("Press Ctrl+C to stop the server")
    
    # Try to open browser automatically
    try:
        webbrowser.open(url)
    except:
        print(f"Please open {url} in your browser")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {PORT}")
    
    start_server(port)

