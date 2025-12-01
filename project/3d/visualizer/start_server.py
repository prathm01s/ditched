#!/usr/bin/env python3
"""
Simple HTTP server to run the QuickHull 3D Visualizer
"""
import http.server
import socketserver
import os
import sys
import socket

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        # Print requests for debugging
        print(f"[{self.log_date_time_string()}] {format % args}")

def find_free_port(start_port=8000, max_attempts=20):
    """Find a free port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    # Change to the visualizer directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Find a free port
    port = find_free_port(PORT)
    if port is None:
        print(f"Error: Could not find an available port starting from {PORT}")
        print("Please close other servers or try again later.")
        sys.exit(1)
    
    Handler = MyHTTPRequestHandler
    
    try:
        # Allow address reuse to prevent "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", port), Handler)
        # Set socket options for better connection handling
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        httpd.timeout = None  # No timeout
        try:
            print("=" * 60)
            print("QuickHull 3D Visualizer Server")
            print("=" * 60)
            print(f"\nServer running at:")
            print(f"  http://localhost:{port}")
            print(f"  http://127.0.0.1:{port}")
            print(f"\nServing directory: {os.getcwd()}")
            print(f"\nOpen your browser and navigate to the URL above")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 60)
            print()
            
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
        finally:
            httpd.shutdown()
            httpd.server_close()
    except OSError as e:
        print(f"\nError starting server: {e}")
        if "Only one usage" in str(e) or e.errno == 10048:
            print(f"\nPort {port} is already in use.")
            print("Please close the other server or wait a moment and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
