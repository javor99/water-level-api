#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Water Level Server Startup Program
Starts the Flask server with proper configuration
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_files = [
        'water_level_server_with_municipalities.py',
        'water_levels.db'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def start_server():
    """Start the Flask server."""
    print("🚀 Starting Water Level Server...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the server
    try:
        print("Starting Flask server on port 5001...")
        print("Server will be available at: http://localhost:5001")
        print("Press Ctrl+C to stop the server")
        print()
        
        # Start the server process
        # Set environment variable to disable debug mode
        env = os.environ.copy()
        env["FLASK_DEBUG"] = "0"
        env["FLASK_ENV"] = "production"
        process = subprocess.Popen([
            sys.executable, 
            'water_level_server_with_municipalities.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # Print server output
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
