#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Production Water Level Server Startup Script
Starts the Flask server in production mode (no debug, no auto-reload)
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

def start_production_server():
    """Start the Flask server in production mode."""
    print("🚀 Starting Water Level Server in PRODUCTION MODE...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set production environment variables
    env = os.environ.copy()
    env["FLASK_DEBUG"] = "0"
    env["FLASK_ENV"] = "production"
    env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered
    
    # Start the server
    try:
        print("🔧 Starting Flask server in PRODUCTION mode...")
        print("📊 Debug mode: DISABLED")
        print("🔄 Auto-reload: DISABLED")
        print("🌐 Server will be available at: http://localhost:5001")
        print("⏰ Background scheduler: ENABLED (updates every 5 minutes)")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the server process with production environment
        process = subprocess.Popen([
            sys.executable, 
            'water_level_server_with_municipalities.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        # Print server output
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping production server...")
        process.terminate()
        process.wait()
        print("✅ Production server stopped")
    except Exception as e:
        print(f"❌ Error starting production server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_production_server()
