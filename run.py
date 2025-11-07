#!/usr/bin/env python3
"""
Main application entry point - Modular Architecture
Replaces the monolithic water_level_server_with_municipalities.py
"""
import os
from app import create_app
from app.config import HOST, PORT

if __name__ == '__main__':
    app = create_app()
    
    print("\n" + "="*60)
    print("🌊 Water Level Predictions API Server - Modular Architecture")
    print("="*60)
    print("\n📋 API Information:")
    print("  • Version: 2.0.0 (Modular)")
    print("  • Architecture: Blueprint-based with separated concerns")
    print("  • Database: SQLite (water_levels.db)")
    print("  • Background Jobs: Auto-update every 5 minutes")
    print("\n🔐 Default Users:")
    print("  • superadmin@superadmin.com (password: 12345678)")
    print("  • admin@admin.com (password: 12345678)")
    print("\n📍 Server Configuration:")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    
    # Check environment variables for debug mode
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"  • Debug mode: {'ENABLED ⚠️' if debug_mode else 'DISABLED ✓'}")
    
    print("\n🚀 Starting server...")
    print(f"  • URL: http://{HOST}:{PORT}")
    print(f"  • API Docs: http://{HOST}:{PORT}/")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host=HOST, port=PORT, debug=debug_mode)



