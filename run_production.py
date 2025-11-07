#!/usr/bin/env python3
"""
Production Server Startup - Modular Architecture
Starts the refactored Flask server in production mode (no debug, no auto-reload)
"""
import os
import sys
from pathlib import Path
from app import create_app
from app.config import HOST, PORT


def check_requirements():
    """Check if required files exist."""
    required_files = ['water_levels.db', 'app/']
    missing = []
    
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files/directories: {', '.join(missing)}")
        return False
    
    print("✅ All required files found")
    return True


def start_production_server():
    """Start the Flask server in production mode."""
    print("=" * 70)
    print("🚀 STARTING WATER LEVEL API SERVER - PRODUCTION MODE")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Set production environment
    os.environ["FLASK_ENV"] = "production"
    os.environ["FLASK_DEBUG"] = "0"
    
    print("\n📋 Configuration:")
    print(f"  • Architecture: Modular (Blueprint-based)")
    print(f"  • Host: {HOST}")
    print(f"  • Port: {PORT}")
    print(f"  • Debug mode: DISABLED ✓")
    print(f"  • Auto-reload: DISABLED ✓")
    print(f"  • Environment: PRODUCTION")
    print(f"  • Database: water_levels.db")
    
    print("\n🔐 Default Users:")
    print("  • superadmin@superadmin.com (password: 12345678)")
    print("  • admin@admin.com (password: 12345678)")
    
    print("\n⏰ Background Services:")
    print("  • Background scheduler: ENABLED (updates every 5 minutes)")
    print("  • Email notifications: ENABLED")
    print("  • Data auto-update: ENABLED")
    
    print("\n🌐 Server Starting...")
    print(f"  • URL: http://{HOST}:{PORT}")
    print(f"  • API Documentation: http://{HOST}:{PORT}/")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    try:
        # Create and run the app
        app = create_app()
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping production server...")
        print("✅ Server stopped gracefully")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error starting production server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_production_server()



