#!/bin/bash

# Production Water Level Server Startup Script
echo "🚀 Starting Water Level Server in PRODUCTION MODE..."
echo "============================================================"

# Set production environment variables
export FLASK_DEBUG=0
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

echo "🔧 Debug mode: DISABLED"
echo "🔄 Auto-reload: DISABLED"
echo "🌐 Server will be available at: http://localhost:5001"
echo "⏰ Background scheduler: ENABLED (updates every 5 minutes)"
echo "============================================================"

# Start the server
python3 water_level_server_with_municipalities.py
