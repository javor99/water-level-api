#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.
# Production server startup using Gunicorn
# This is the PROPER way to run Flask in production

echo "=========================================="
echo "🏭 Starting Production Server (Gunicorn)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • WSGI Server: Gunicorn"
echo "  • Workers: 4"
echo "  • Threads: 2 per worker"
echo "  • Host: 0.0.0.0"
echo "  • Port: 5001"
echo "  • Timeout: 120s"
echo ""

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "❌ Gunicorn is not installed!"
    echo ""
    echo "Install it with:"
    echo "  pip install gunicorn"
    echo ""
    exit 1
fi

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=0

# Start Gunicorn
echo "🚀 Starting Gunicorn..."
echo "   Press Ctrl+C to stop"
echo ""

gunicorn \
    --bind 0.0.0.0:5001 \
    --workers 4 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    "app:create_app()"



