#!/bin/bash
# Start Background Scheduler (separate from web server)

echo "=========================================="
echo "⏰ Starting Background Scheduler"
echo "=========================================="
echo ""
echo "This scheduler runs independently of the web server"
echo "It updates water levels and sends alerts every 12 hours"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 background_scheduler.py

