#!/bin/bash
# Start complete production system: Web Server + Background Scheduler

echo "=========================================="
echo "🚀 Starting Complete Production System"
echo "=========================================="
echo ""
echo "This script starts:"
echo "  1. Background Scheduler (updates every 12 hours)"
echo "  2. Gunicorn Web Server (4 workers)"
echo ""
echo "Both processes will run in the background."
echo "Use './stop_production.sh' to stop all services"
echo ""

# Kill any existing processes
pkill -f "python3 background_scheduler.py" 2>/dev/null
pkill -f "gunicorn.*app:create_app" 2>/dev/null

echo "Starting Background Scheduler..."
nohup python3 background_scheduler.py > scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "✅ Background Scheduler started (PID: $SCHEDULER_PID)"

sleep 2

echo "Starting Gunicorn Web Server..."
nohup bash run_production_gunicorn.sh > gunicorn.log 2>&1 &
GUNICORN_PID=$!
echo "✅ Gunicorn Web Server started (PID: $GUNICORN_PID)"

echo ""
echo "=========================================="
echo "✅ All services started successfully!"
echo "=========================================="
echo ""
echo "Service Status:"
echo "  • Background Scheduler: PID $SCHEDULER_PID (logs: scheduler.log)"
echo "  • Gunicorn Web Server: PID $GUNICORN_PID (logs: gunicorn.log)"
echo ""
echo "View logs:"
echo "  • tail -f scheduler.log"
echo "  • tail -f gunicorn.log"
echo ""
echo "Stop services:"
echo "  • ./stop_production.sh"
echo ""

