#!/bin/bash
# Stop all production services

echo "=========================================="
echo "🛑 Stopping Production Services"
echo "=========================================="
echo ""

echo "Stopping Background Scheduler..."
pkill -f "python3 background_scheduler.py"
echo "✅ Background Scheduler stopped"

echo "Stopping Gunicorn Web Server..."
pkill -f "gunicorn.*app:create_app"
echo "✅ Gunicorn Web Server stopped"

echo ""
echo "=========================================="
echo "✅ All services stopped"
echo "=========================================="
echo ""

