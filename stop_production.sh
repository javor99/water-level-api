#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.
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

