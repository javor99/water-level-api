#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.
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

