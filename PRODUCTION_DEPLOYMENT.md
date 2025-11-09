# Production Deployment Guide

## The Problem (FIXED)
When running with gunicorn's 4 workers, the background scheduler was starting 4 times (once per worker), causing **4 duplicate alert emails**.

## The Solution
The background scheduler now runs as a **separate process** from the web server.

## How to Start Production System

### Option 1: Start Everything at Once (RECOMMENDED)
```bash
./start_all_production.sh
```
This starts:
- Background Scheduler (runs every 12 hours, sends alerts)
- Gunicorn Web Server (4 workers, serves API)

Both run as background processes with logs.

### Option 2: Start Services Separately
```bash
# Terminal 1 - Background Scheduler
./start_scheduler.sh

# Terminal 2 - Web Server
./run_production_gunicorn.sh
```

## How to Stop Production System
```bash
./stop_production.sh
```

## View Logs
```bash
# Scheduler logs
tail -f scheduler.log
tail -f background_scheduler.log

# Web server logs
tail -f gunicorn.log
tail -f server.log
```

## Service Details

### Background Scheduler
- **File**: `background_scheduler.py`
- **Function**: Updates water levels and predictions every 12 hours
- **Alerts**: Sends email alerts when thresholds are exceeded
- **Log**: `scheduler.log` and `background_scheduler.log`

### Web Server
- **File**: `run_production_gunicorn.sh`
- **Workers**: 4 (for handling concurrent requests)
- **Port**: 5001
- **Log**: `gunicorn.log` and `server.log`

## Important Notes
⚠️ **DO NOT** run `run_production_gunicorn.sh` without running the scheduler separately!
⚠️ Each gunicorn worker creates its own Flask app - never start the scheduler inside `create_app()`

## Verification
After starting, verify only ONE scheduler is running:
```bash
ps aux | grep background_scheduler.py
```
You should see only ONE process (plus the grep command itself).

## Architecture
```
┌─────────────────────────────────────┐
│   start_all_production.sh           │
└──────────┬──────────────────────────┘
           │
           ├─► Background Scheduler (1 process)
           │   └─► Updates data every 12h
           │   └─► Sends alert emails
           │
           └─► Gunicorn Web Server
               └─► Worker 1 ─► Flask App
               └─► Worker 2 ─► Flask App
               └─► Worker 3 ─► Flask App
               └─► Worker 4 ─► Flask App
```

Each Flask app serves API requests, but **ONLY ONE** scheduler runs to prevent duplicates.

