# Database Migrations

This directory contains database migration scripts to safely update the database schema without losing data.

## Running Migrations

### Run a specific migration:
```bash
python3 migrations/001_add_last_alert_sent_at.py
```

### Run all pending migrations:
```bash
python3 migrations/run_all_migrations.py
```

## Migration List

### 001_add_last_alert_sent_at.py
- **Date**: 2025-01-15
- **Description**: Adds `last_alert_sent_at` column to `station_subscriptions` table
- **Purpose**: Enables once-per-day alert limit feature
- **Safe**: Yes - only adds a new column, doesn't modify existing data

## Migration Safety

All migrations are designed to be:
- **Idempotent**: Can be run multiple times safely
- **Non-destructive**: Don't delete or modify existing data
- **Reversible**: Include rollback instructions (where possible)

## Notes

- SQLite has limitations on ALTER TABLE operations
- Some migrations cannot be fully rolled back (SQLite doesn't support DROP COLUMN)
- Always backup your database before running migrations in production

