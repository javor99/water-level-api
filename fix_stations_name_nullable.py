#!/usr/bin/env python3
"""Make stations.name field nullable"""

import sqlite3

# Connect to database
conn = sqlite3.connect('water_levels.db')
cursor = conn.cursor()

print("🔧 Making stations.name field nullable...")

# SQLite doesn't support ALTER COLUMN directly, so we need to recreate the table
# First, get the current schema
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='stations'")
schema = cursor.fetchone()[0]
print(f"📋 Current schema: {schema}")

# Create a new table with nullable name
cursor.execute("""
CREATE TABLE stations_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT UNIQUE NOT NULL,
    name TEXT,
    latitude REAL,
    longitude REAL,
    location_type TEXT DEFAULT 'stream',
    station_owner TEXT,
    municipality_id INTEGER,
    created_by TEXT,
    updated_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_30_days_min_cm REAL,
    last_30_days_max_cm REAL,
    last_30_days_min_m REAL,
    last_30_days_max_m REAL
)
""")

# Copy data from old table to new table
cursor.execute("""
INSERT INTO stations_new 
SELECT * FROM stations
""")

# Drop old table and rename new table
cursor.execute("DROP TABLE stations")
cursor.execute("ALTER TABLE stations_new RENAME TO stations")

conn.commit()
conn.close()

print("✅ Stations.name field is now nullable")
