#!/usr/bin/env python3
"""
Add missing columns to stations table
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_missing_columns():
    """Add missing columns to stations table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Adding missing columns to stations table...")
    
    # Add missing columns to stations table
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_min_cm REAL')
        print("✅ Added last_30_days_min_cm to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_min_cm already exists in stations")
        else:
            print(f"❌ Error adding last_30_days_min_cm: {e}")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_max_cm REAL')
        print("✅ Added last_30_days_max_cm to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_max_cm already exists in stations")
        else:
            print(f"❌ Error adding last_30_days_max_cm: {e}")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_min_m REAL')
        print("✅ Added last_30_days_min_m to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_min_m already exists in stations")
        else:
            print(f"❌ Error adding last_30_days_min_m: {e}")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_max_m REAL')
        print("✅ Added last_30_days_max_m to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_max_m already exists in stations")
        else:
            print(f"❌ Error adding last_30_days_max_m: {e}")
    
    conn.commit()
    conn.close()
    
    print("🎉 Missing columns added to stations table!")
    print("✅ You can now create stations without 500 errors")

if __name__ == "__main__":
    add_missing_columns()
