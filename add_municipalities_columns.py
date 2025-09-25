#!/usr/bin/env python3
"""
Add ONLY the missing columns to municipalities table
Does NOT drop or recreate tables - just adds missing columns
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_missing_columns():
    """Add only the missing columns to municipalities table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Adding missing columns to municipalities table...")
    
    # Add missing columns to municipalities table
    try:
        cursor.execute('ALTER TABLE municipalities ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        print("✅ Added created_at to municipalities")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ created_at already exists in municipalities")
        else:
            print(f"❌ Error adding created_at: {e}")
    
    try:
        cursor.execute('ALTER TABLE municipalities ADD COLUMN created_by TEXT')
        print("✅ Added created_by to municipalities")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ created_by already exists in municipalities")
        else:
            print(f"❌ Error adding created_by: {e}")
    
    try:
        cursor.execute('ALTER TABLE municipalities ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        print("✅ Added updated_at to municipalities")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ updated_at already exists in municipalities")
        else:
            print(f"❌ Error adding updated_at: {e}")
    
    try:
        cursor.execute('ALTER TABLE municipalities ADD COLUMN updated_by TEXT')
        print("✅ Added updated_by to municipalities")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ updated_by already exists in municipalities")
        else:
            print(f"❌ Error adding updated_by: {e}")
    
    conn.commit()
    conn.close()
    print("🎉 All missing columns added to municipalities table!")
    print("✅ Added: created_at, created_by, updated_at, updated_by")

if __name__ == "__main__":
    add_missing_columns()
