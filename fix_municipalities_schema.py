#!/usr/bin/env python3
"""
Fix municipalities table schema to include missing columns
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fix_municipalities_schema():
    """Add missing columns to municipalities table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing municipalities table schema...")
    
    # Check if columns exist and add them if they don't
    cursor.execute("PRAGMA table_info(municipalities)")
    columns = [row[1] for row in cursor.fetchall()]
    
    required_columns = {
        'region': 'TEXT',
        'population': 'INTEGER',
        'area_km2': 'REAL'
    }
    
    for column, column_type in required_columns.items():
        if column not in columns:
            print(f"➕ Adding column: {column}")
            cursor.execute(f"ALTER TABLE municipalities ADD COLUMN {column} {column_type}")
        else:
            print(f"✅ Column {column} already exists")
    
    conn.commit()
    conn.close()
    print("🎉 Municipalities schema fixed!")

if __name__ == "__main__":
    fix_municipalities_schema()
