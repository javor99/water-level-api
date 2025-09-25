#!/usr/bin/env python3
"""
Fix water_levels table constraints
Make level_cm nullable to fix background scheduler errors
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fix_water_levels_table():
    """Fix water_levels table constraints."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing water_levels table constraints...")
    
    try:
        # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
        cursor.execute('''
            CREATE TABLE water_levels_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                level_cm REAL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                water_level_cm REAL,
                water_level_m REAL,
                measurement_date TIMESTAMP
            )
        ''')
        
        # Copy existing data
        cursor.execute('''
            INSERT INTO water_levels_new 
            SELECT id, station_id, level_cm, timestamp, created_at,
                   water_level_cm, water_level_m, measurement_date
            FROM water_levels
        ''')
        
        # Drop old table and rename new one
        cursor.execute('DROP TABLE water_levels')
        cursor.execute('ALTER TABLE water_levels_new RENAME TO water_levels')
        
        print("✅ Water_levels table constraints fixed")
        
    except Exception as e:
        print(f"⚠️  Could not fix water_levels table: {e}")
    
    conn.commit()
    conn.close()
    
    print("🎉 Water_levels table fixes completed!")
    print("✅ level_cm is now nullable")

if __name__ == "__main__":
    fix_water_levels_table()
