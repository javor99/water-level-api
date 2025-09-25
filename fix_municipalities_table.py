#!/usr/bin/env python3
"""
Fix the municipalities table to add UNIQUE constraint on name field
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fix_municipalities_table():
    """Fix the municipalities table structure."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing municipalities table structure...")
    
    # Check if name column has UNIQUE constraint
    cursor.execute("PRAGMA table_info(municipalities)")
    columns = cursor.fetchall()
    
    # Check if name is unique
    cursor.execute("PRAGMA index_list(municipalities)")
    indexes = cursor.fetchall()
    
    has_unique_name = False
    for index in indexes:
        cursor.execute(f"PRAGMA index_info({index[1]})")
        index_columns = cursor.fetchall()
        if len(index_columns) == 1 and index_columns[0][2] == 'name' and index[2] == 1:
            has_unique_name = True
            break
    
    if not has_unique_name:
        print("🔧 Adding UNIQUE constraint to municipalities.name...")
        
        # Create new table with UNIQUE constraint
        cursor.execute('''
            CREATE TABLE municipalities_new (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                region TEXT,
                population INTEGER,
                area_km2 REAL,
                description TEXT,
                created_by TEXT,
                updated_by TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        # Copy data from old table
        cursor.execute('''
            INSERT INTO municipalities_new 
            SELECT * FROM municipalities
        ''')
        
        # Drop old table and rename new one
        cursor.execute('DROP TABLE municipalities')
        cursor.execute('ALTER TABLE municipalities_new RENAME TO municipalities')
        
        print("✅ Added UNIQUE constraint to municipalities.name")
    else:
        print("✅ municipalities.name already has UNIQUE constraint")
    
    conn.commit()
    conn.close()
    
    print("🎉 Municipalities table fixed!")
    print("✅ You can now create municipalities without 500 errors")

if __name__ == "__main__":
    fix_municipalities_table()
