#!/usr/bin/env python3
"""
Fix all missing columns in the database schema
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fix_complete_schema():
    """Add all missing columns to fix the database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing complete database schema...")
    
    # Fix stations table
    print("📋 Fixing stations table...")
    cursor.execute("PRAGMA table_info(stations)")
    station_columns = [row[1] for row in cursor.fetchall()]
    
    if 'id' not in station_columns:
        print("➕ Adding id column to stations")
        cursor.execute("ALTER TABLE stations ADD COLUMN id INTEGER PRIMARY KEY AUTOINCREMENT")
    
    if 'updated_at' not in station_columns:
        print("➕ Adding updated_at column to stations")
        cursor.execute("ALTER TABLE stations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    
    # Fix water_levels table
    print("📋 Fixing water_levels table...")
    cursor.execute("PRAGMA table_info(water_levels)")
    water_columns = [row[1] for row in cursor.fetchall()]
    
    if 'level_cm' not in water_columns:
        print("➕ Adding level_cm column to water_levels")
        cursor.execute("ALTER TABLE water_levels ADD COLUMN level_cm REAL NOT NULL")
    
    if 'timestamp' not in water_columns:
        print("➕ Adding timestamp column to water_levels")
        cursor.execute("ALTER TABLE water_levels ADD COLUMN timestamp TIMESTAMP NOT NULL")
    
    # Fix predictions table
    print("📋 Fixing predictions table...")
    cursor.execute("PRAGMA table_info(predictions)")
    pred_columns = [row[1] for row in cursor.fetchall()]
    
    if 'predicted_level_cm' not in pred_columns:
        print("➕ Adding predicted_level_cm column to predictions")
        cursor.execute("ALTER TABLE predictions ADD COLUMN predicted_level_cm REAL NOT NULL")
    
    if 'confidence_score' not in pred_columns:
        print("➕ Adding confidence_score column to predictions")
        cursor.execute("ALTER TABLE predictions ADD COLUMN confidence_score REAL")
    
    if 'model_used' not in pred_columns:
        print("➕ Adding model_used column to predictions")
        cursor.execute("ALTER TABLE predictions ADD COLUMN model_used TEXT")
    
    conn.commit()
    conn.close()
    print("🎉 Database schema fixed!")

if __name__ == "__main__":
    fix_complete_schema()
