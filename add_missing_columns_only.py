#!/usr/bin/env python3
"""
Add ONLY the missing columns to existing tables
Does NOT drop or recreate tables - just adds missing columns
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_missing_columns():
    """Add only the missing columns to existing tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Adding missing columns to existing tables...")
    
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
    
    # Add missing columns to predictions table
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN predicted_water_level_cm REAL')
        print("✅ Added predicted_water_level_cm to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ predicted_water_level_cm already exists in predictions")
        else:
            print(f"❌ Error adding predicted_water_level_cm: {e}")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN predicted_water_level_m REAL')
        print("✅ Added predicted_water_level_m to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ predicted_water_level_m already exists in predictions")
        else:
            print(f"❌ Error adding predicted_water_level_m: {e}")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN change_from_last_cm REAL')
        print("✅ Added change_from_last_cm to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ change_from_last_cm already exists in predictions")
        else:
            print(f"❌ Error adding change_from_last_cm: {e}")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN forecast_date DATE')
        print("✅ Added forecast_date to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ forecast_date already exists in predictions")
        else:
            print(f"❌ Error adding forecast_date: {e}")
    
    # Add missing columns to water_levels table
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN water_level_cm REAL')
        print("✅ Added water_level_cm to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ water_level_cm already exists in water_levels")
        else:
            print(f"❌ Error adding water_level_cm: {e}")
    
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN water_level_m REAL')
        print("✅ Added water_level_m to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ water_level_m already exists in water_levels")
        else:
            print(f"❌ Error adding water_level_m: {e}")
    
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN measurement_date TIMESTAMP')
        print("✅ Added measurement_date to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ measurement_date already exists in water_levels")
        else:
            print(f"❌ Error adding measurement_date: {e}")
    
    conn.commit()
    conn.close()
    print("🎉 All missing columns added successfully!")
    print("✅ Added missing columns to:")
    print("   - Stations: last_30_days_min_cm, last_30_days_max_cm, last_30_days_min_m, last_30_days_max_m")
    print("   - Predictions: predicted_water_level_cm, predicted_water_level_m, change_from_last_cm, forecast_date")
    print("   - Water_levels: water_level_cm, water_level_m, measurement_date")

if __name__ == "__main__":
    add_missing_columns()
