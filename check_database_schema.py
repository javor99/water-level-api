#!/usr/bin/env python3
"""
Check if all required tables and columns exist in the database
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def check_database_schema():
    """Check if all required tables and columns exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔍 Checking database schema...")
    
    # Required tables and their columns
    required_tables = {
        'users': ['id', 'email', 'password_hash', 'role', 'created_at', 'last_login', 'is_active', 'created_by'],
        'municipalities': ['id', 'name', 'region', 'population', 'area_km2', 'description', 'created_by', 'updated_by', 'created_at', 'updated_at'],
        'stations': ['id', 'station_id', 'name', 'latitude', 'longitude', 'location_type', 'station_owner', 'municipality_id', 'created_by', 'updated_by', 'created_at', 'updated_at'],
        'water_levels': ['id', 'station_id', 'level_cm', 'timestamp', 'created_at'],
        'predictions': ['id', 'station_id', 'prediction_date', 'predicted_level_cm', 'confidence_score', 'model_used', 'created_at'],
        'min_max_values': ['id', 'station_id', 'min_level_cm', 'max_level_cm', 'updated_at'],
        'subscriptions': ['id', 'user_email', 'station_id', 'threshold_cm', 'is_active', 'created_at'],
        'station_subscriptions': ['user_email', 'station_id', 'threshold_percentage', 'is_active', 'updated_at']
    }
    
    missing_tables = []
    missing_columns = {}
    
    # Check each table
    for table_name, required_columns in required_tables.items():
        print(f"\n📋 Checking table: {table_name}")
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            print(f"❌ Table {table_name} does not exist")
            missing_tables.append(table_name)
            continue
        
        # Check columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [row[1] for row in cursor.fetchall()]
        
        table_missing_columns = []
        for column in required_columns:
            if column not in existing_columns:
                table_missing_columns.append(column)
        
        if table_missing_columns:
            print(f"❌ Missing columns in {table_name}: {table_missing_columns}")
            missing_columns[table_name] = table_missing_columns
        else:
            print(f"✅ Table {table_name} has all required columns")
    
    conn.close()
    
    # Summary
    print("\n" + "="*50)
    print("📊 SCHEMA CHECK SUMMARY")
    print("="*50)
    
    if missing_tables:
        print(f"❌ Missing tables: {missing_tables}")
    else:
        print("✅ All required tables exist")
    
    if missing_columns:
        print(f"❌ Missing columns:")
        for table, columns in missing_columns.items():
            print(f"   {table}: {columns}")
    else:
        print("✅ All required columns exist")
    
    if not missing_tables and not missing_columns:
        print("🎉 Database schema is complete!")
    else:
        print("⚠️  Database schema needs to be fixed")

if __name__ == "__main__":
    check_database_schema()
