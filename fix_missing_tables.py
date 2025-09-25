#!/usr/bin/env python3
"""
Fix Missing Database Tables and Constraints
Adds the missing last_30_days_historical table and fixes prediction constraints
"""

import sqlite3

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fix_missing_tables():
    """Add missing tables and fix constraints."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing missing database tables and constraints...")
    
    # 1. Create the missing last_30_days_historical table
    print("📊 Creating last_30_days_historical table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS last_30_days_historical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            water_level_cm REAL NOT NULL,
            water_level_m REAL NOT NULL,
            measurement_date TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            min_level_cm REAL,
            max_level_cm REAL
        )
    ''')
    print("✅ last_30_days_historical table created")
    
    # 2. Fix predictions table - make predicted_level_cm nullable
    print("🔧 Fixing predictions table constraints...")
    try:
        # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
        cursor.execute('''
            CREATE TABLE predictions_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_level_cm REAL,
                confidence_score REAL,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_water_level_cm REAL,
                predicted_water_level_m REAL,
                change_from_last_cm REAL,
                forecast_date DATE
            )
        ''')
        
        # Copy existing data
        cursor.execute('''
            INSERT INTO predictions_new 
            SELECT id, station_id, prediction_date, predicted_level_cm, 
                   confidence_score, model_used, created_at,
                   predicted_water_level_cm, predicted_water_level_m, 
                   change_from_last_cm, forecast_date
            FROM predictions
        ''')
        
        # Drop old table and rename new one
        cursor.execute('DROP TABLE predictions')
        cursor.execute('ALTER TABLE predictions_new RENAME TO predictions')
        
        print("✅ Predictions table constraints fixed")
        
    except Exception as e:
        print(f"⚠️  Could not fix predictions table: {e}")
    
    # 3. Create indexes for better performance
    print("📈 Creating indexes...")
    try:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_30_days_station_id ON last_30_days_historical(station_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_30_days_date ON last_30_days_historical(measurement_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_station_id ON predictions(station_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
        print("✅ Indexes created")
    except Exception as e:
        print(f"⚠️  Could not create indexes: {e}")
    
    conn.commit()
    conn.close()
    
    print("🎉 Database fixes completed!")
    print("✅ Added last_30_days_historical table")
    print("✅ Fixed predictions table constraints")
    print("✅ Created performance indexes")

if __name__ == "__main__":
    fix_missing_tables()
