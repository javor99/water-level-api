#!/usr/bin/env python3
"""
Recreate database EXACTLY as it currently is, just adding missing columns
FIXED: Users table has correct structure like init_db.py
UPDATED: Includes last_30_days_historical table and fixed predictions constraints
"""

import sqlite3
import bcrypt

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def recreate_exact_db():
    """Recreate database exactly as it is, just adding missing columns."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🗄️  Recreating database EXACTLY as current...")
    
    # FIRST: Drop all existing tables
    print("🗑️  Dropping existing tables...")
    
    tables_to_drop = [
        'station_subscriptions',
        'subscriptions', 
        'min_max_values',
        'predictions',
        'water_levels',
        'stations',
        'municipalities',
        'users',
        'last_30_days_historical'  # Add this table to drop list
    ]
    
    for table in tables_to_drop:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
        print(f"🗑️  Dropped table: {table}")
    
    print("✅ All existing tables dropped")
    
    # SECOND: Recreate EXACTLY as current database
    print("🏗️  Recreating tables EXACTLY as current...")
    
    # Create users table with CORRECT structure
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            created_by INTEGER
        )
    ''')
    print("✅ Users table created with CORRECT structure")
    
    # Create municipalities table
    cursor.execute('''
        CREATE TABLE municipalities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            region TEXT,
            population INTEGER,
            area_km2 REAL,
            description TEXT,
            created_by TEXT,
            updated_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Municipalities table created")
    
    # Create stations table
    cursor.execute('''
        CREATE TABLE stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
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
    ''')
    print("✅ Stations table created")
    
    # Create water_levels table
    cursor.execute('''
        CREATE TABLE water_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            level_cm REAL NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            water_level_cm REAL,
            water_level_m REAL,
            measurement_date TIMESTAMP
        )
    ''')
    print("✅ Water levels table created")
    
    # Create predictions table with FIXED constraints (predicted_level_cm nullable)
    cursor.execute('''
        CREATE TABLE predictions (
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
    print("✅ Predictions table created with FIXED constraints")
    
    # Create min_max_values table
    cursor.execute('''
        CREATE TABLE min_max_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            min_level_cm REAL,
            max_level_cm REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Min/Max values table created")
    
    # Create subscriptions table
    cursor.execute('''
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            station_id TEXT NOT NULL,
            threshold_cm REAL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Subscriptions table created")
    
    # Create station_subscriptions table
    cursor.execute('''
        CREATE TABLE station_subscriptions (
            user_email TEXT NOT NULL,
            station_id TEXT NOT NULL,
            threshold_percentage REAL,
            is_active BOOLEAN DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_email, station_id)
        )
    ''')
    print("✅ Station subscriptions table created")
    
    # Create last_30_days_historical table (MISSING TABLE)
    cursor.execute('''
        CREATE TABLE last_30_days_historical (
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
    print("✅ last_30_days_historical table created (MISSING TABLE ADDED)")
    
    # Create indexes for better performance
    print("📈 Creating indexes...")
    try:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_30_days_station_id ON last_30_days_historical(station_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_30_days_date ON last_30_days_historical(measurement_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_station_id ON predictions(station_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)')
        print("✅ Indexes created")
    except Exception as e:
        print(f"⚠️  Could not create indexes: {e}")
    
    # Create default users with CORRECT password hashing
    print("👤 Creating default users...")
    
    # Superadmin user
    superadmin_password = "12345678"
    superadmin_hash = hash_password(superadmin_password)
    cursor.execute('''
        INSERT INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('superadmin@superadmin.com', superadmin_hash, 'superadmin'))
    
    # Admin user
    admin_password = "12345678"
    admin_hash = hash_password(admin_password)
    cursor.execute('''
        INSERT INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('admin@admin.com', admin_hash, 'admin'))
    
    print("✅ Default users created")
    
    conn.commit()
    conn.close()
    print("🎉 Database recreated EXACTLY as current with ALL missing components!")
    print("✅ FIXED: Users table has correct structure")
    print("✅ ADDED: last_30_days_historical table (was missing)")
    print("✅ FIXED: predictions table constraints (predicted_level_cm nullable)")
    print("✅ Added missing columns:")
    print("   - Stations: last_30_days_min_cm, last_30_days_max_cm, last_30_days_min_m, last_30_days_max_m")
    print("   - Predictions: predicted_water_level_cm, predicted_water_level_m, change_from_last_cm, forecast_date")
    print("   - Water_levels: water_level_cm, water_level_m, measurement_date")
    print("✅ You can now login with:")
    print("   - superadmin@superadmin.com / 12345678")
    print("   - admin@admin.com / 12345678")

if __name__ == "__main__":
    recreate_exact_db()
