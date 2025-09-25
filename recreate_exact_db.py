#!/usr/bin/env python3
"""
Recreate database EXACTLY as it currently is, just adding missing columns
"""

import sqlite3
import bcrypt

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

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
        'users'
    ]
    
    for table in tables_to_drop:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
        print(f"🗑️  Dropped table: {table}")
    
    print("✅ All existing tables dropped")
    
    # SECOND: Recreate EXACTLY as current database
    print("🏗️  Recreating tables EXACTLY as current...")
    
    # Create users table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN,
            created_by INTEGER
        )
    ''')
    print("✅ Users table recreated exactly as current")
    
    # Create municipalities table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE municipalities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
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
    print("✅ Municipalities table recreated exactly as current")
    
    # Create stations table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE stations (
            id INTEGER PRIMARY KEY,
            station_id TEXT NOT NULL,
            name TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            location_type TEXT,
            station_owner TEXT,
            municipality_id INTEGER,
            created_by TEXT,
            updated_by TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    print("✅ Stations table recreated exactly as current")
    
    # Create water_levels table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE water_levels (
            id INTEGER PRIMARY KEY,
            station_id TEXT NOT NULL,
            level_cm REAL NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP
        )
    ''')
    print("✅ Water levels table recreated exactly as current")
    
    # Create predictions table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            station_id TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_level_cm REAL NOT NULL,
            confidence_score REAL,
            model_used TEXT,
            created_at TIMESTAMP
        )
    ''')
    print("✅ Predictions table recreated exactly as current")
    
    # Create min_max_values table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE min_max_values (
            id INTEGER PRIMARY KEY,
            station_id TEXT NOT NULL,
            min_level_cm REAL,
            max_level_cm REAL,
            updated_at TIMESTAMP
        )
    ''')
    print("✅ Min/Max values table recreated exactly as current")
    
    # Create subscriptions table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE subscriptions (
            id INTEGER PRIMARY KEY,
            user_email TEXT NOT NULL,
            station_id TEXT NOT NULL,
            threshold_cm REAL,
            is_active BOOLEAN,
            created_at TIMESTAMP
        )
    ''')
    print("✅ Subscriptions table recreated exactly as current")
    
    # Create station_subscriptions table - EXACTLY as current
    cursor.execute('''
        CREATE TABLE station_subscriptions (
            user_email TEXT NOT NULL,
            station_id TEXT NOT NULL,
            threshold_percentage REAL,
            is_active BOOLEAN,
            updated_at TIMESTAMP,
            PRIMARY KEY (user_email, station_id)
        )
    ''')
    print("✅ Station subscriptions table recreated exactly as current")
    
    # THIRD: Add missing columns to existing tables
    print("🔧 Adding missing columns...")
    
    # Add missing columns to stations table
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_min_cm REAL')
        print("✅ Added last_30_days_min_cm to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_min_cm already exists in stations")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_max_cm REAL')
        print("✅ Added last_30_days_max_cm to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_max_cm already exists in stations")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_min_m REAL')
        print("✅ Added last_30_days_min_m to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_min_m already exists in stations")
    
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN last_30_days_max_m REAL')
        print("✅ Added last_30_days_max_m to stations")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ last_30_days_max_m already exists in stations")
    
    # Add missing columns to predictions table
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN predicted_water_level_cm REAL')
        print("✅ Added predicted_water_level_cm to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ predicted_water_level_cm already exists in predictions")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN predicted_water_level_m REAL')
        print("✅ Added predicted_water_level_m to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ predicted_water_level_m already exists in predictions")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN change_from_last_cm REAL')
        print("✅ Added change_from_last_cm to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ change_from_last_cm already exists in predictions")
    
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN forecast_date DATE')
        print("✅ Added forecast_date to predictions")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ forecast_date already exists in predictions")
    
    # Add missing columns to water_levels table
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN water_level_cm REAL')
        print("✅ Added water_level_cm to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ water_level_cm already exists in water_levels")
    
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN water_level_m REAL')
        print("✅ Added water_level_m to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ water_level_m already exists in water_levels")
    
    try:
        cursor.execute('ALTER TABLE water_levels ADD COLUMN measurement_date TIMESTAMP')
        print("✅ Added measurement_date to water_levels")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ measurement_date already exists in water_levels")
    
    # Create default users
    print("👤 Creating default users...")
    
    # Superadmin user
    superadmin_password = "12345678"
    superadmin_hash = bcrypt.hashpw(superadmin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute('''
        INSERT INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('superadmin@superadmin.com', superadmin_hash, 'superadmin'))
    
    # Admin user
    admin_password = "12345678"
    admin_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute('''
        INSERT INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('admin@admin.com', admin_hash, 'admin'))
    
    print("✅ Default users created")
    
    conn.commit()
    conn.close()
    print("🎉 Database recreated EXACTLY as current with missing columns added!")
    print("✅ Added missing columns:")
    print("   - Stations: last_30_days_min_cm, last_30_days_max_cm, last_30_days_min_m, last_30_days_max_m")
    print("   - Predictions: predicted_water_level_cm, predicted_water_level_m, change_from_last_cm, forecast_date")
    print("   - Water_levels: water_level_cm, water_level_m, measurement_date")

if __name__ == "__main__":
    recreate_exact_db()
