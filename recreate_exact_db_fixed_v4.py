#!/usr/bin/env python3
"""
Recreate database with ALL fixes applied:
1. Fixed Vandah API KeyError for 'name' field
2. Made stations.name nullable 
3. Fixed background scheduler water_levels INSERT
4. All previous fixes from v3
"""

import sqlite3
import bcrypt
from datetime import datetime

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_all_tables():
    """Initialize all database tables with ALL fixes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🗄️ Initializing all database tables with ALL fixes...")
    
    # FIRST: Drop all existing tables in correct order
    print("🗑️ Dropping existing tables...")
    tables_to_drop = [
        'station_subscriptions',
        'subscriptions', 
        'min_max_values',
        'last_30_days_historical',
        'predictions',
        'water_levels',
        'stations',
        'municipalities',
        'users'
    ]
    
    for table in tables_to_drop:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
        print(f"🗑️ Dropped table: {table}")
    
    print("✅ All existing tables dropped")
    
    # SECOND: Create all tables with ALL fixes
    print("🏗️ Creating new tables with ALL fixes...")
    
    # Create users table with proper constraints
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
    print("✅ Users table created")
    
    # Create municipalities table with UNIQUE name constraint
    cursor.execute('''
        CREATE TABLE municipalities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            region TEXT,
            population INTEGER,
            area_km2 REAL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT
        )
    ''')
    print("✅ Municipalities table created with UNIQUE name")
    
    # Create stations table with NULLABLE name field (FIX #2)
    cursor.execute('''
        CREATE TABLE stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT UNIQUE NOT NULL,
            name TEXT,
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
    print("✅ Stations table created with NULLABLE name field")
    
    # Create water_levels table with proper structure
    cursor.execute('''
        CREATE TABLE water_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            level_cm REAL,
            timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            water_level_cm REAL,
            water_level_m REAL,
            measurement_date TIMESTAMP
        )
    ''')
    print("✅ Water levels table created")
    
    # Create predictions table with nullable predicted_level_cm
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
    print("✅ Predictions table created with nullable predicted_level_cm")
    
    # Create last_30_days_historical table
    cursor.execute('''
        CREATE TABLE last_30_days_historical (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            measurement_date DATE NOT NULL,
            water_level_cm REAL,
            water_level_m REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Last 30 days historical table created")
    
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
    
    # Create default users with proper password hashing
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
    
    print("✅ Default users created with proper password hashing")
    
    conn.commit()
    conn.close()
    
    print("🎉 All tables initialized successfully with ALL fixes!")
    print("✅ Fixes applied:")
    print("  - Vandah API KeyError for 'name' field (FIX #1)")
    print("  - Stations.name field made nullable (FIX #2)")  
    print("  - Background scheduler water_levels INSERT fixed (FIX #3)")
    print("  - All previous fixes from v3 included")

if __name__ == "__main__":
    init_all_tables()
