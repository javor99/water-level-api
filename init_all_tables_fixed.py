#!/usr/bin/env python3
"""
Initialize all database tables with complete schema
"""

import sqlite3
import bcrypt

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_all_tables():
    """Initialize all database tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🗄️  Initializing all database tables...")
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
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
    
    # Create municipalities table with ALL required columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS municipalities (
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
        CREATE TABLE IF NOT EXISTS stations (
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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Stations table created")
    
    # Create water_levels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS water_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            level_cm REAL NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Water levels table created")
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            predicted_level_cm REAL NOT NULL,
            confidence_score REAL,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Predictions table created")
    
    # Create min_max_values table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS min_max_values (
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
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            station_id TEXT NOT NULL,
            threshold_cm REAL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Subscriptions table created")
    
    # Create default users
    print("👤 Creating default users...")
    
    # Superadmin user
    superadmin_password = "12345678"
    superadmin_hash = bcrypt.hashpw(superadmin_password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute('''
        INSERT OR IGNORE INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('superadmin@superadmin.com', superadmin_hash, 'superadmin'))
    
    # Admin user
    admin_password = "12345678"
    admin_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute('''
        INSERT OR IGNORE INTO users (email, password_hash, role)
        VALUES (?, ?, ?)
    ''', ('admin@admin.com', admin_hash, 'admin'))
    
    print("✅ Default users created")
    
    conn.commit()
    conn.close()
    print("🎉 All tables initialized successfully!")

if __name__ == "__main__":
    init_all_tables()
