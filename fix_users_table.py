#!/usr/bin/env python3
"""
Fix the users table structure to match init_db.py
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

def fix_users_table():
    """Fix the users table structure and recreate users."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing users table structure...")
    
    # Drop and recreate users table with correct structure
    cursor.execute('DROP TABLE IF EXISTS users')
    
    # Create users table with CORRECT structure (like init_db.py)
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
    print("✅ Users table recreated with correct structure")
    
    # Create default users with correct password hashing
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
    
    print("🎉 Users table fixed!")
    print("✅ You can now login with:")
    print("   - superadmin@superadmin.com / 12345678")
    print("   - admin@admin.com / 12345678")

if __name__ == "__main__":
    fix_users_table()
