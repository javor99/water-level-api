#!/usr/bin/env python3
"""
Fix the password hashing for superadmin and admin users
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

def fix_passwords():
    """Fix the password hashing for default users."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🔧 Fixing password hashing for default users...")
    
    # Get the correct password hash
    superadmin_password = "12345678"
    superadmin_hash = hash_password(superadmin_password)
    
    admin_password = "12345678"
    admin_hash = hash_password(admin_password)
    
    # Update superadmin password
    cursor.execute('''
        UPDATE users 
        SET password_hash = ? 
        WHERE email = ?
    ''', (superadmin_hash, 'superadmin@superadmin.com'))
    
    # Update admin password
    cursor.execute('''
        UPDATE users 
        SET password_hash = ? 
        WHERE email = ?
    ''', (admin_hash, 'admin@admin.com'))
    
    conn.commit()
    conn.close()
    
    print("✅ Password hashing fixed for superadmin and admin users")
    print("✅ You can now login with:")
    print("   - superadmin@superadmin.com / 12345678")
    print("   - admin@admin.com / 12345678")

if __name__ == "__main__":
    fix_passwords()
