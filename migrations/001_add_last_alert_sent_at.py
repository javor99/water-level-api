#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration 001: Add last_alert_sent_at column to station_subscriptions table

This migration adds the last_alert_sent_at column to track when alerts were last sent,
enabling the once-per-day alert limit feature.

Date: 2025-01-15
"""

import sqlite3
import os
from datetime import datetime


def get_db_connection(db_path='water_levels.db'):
    """Create a database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate():
    """Run the migration."""
    db_path = 'water_levels.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return False
    
    print("=" * 70)
    print("🔄 Migration 001: Add last_alert_sent_at column")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='station_subscriptions'
        """)
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("❌ Table 'station_subscriptions' does not exist!")
            print("   Please run init_db.py first to create the database schema.")
            conn.close()
            return False
        
        # Check if column already exists
        if check_column_exists(cursor, 'station_subscriptions', 'last_alert_sent_at'):
            print("✅ Column 'last_alert_sent_at' already exists in 'station_subscriptions' table")
            print("   Migration already applied - skipping.")
            conn.close()
            return True
        
        # Add the column
        print("📝 Adding 'last_alert_sent_at' column to 'station_subscriptions' table...")
        cursor.execute("""
            ALTER TABLE station_subscriptions 
            ADD COLUMN last_alert_sent_at TIMESTAMP
        """)
        
        conn.commit()
        
        # Verify the column was added
        if check_column_exists(cursor, 'station_subscriptions', 'last_alert_sent_at'):
            print("✅ Successfully added 'last_alert_sent_at' column")
            
            # Show table structure
            cursor.execute("PRAGMA table_info(station_subscriptions)")
            columns = cursor.fetchall()
            print()
            print("📋 Updated table structure:")
            print("   Columns in station_subscriptions:")
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                is_nullable = "NULL" if col[3] == 0 else "NOT NULL"
                print(f"     - {col_name} ({col_type}) {is_nullable}")
            
            conn.close()
            print()
            print("=" * 70)
            print("✅ Migration 001 completed successfully!")
            print("=" * 70)
            return True
        else:
            print("❌ Column was not added successfully")
            conn.rollback()
            conn.close()
            return False
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False


def rollback():
    """Rollback the migration (remove the column)."""
    db_path = 'water_levels.db'
    
    if not os.path.exists(db_path):
        print(f"❌ Database file {db_path} not found!")
        return False
    
    print("=" * 70)
    print("⏪ Rollback Migration 001: Remove last_alert_sent_at column")
    print("=" * 70)
    print("⚠️  WARNING: SQLite does not support DROP COLUMN directly.")
    print("   This rollback requires recreating the table, which will")
    print("   DELETE ALL DATA in station_subscriptions!")
    print()
    
    response = input("Are you sure you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Rollback cancelled")
        return False
    
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if column exists
        if not check_column_exists(cursor, 'station_subscriptions', 'last_alert_sent_at'):
            print("✅ Column 'last_alert_sent_at' does not exist - nothing to rollback")
            conn.close()
            return True
        
        print("⚠️  SQLite limitation: Cannot drop column directly.")
        print("   To remove this column, you would need to:")
        print("   1. Create a new table without the column")
        print("   2. Copy data from old table to new table")
        print("   3. Drop old table and rename new table")
        print()
        print("   This is a destructive operation and not implemented here.")
        print("   If you need to rollback, please restore from a backup.")
        
        conn.close()
        return False
        
    except Exception as e:
        print(f"❌ Error during rollback: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'rollback':
        success = rollback()
    else:
        success = migrate()
    
    sys.exit(0 if success else 1)

