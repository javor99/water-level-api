#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update database to add municipalities table and municipality_id to stations
"""

import sqlite3
import json

def update_database():
    """Add municipalities table and update stations table."""
    conn = sqlite3.connect('water_levels.db')
    cursor = conn.cursor()
    
    print("Creating municipalities table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS municipalities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
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
    
    print("Adding municipality_id column to stations table...")
    try:
        cursor.execute('ALTER TABLE stations ADD COLUMN municipality_id INTEGER')
        print("Added municipality_id column to stations table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("municipality_id column already exists")
        else:
            raise e
    
    print("Adding foreign key constraint...")
    try:
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_stations_municipality 
            ON stations(municipality_id)
        ''')
    except sqlite3.OperationalError:
        pass
    
    # Create Høje-Taastrup municipality
    print("Creating Høje-Taastrup municipality...")
    cursor.execute('''
        INSERT OR IGNORE INTO municipalities 
        (name, region, population, area_km2, description, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        'Høje-Taastrup',
        'Capital Region of Denmark',
        50000,
        78.41,
        'Municipality in the Capital Region of Denmark, located west of Copenhagen',
        'system'
    ))
    
    # Get the municipality ID
    cursor.execute('SELECT id FROM municipalities WHERE name = ?', ('Høje-Taastrup',))
    municipality = cursor.fetchone()
    if municipality:
        municipality_id = municipality[0]
        print(f"Høje-Taastrup municipality created with ID: {municipality_id}")
        
        # Update all existing stations to belong to Høje-Taastrup
        print("Associating all existing stations with Høje-Taastrup municipality...")
        cursor.execute('''
            UPDATE stations 
            SET municipality_id = ?
            WHERE municipality_id IS NULL
        ''', (municipality_id,))
        
        updated_count = cursor.rowcount
        print(f"Updated {updated_count} stations to belong to Høje-Taastrup municipality")
    
    conn.commit()
    conn.close()
    print("Database update completed successfully!")

if __name__ == "__main__":
    update_database()
