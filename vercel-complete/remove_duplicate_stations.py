#!/usr/bin/env python3
"""
Remove duplicate/test stations from Høje Taastrup municipality
Keep only the original 7 stations: 70000864-70000927
"""

import sqlite3
import requests

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def remove_test_stations():
    """Remove test stations from municipality 1."""
    
    # Stations to keep (original Høje Taastrup stations)
    keep_stations = [
        '70000864', '70000865', '70000923', '70000924', 
        '70000925', '70000926', '70000927'
    ]
    
    # Get all stations in municipality 1
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT station_id, name FROM stations WHERE municipality_id = 1")
    all_stations = cursor.fetchall()
    
    print(f"Found {len(all_stations)} stations in municipality 1:")
    for station in all_stations:
        print(f"  - {station['station_id']}: {station['name']}")
    
    # Identify stations to remove
    stations_to_remove = []
    for station in all_stations:
        if station['station_id'] not in keep_stations:
            stations_to_remove.append(station['station_id'])
    
    print(f"\nStations to remove ({len(stations_to_remove)}):")
    for station_id in stations_to_remove:
        cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
        name = cursor.fetchone()['name']
        print(f"  - {station_id}: {name}")
    
    if not stations_to_remove:
        print("No stations to remove!")
        conn.close()
        return
    
    # Confirm removal
    print(f"\n⚠️  This will remove {len(stations_to_remove)} stations from municipality 1.")
    print("Stations to keep:")
    for station_id in keep_stations:
        cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
        name = cursor.fetchone()['name']
        print(f"  ✅ {station_id}: {name}")
    
    response = input("\nProceed with removal? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        conn.close()
        return
    
    # Remove stations
    removed_count = 0
    for station_id in stations_to_remove:
        try:
            # Remove from stations table
            cursor.execute("DELETE FROM stations WHERE station_id = ?", (station_id,))
            
            # Remove from water_levels table
            cursor.execute("DELETE FROM water_levels WHERE station_id = ?", (station_id,))
            
            # Remove from last_30_days_historical table
            cursor.execute("DELETE FROM last_30_days_historical WHERE station_id = ?", (station_id,))
            
            # Remove from predictions table
            cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
            
            removed_count += 1
            print(f"✅ Removed station {station_id}")
            
        except Exception as e:
            print(f"❌ Error removing station {station_id}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Successfully removed {removed_count} stations!")
    print(f"Municipality 1 now has {len(keep_stations)} stations (the original Høje Taastrup stations).")

if __name__ == "__main__":
    remove_test_stations()
