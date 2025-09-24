#!/usr/bin/env python3
"""
Clean up test stations and keep only original ones:
- Høje-Taastrup (municipality_id=1): Keep 7 original stations
- Struer (municipality_id=9): Keep 3 original stations
"""

import sqlite3

def cleanup_stations():
    conn = sqlite3.connect('water_levels.db')
    cursor = conn.cursor()
    
    print("🧹 CLEANING UP TEST STATIONS")
    print("=" * 50)
    
    # Define the stations to keep
    stations_to_keep = {
        # Høje-Taastrup original 7 stations
        1: [
            "70000864",  # Hove å, Tostholm bro
            "70000865",  # Sengeløse å, Sengeløse mose
            "70000923",  # Enghave Å, Rolandsvej 3
            "70000924",  # Ll. Vejleå, Lille Solhøjvej 42
            "70000925",  # Spangå, Ågesholmvej
            "70000926",  # Nybølle Å, Ledøje Plantage
            "70000927",  # Hakkemosegrøften, Ole Rømers Vej
        ],
        # Struer original 3 stations
        9: [
            "70000597",  # Hestbæk, Rødebrovej
            "70000598",  # Hestbæk, Stadsbjergvej
            "70000600",  # Struer Water Station
        ]
    }
    
    # Get all current stations
    cursor.execute("SELECT station_id, name, municipality_id FROM stations")
    all_stations = cursor.fetchall()
    
    print(f"📊 Current stations: {len(all_stations)}")
    
    # Find stations to delete
    stations_to_delete = []
    for station_id, name, municipality_id in all_stations:
        if municipality_id in stations_to_keep:
            if station_id not in stations_to_keep[municipality_id]:
                stations_to_delete.append((station_id, name, municipality_id))
        else:
            # Delete stations from unknown municipalities
            stations_to_delete.append((station_id, name, municipality_id))
    
    print(f"🗑️  Stations to delete: {len(stations_to_delete)}")
    
    if stations_to_delete:
        print("\nStations being deleted:")
        for station_id, name, municipality_id in stations_to_delete:
            print(f"  ❌ {station_id}: {name} (municipality {municipality_id})")
        
        # Delete stations and their related data
        for station_id, name, municipality_id in stations_to_delete:
            print(f"\n🗑️  Deleting station {station_id}: {name}")
            
            # Delete related data
            cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
            cursor.execute("DELETE FROM last_30_days_historical WHERE station_id = ?", (station_id,))
            cursor.execute("DELETE FROM stations WHERE station_id = ?", (station_id,))
            
            print(f"  ✅ Deleted station {station_id} and all related data")
    
    # Show final result
    print(f"\n📊 FINAL RESULT")
    print("=" * 30)
    
    for municipality_id, station_ids in stations_to_keep.items():
        cursor.execute("SELECT name FROM municipalities WHERE id = ?", (municipality_id,))
        municipality_name = cursor.fetchone()[0]
        
        print(f"\n{municipality_name} (ID: {municipality_id}):")
        for station_id in station_ids:
            cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
            result = cursor.fetchone()
            if result:
                print(f"  ✅ {station_id}: {result[0]}")
            else:
                print(f"  ❌ {station_id}: NOT FOUND")
    
    # Commit changes
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Cleanup completed!")
    print(f"✅ Kept {sum(len(stations) for stations in stations_to_keep.values())} original stations")
    print(f"🗑️  Deleted {len(stations_to_delete)} test stations")

if __name__ == "__main__":
    cleanup_stations()
