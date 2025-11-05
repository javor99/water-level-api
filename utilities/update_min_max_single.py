#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update Min/Max for Single Station
Simple script that fetches 5 years of data in one call and updates min/max
"""

import sys
import sqlite3
import requests
from datetime import datetime, timedelta

def get_db_connection():
    """Create a database connection."""
    return sqlite3.connect('water_levels.db')

def fetch_5_years_data(station_id: str):
    """Fetch 5 years of data in one API call."""
    print(f"🔍 Fetching 5 years of data for station {station_id}...")
    
    # Calculate date range - 5 years back
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        url = "https://vandah.miljoeportal.dk/api/water-levels"
        params = {
            "stationId": station_id,
            "from": start_date.strftime("%Y-%m-%dT00:00Z"),
            "to": end_date.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }
        
        print("🌐 Making API call...")
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        raw = r.json()

        if not raw or not raw[0].get("results"):
            print("❌ No data returned from API")
            return []

        recs = raw[0]["results"]
        print(f"📊 Got {len(recs)} raw records from API")
        
        # Process the data
        all_data = []
        for rec in recs:
            try:
                dt = datetime.strptime(rec["measurementDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                level_cm = float(rec["result"])
                all_data.append({
                    'date': dt.date(),
                    'datetime': dt,
                    'water_level_cm': level_cm,
                    'water_level_m': level_cm / 100.0
                })
            except (ValueError, KeyError) as e:
                continue  # Skip invalid records
        
        print(f"✅ Processed {len(all_data)} valid records")
        return all_data
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return []

def calculate_minmax(station_id: str, historical_data: list):
    """Calculate min/max values."""
    print(f"🧮 Calculating min/max values...")
    
    if not historical_data:
        return None
    
    levels_cm = [d['water_level_cm'] for d in historical_data]
    levels_m = [d['water_level_m'] for d in historical_data]
    
    min_cm = min(levels_cm)
    max_cm = max(levels_cm)
    min_m = min(levels_m)
    max_m = max(levels_m)
    
    # Find the dates when min/max occurred
    min_record = min(historical_data, key=lambda x: x['water_level_cm'])
    max_record = max(historical_data, key=lambda x: x['water_level_cm'])
    
    print(f"📈 Min: {min_cm:.2f} cm ({min_m:.2f} m) on {min_record['date']}")
    print(f"�� Max: {max_cm:.2f} cm ({max_m:.2f} m) on {max_record['date']}")
    
    return {
        'station_id': station_id,
        'min_level_cm': min_cm,
        'max_level_cm': max_cm,
        'min_level_m': min_m,
        'max_level_m': max_m,
        'min_date': min_record['date'],
        'max_date': max_record['date'],
        'total_records': len(historical_data),
        'date_range': {
            'start': min(d['date'] for d in historical_data),
            'end': max(d['date'] for d in historical_data)
        }
    }

def update_database(station_minmax: dict):
    """Update database with min/max values."""
    print(f"💾 Updating database...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE stations 
            SET min_level_cm = ?, max_level_cm = ?,
                min_level_m = ?, max_level_m = ?
            WHERE station_id = ?
        """, (
            station_minmax['min_level_cm'],
            station_minmax['max_level_cm'],
            station_minmax['min_level_m'],
            station_minmax['max_level_m'],
            station_minmax['station_id']
        ))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"✅ Successfully updated {cursor.rowcount} row(s)")
            return True
        else:
            print(f"⚠️  No rows updated - station may not exist")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python3 update_min_max_single.py <station_id>")
        print("Example: python3 update_min_max_single.py 70000864")
        sys.exit(1)
    
    station_id = sys.argv[1]
    
    print("📊 UPDATING MIN/MAX FOR SINGLE STATION")
    print("=" * 50)
    print(f"Station ID: {station_id}")
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Fetch 5 years of data
        historical_data = fetch_5_years_data(station_id)
        
        if not historical_data:
            print("❌ No historical data found")
            sys.exit(1)
        
        print()
        
        # Calculate min/max
        minmax_data = calculate_minmax(station_id, historical_data)
        
        if not minmax_data:
            print("❌ Failed to calculate min/max")
            sys.exit(1)
        
        print()
        
        # Update database
        success = update_database(minmax_data)
        
        if success:
            print()
            print("🎉 SUCCESS!")
            print(f"✅ Station {station_id} min/max updated")
            print(f"📊 Min: {minmax_data['min_level_cm']:.2f} cm, Max: {minmax_data['max_level_cm']:.2f} cm")
            print(f"📅 Based on {minmax_data['total_records']} records")
        else:
            print("❌ Database update failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
