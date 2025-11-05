#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate historical min/max water levels for each station from the last 5 years
"""

import os
import sqlite3
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
import time

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    return conn

def fetch_historical_water_data(station_id: str, years_back: int = 5) -> list:
    """Fetch historical water level data for a station going back specified years."""
    print(f"Fetching {years_back} years of data for station {station_id}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    all_data = []
    
    # Fetch data in chunks to avoid API limits
    current_start = start_date
    chunk_days = 90  # 3 months at a time
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        try:
            url = "https://vandah.miljoeportal.dk/api/water-levels"
            params = {
                "stationId": station_id,
                "from": current_start.strftime("%Y-%m-%dT00:00Z"),
                "to": current_end.strftime("%Y-%m-%dT00:00Z"),
                "format": "json",
            }
            
            print(f"  Fetching {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            raw = r.json()

            if raw and raw[0].get("results"):
                recs = raw[0]["results"]
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
            
            # Be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error fetching chunk: {e}")
        
        current_start = current_end
    
    print(f"  Found {len(all_data)} records")
    return all_data

def calculate_minmax_for_station(station_id: str, historical_data: list) -> dict:
    """Calculate min/max values from historical data."""
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

def update_database_with_historical_minmax(station_minmax: dict):
    """Update the database with calculated historical min/max values."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()

def main():
    """Main function to calculate and update historical min/max values."""
    print("=== CALCULATING HISTORICAL MIN/MAX WATER LEVELS ===")
    print(f"Fetching data for the last 5 years...")
    print(f"Started at: {datetime.now()}")
    
    # Get all stations
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT station_id, name FROM stations ORDER BY station_id")
    stations = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(stations)} stations to process")
    
    all_results = []
    
    for i, (station_id, station_name) in enumerate(stations, 1):
        print(f"\n[{i}/{len(stations)}] Processing station {station_id} ({station_name})")
        
        try:
            # Fetch historical data
            historical_data = fetch_historical_water_data(station_id, years_back=5)
            
            if historical_data:
                # Calculate min/max
                minmax_data = calculate_minmax_for_station(station_id, historical_data)
                
                if minmax_data:
                    # Update database
                    update_database_with_historical_minmax(minmax_data)
                    all_results.append(minmax_data)
                    
                    print(f"  ✅ Updated: Min={minmax_data['min_level_cm']:.2f}cm (on {minmax_data['min_date']}), Max={minmax_data['max_level_cm']:.2f}cm (on {minmax_data['max_date']})")
                    print(f"  📊 Total records: {minmax_data['total_records']}, Date range: {minmax_data['date_range']['start']} to {minmax_data['date_range']['end']}")
                else:
                    print(f"  ❌ Failed to calculate min/max")
            else:
                print(f"  ❌ No historical data found")
                
        except Exception as e:
            print(f"  ❌ Error processing station {station_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== SUMMARY ===")
    print(f"Completed at: {datetime.now()}")
    print(f"Successfully processed: {len(all_results)}/{len(stations)} stations")
    
    if all_results:
        print(f"\n=== HISTORICAL MIN/MAX VALUES ===")
        for result in all_results:
            print(f"Station {result['station_id']}:")
            print(f"  Min: {result['min_level_cm']:.2f} cm ({result['min_level_m']:.2f} m) on {result['min_date']}")
            print(f"  Max: {result['max_level_cm']:.2f} cm ({result['max_level_m']:.2f} m) on {result['max_date']}")
            print(f"  Records: {result['total_records']} from {result['date_range']['start']} to {result['date_range']['end']}")
            print()

if __name__ == "__main__":
    main()
