#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update Last 30 Days of Water Levels
Simple script that only updates the last 30 days of water level data for all stations
"""

import os
import sqlite3
import pandas as pd
import json
import requests
from datetime import datetime, timedelta

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    return conn

def fetch_water_daily(vandah_station_id: str, past_days: int) -> pd.DataFrame:
    """Vandah 15-min → daily mean (cm); columns: date, water_level_cm."""
    to_time = datetime.now().replace(microsecond=0)
    from_time = to_time - timedelta(days=past_days)

    url = "https://vandah.miljoeportal.dk/api/water-levels"
    params = {
        "stationId": vandah_station_id,
        "from": from_time.strftime("%Y-%m-%dT%H:%MZ"),
        "to":   to_time.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()

    if not raw or not raw[0].get("results"):
        print(f"Warning: No water level data for station {vandah_station_id}")
        return pd.DataFrame()

    recs = raw[0]["results"]
    df = pd.DataFrame({
        "dt": pd.to_datetime([rr["measurementDateTime"] for rr in recs], utc=True),
        "water_level_cm": [rr["result"] for rr in recs],
    })
    df["date"] = df["dt"].dt.date
    daily = df.groupby("date", as_index=False)["water_level_cm"].mean()
    return daily

def update_30_days_water_levels():
    """Update last 30 days of water levels for all stations."""
    print("📊 UPDATING LAST 30 DAYS OF WATER LEVELS")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get station coordinates
    with open('station_coordinates.json', 'r') as f:
        stations = json.load(f)
    
    print(f"Processing {len(stations)} stations...")
    print()
    
    successful_stations = 0
    failed_stations = 0
    
    for station in stations:
        station_id = station['id']
        station_name = station['name']
        
        print(f"[{station_id}] {station_name}")
        
        try:
            # Fetch water data for this specific station (last 30 days)
            water_data = fetch_water_daily(station_id, 30)
            
            if not water_data.empty:
                # Delete existing historical data for this station
                cursor.execute("DELETE FROM last_30_days_historical WHERE station_id = ?", (station_id,))
                
                # Insert historical data (last 30 days)
                records_inserted = 0
                for _, row in water_data.iterrows():
                    cursor.execute("""
                        INSERT INTO last_30_days_historical 
                        (station_id, measurement_date, water_level_cm, water_level_m, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        station_id,
                        row['date'],
                        row['water_level_cm'],
                        row['water_level_cm'] / 100.0,  # Convert to meters
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    records_inserted += 1
                
                print(f"  ✅ Updated {records_inserted} records")
                successful_stations += 1
            else:
                print(f"  ⚠️  No water data available")
                failed_stations += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_stations += 1
        
        print()
    
    conn.commit()
    conn.close()
    
    print("=" * 50)
    print("�� SUMMARY")
    print(f"Completed at: {datetime.now()}")
    print(f"Successful stations: {successful_stations}")
    print(f"Failed stations: {failed_stations}")
    print(f"Total stations: {len(stations)}")
    
    # Show database summary
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM last_30_days_historical")
    hist_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(measurement_date), MAX(measurement_date) FROM last_30_days_historical")
    date_range = cursor.fetchone()
    
    print()
    print("Database Status:")
    print(f"  Historical records: {hist_count}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")
    
    conn.close()

def main():
    """Main function."""
    update_30_days_water_levels()

if __name__ == "__main__":
    main()
