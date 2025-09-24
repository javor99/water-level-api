#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Background Scheduler for Water Level System
Automatically updates 30-day history and predictions for all stations every 5 minutes.
"""

import os
import sys
import sqlite3
import threading
import time
import subprocess
from datetime import datetime, timedelta
import requests
import pandas as pd

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_all_stations():
    """Get all stations from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT station_id, name, latitude, longitude
        FROM stations 
        ORDER BY name
    """)
    
    stations = cursor.fetchall()
    conn.close()
    return stations

def fetch_water_daily(station_id: str, past_days: int) -> pd.DataFrame:
    """Fetch daily water level data for a station from Vandah API."""
    to_time = datetime.now().replace(microsecond=0)
    from_time = to_time - timedelta(days=past_days)

    url = "https://vandah.miljoeportal.dk/api/water-levels"
    params = {
        "stationId": station_id,
        "from": from_time.strftime("%Y-%m-%dT%H:%MZ"),
        "to": to_time.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        raw = r.json()

        if not raw or not raw[0].get("results"):
            return pd.DataFrame()

        recs = raw[0]["results"]
        df = pd.DataFrame({
            "dt": pd.to_datetime([rr["measurementDateTime"] for rr in recs], utc=True),
            "water_level_cm": [rr["result"] for rr in recs],
        })
        df["date"] = df["dt"].dt.date
        daily = df.groupby("date", as_index=False)["water_level_cm"].mean()
        return daily
        
    except Exception as e:
        print(f"  ❌ Error fetching water data for {station_id}: {e}")
        return pd.DataFrame()

def update_30_day_history_for_station(station_id: str, station_name: str):
    """Update 30-day historical data for a single station."""
    try:
        # Fetch water data for last 30 days
        water_data = fetch_water_daily(station_id, 30)
        
        if water_data.empty:
            return False
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete existing historical data for this station
        cursor.execute("DELETE FROM last_30_days_historical WHERE station_id = ?", (station_id,))
        
        # Insert new historical data
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
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error updating 30-day history for {station_id}: {e}")
        return False

def update_current_water_level_for_station(station_id: str, station_name: str):
    """Update current water level with the most recent measurement from 30-day history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent measurement from 30-day history
        cursor.execute("""
            SELECT water_level_cm, water_level_m, measurement_date
            FROM last_30_days_historical 
            WHERE station_id = ? 
            ORDER BY measurement_date DESC 
            LIMIT 1
        """, (station_id,))
        
        latest_measurement = cursor.fetchone()
        
        if not latest_measurement:
            conn.close()
            return False
        
        # Insert or update current water level
        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, water_level_cm, water_level_m, measurement_date, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['water_level_cm'],
            latest_measurement['water_level_m'],
            latest_measurement['measurement_date'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to update current water level for {station_id}: {e}")
        return False

def update_predictions_for_station(station_id: str, station_name: str, latitude: float, longitude: float):
    """Update predictions for a single station."""
    try:
        # Run the prediction script
        cmd = [
            'python3', 'predict_unseen_station.py',
            '--vandah_id', station_id,
            '--lat', str(latitude),
            '--lon', str(longitude),
            '--unseen_strategy', 'nearest',
            '--anchor', 'none',
            '--past_days', '60'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Save predictions to database
            csv_path = f'predictions/predictions_{station_id}_unseen.csv'
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Delete existing predictions for this station
                cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
                
                # Insert new predictions
                records_inserted = 0
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO predictions
                        (station_id, prediction_date, predicted_water_level_cm, predicted_water_level_m,
                         change_from_last_cm, forecast_date, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        station_id,
                        row['date'],
                        row['predicted_water_level_cm'],
                        row['predicted_water_level_m'],
                        row['change_from_last_daily_mean_cm'],
                        row['date'],
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    records_inserted += 1
                
                conn.commit()
                conn.close()
                
                return True
            else:
                return False
        else:
            return False
            
    except Exception as e:
        print(f"  ❌ Error updating predictions for {station_id}: {e}")
        return False

def update_all_stations():
    """Update 30-day history, current water level, and predictions for all stations."""
    print(f"🔄 Starting automatic update at {datetime.now()}")
    
    stations = get_all_stations()
    total_stations = len(stations)
    
    if total_stations == 0:
        print("  ⚠️  No stations found in database")
        return
    
    print(f"  📊 Updating {total_stations} stations...")
    
    results = {
        '30_day_history': 0,
        'current_water_level': 0,
        'predictions': 0,
        'total': total_stations
    }
    
    for i, station in enumerate(stations, 1):
        station_id = station['station_id']
        station_name = station['name']
        latitude = station['latitude']
        longitude = station['longitude']
        
        print(f"  [{i}/{total_stations}] {station_name} ({station_id})")
        
        # Update 30-day history
        if update_30_day_history_for_station(station_id, station_name):
            results['30_day_history'] += 1
        
        # Update current water level
        if update_current_water_level_for_station(station_id, station_name):
            results['current_water_level'] += 1
        
        # Update predictions
        if update_predictions_for_station(station_id, station_name, latitude, longitude):
            results['predictions'] += 1
    
    print(f"✅ Update completed at {datetime.now()}")
    print(f"  📈 Results: {results['30_day_history']}/{total_stations} history, "
          f"{results['current_water_level']}/{total_stations} current level, "
          f"{results['predictions']}/{total_stations} predictions")

def background_scheduler():
    """Background scheduler that runs every 5 minutes."""
    print("🚀 Background scheduler started - updating every 5 minutes")
    print("📅 Next update scheduled in 5 minutes...")
    
    while True:
        try:
            print(f"\n⏰ [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Starting scheduled update cycle...")
            update_all_stations()
            print(f"✅ [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Update cycle completed successfully")
            print("⏳ Waiting 5 minutes until next update...")
        except Exception as e:
            print(f"❌ [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error in background scheduler: {e}")
        
        # Wait 5 minutes (300 seconds)
        time.sleep(300)
    """Background scheduler that runs every 5 minutes."""
    print("🚀 Background scheduler started - updating every 5 minutes")
    
    while True:
        try:
            update_all_stations()
        except Exception as e:
            print(f"❌ Error in background scheduler: {e}")
        
        # Wait 5 minutes (300 seconds)
        time.sleep(300)

def start_background_scheduler():
    """Start the background scheduler in a separate thread."""
    thread = threading.Thread(target=background_scheduler)
    thread.daemon = True
    thread.start()
    print("✅ Background scheduler thread started")

if __name__ == "__main__":
    # If run directly, start the scheduler
    background_scheduler()
