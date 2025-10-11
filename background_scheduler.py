#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Background Scheduler for Water Level System
Automatically updates 30-day history and predictions for all stations every 24 hours.
Enhanced with detailed logging.
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
from email_service import send_water_level_alert

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
            "level_cm": [rr["result"] for rr in recs],
        })
        df["date"] = df["dt"].dt.date
        daily = df.groupby("date", as_index=False)["level_cm"].mean()
        return daily
        
    except Exception as e:
        print(f"    ❌ Error fetching water data for {station_id}: {e}")
        return pd.DataFrame()

def update_30_day_history_for_station(station_id: str, station_name: str):
    """Update 30-day historical data for a single station."""
    try:
        # Fetch water data for last 30 days
        water_data = fetch_water_daily(station_id, 40)
        
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
                (station_id, timestamp, level_cm, level_cm, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                station_id,
                row['date'],
                row['level_cm'],
                row['level_cm'] / 100.0,  # Convert to meters
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            records_inserted += 1
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error updating 30-day history for {station_id}: {e}")
        return False

def update_current_water_level_for_station(station_id: str, station_name: str):
    """Update current water level with the most recent measurement from 30-day history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent measurement from 30-day history
        cursor.execute("""
            SELECT level_cm, level_cm, timestamp
            FROM last_30_days_historical 
            WHERE station_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (station_id,))
        
        latest_measurement = cursor.fetchone()
        
        if not latest_measurement:
            conn.close()
            return False
        
        # Insert or update current water level
        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, level_cm, timestamp, level_cm, level_cm, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['level_cm'],
            latest_measurement['timestamp'],
            latest_measurement['level_cm'],
            latest_measurement['level_cm'],
            latest_measurement['timestamp'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to update current water level for {station_id}: {e}")
        return False

def update_predictions_for_station(station_id: str, station_name: str, latitude: float, longitude: float):
    """Update predictions for a single station."""
    print(f"    🔍🔍🔍 STARTING PREDICTIONS FOR {station_id} 🔍🔍🔍")
    try:
        print(f"    🔍 Station ID: {station_id}")
        print(f"    🔍 Station Name: {station_name}")
        print(f"    🔍 Latitude: {latitude}")
        print(f"    🔍 Longitude: {longitude}")
        
        # Run the prediction script
        cmd = [
            'python3', 'predict_unseen_station.py',
            '--vandah_id', station_id,
            '--lat', str(latitude),
            '--lon', str(longitude),
            '--unseen_strategy', 'nearest',
            '--anchor', 'none',
            '--past_days', '40'
        ]
        
        print(f"    🔍 Command: {' '.join(cmd)}")
        print(f"    🔍 Running prediction script...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print(f"    🔍 Return code: {result.returncode}")
        print(f"    🔍 Stdout length: {len(result.stdout)}")
        print(f"    🔍 Stderr length: {len(result.stderr)}")
        
        if result.stdout:
            print(f"    🔍 Stdout: {result.stdout[:500]}...")
        if result.stderr:
            print(f"    🔍 Stderr: {result.stderr[:500]}...")
        
        if result.returncode == 0:
            print(f"    🔍 Prediction script succeeded!")
            # Save predictions to database
            csv_path = f'predictions/predictions_{station_id}_unseen.csv'
            print(f"    🔍 Looking for CSV: {csv_path}")
            
            if os.path.exists(csv_path):
                print(f"    🔍 CSV exists! Reading data...")
                df = pd.read_csv(csv_path)
                print(f"    🔍 CSV shape: {df.shape}")
                print(f"    🔍 CSV columns: {list(df.columns)}")
                print(f"    🔍 First row: {dict(df.iloc[0]) if len(df) > 0 else 'No data'}")
                
                print(f"    🔍 Connecting to database...")
                conn = get_db_connection()
                cursor = conn.cursor()
                
                print(f"    🔍 Deleting existing predictions...")
                cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
                deleted_count = cursor.rowcount
                print(f"    🔍 Deleted {deleted_count} existing predictions")
                
                # Insert new predictions
                records_inserted = 0
                print(f"    🔍 Inserting {len(df)} new predictions...")
                
                for i, (_, row) in enumerate(df.iterrows()):
                    try:
                        print(f"    🔍 Inserting row {i+1}/{len(df)}: {dict(row)}")
                        cursor.execute("""
                            INSERT INTO predictions
                            (station_id, prediction_date, predicted_water_level_cm,
                             change_from_last_cm, forecast_date, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            station_id,
                            row['date'],
                            row['predicted_water_level_cm'],
                            row['change_from_last_daily_mean_cm'],
                            row['date'],
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ))
                        records_inserted += 1
                        print(f"    🔍 Row {i+1} inserted successfully")
                    except Exception as insert_error:
                        print(f"    ❌❌❌ ERROR INSERTING ROW {i+1}: {insert_error} ❌❌❌")
                        print(f"    🔍 Row data: {dict(row)}")
                        print(f"    🔍 Error type: {type(insert_error)}")
                        import traceback
                        traceback.print_exc()
                        raise
                
                # Save predictions to past_predictions table (historical record)
                print(f"    🔍 Saving predictions to past_predictions table...")
                forecast_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO past_predictions
                        (station_id, prediction_date, predicted_water_level_cm,
                         change_from_last_cm, forecast_created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        station_id,
                        row['date'],
                        row['predicted_water_level_cm'],
                        row['change_from_last_daily_mean_cm'],
                        forecast_timestamp
                    ))
                
                print(f"    🔍 Saved {len(df)} predictions to past_predictions table")
                
                print(f"    🔍 Committing transaction...")
                conn.commit()
                print(f"    🔍 Closing connection...")
                conn.close()
                print(f"    ✅✅✅ SUCCESS: Inserted {records_inserted} prediction records ✅✅✅")
                
                return True
            else:
                print(f"    ❌❌❌ CSV FILE NOT FOUND: {csv_path} ❌❌❌")
                return False
        else:
            print(f"    ❌❌❌ PREDICTION SCRIPT FAILED WITH CODE {result.returncode} ❌❌❌")
            return False
            
    except Exception as e:
        print(f"    ❌❌❌ EXCEPTION IN PREDICTIONS: {e} ❌❌❌")
        print(f"    🔍 Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_and_send_alerts_for_station(station_id: str, station_name: str):
    """Check if predictions exceed threshold and send alerts to subscribers."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the latest prediction for this station
        cursor.execute("""
            SELECT predicted_water_level_cm, prediction_date
            FROM predictions 
            WHERE station_id = ? 
            ORDER BY prediction_date DESC 
            LIMIT 1
        """, (station_id,))
        
        latest_prediction = cursor.fetchone()
        
        if not latest_prediction:
            conn.close()
            return False
        
        current_prediction = latest_prediction['predicted_water_level_cm']
        
        # Get the maximum historical level for this station
        cursor.execute("""
            SELECT MAX(level_cm) as max_level
            FROM last_30_days_historical 
            WHERE station_id = ?
        """, (station_id,))
        
        max_level_result = cursor.fetchone()
        
        if not max_level_result or not max_level_result['max_level']:
            conn.close()
            return False
        
        max_level = max_level_result['max_level']
        
        # Get all active subscriptions for this station
        cursor.execute("""
            SELECT user_email, threshold_percentage
            FROM station_subscriptions 
            WHERE station_id = ? AND is_active = 1
        """, (station_id,))
        
        subscriptions = cursor.fetchall()
        conn.close()
        
        if not subscriptions:
            return False
        
        # Check each subscription and send alerts if needed
        alerts_sent = 0
        for subscription in subscriptions:
            user_email = subscription['user_email']
            threshold_percentage = subscription['threshold_percentage']
            threshold_level = max_level * threshold_percentage
            
            # Check if prediction exceeds threshold
            if current_prediction >= threshold_level:
                print(f"     ALERT: {station_name} prediction ({current_prediction:.2f}m) exceeds threshold ({threshold_percentage*100:.0f}% = {threshold_level:.2f}m)")
                
                # Send alert email
                if send_water_level_alert(
                    user_email=user_email,
                    station_name=station_name,
                    station_id=station_id,
                    current_prediction=current_prediction,
                    max_level=max_level,
                    threshold_percentage=threshold_percentage
                ):
                    alerts_sent += 1
                    print(f"    📧 Alert email sent to {user_email}")
                else:
                    print(f"    ❌ Failed to send alert email to {user_email}")
        
        if alerts_sent > 0:
            print(f"    ✅ {alerts_sent} alert(s) sent for {station_name}")
        
        return alerts_sent > 0
        
    except Exception as e:
        print(f"    ❌ Error checking alerts for {station_id}: {e}")
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
        
        print(f"  [{i}/{total_stations}] Processing {station_name} ({station_id})")
        
        # Update 30-day history
        if update_30_day_history_for_station(station_id, station_name):
            results['30_day_history'] += 1
            print(f"    ✅ 30-day history updated")
        else:
            print(f"    ❌ 30-day history update failed")
        
        # Update current water level
        if update_current_water_level_for_station(station_id, station_name):
            results['current_water_level'] += 1
            print(f"    ✅ Current water level updated")
        else:
            print(f"    ❌ Current water level update failed")
        
        # Update predictions
        if update_predictions_for_station(station_id, station_name, latitude, longitude):
            results['predictions'] += 1
            print(f"    ✅ Predictions updated")
            
            # Check for alerts after successful prediction update
            check_and_send_alerts_for_station(station_id, station_name)
        else:
            print(f"    ❌ Predictions update failed")
    
    print(f"✅ Update completed at {datetime.now()}")
    print(f"  📈 Results: {results['30_day_history']}/{total_stations} history, "
          f"{results['current_water_level']}/{total_stations} current level, "
          f"{results['predictions']}/{total_stations} predictions")

def background_scheduler():
    """Background scheduler that runs every 24 hours."""
    # Create log file for background scheduler
    log_file = open("background_scheduler.log", "a")
    
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        print(log_line.strip())
        log_file.write(log_line)
        log_file.flush()
    
    log_message("🚀 Background scheduler started - updating every 24 hours")
    log_message("📅 Next update scheduled in 24 hours...")
    
    while True:
        try:
            print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scheduled update cycle...")
            update_all_stations()
            print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update cycle completed successfully")
        except Exception as e:
            print(f"❌ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in background scheduler: {e}")
        
        # Wait 24 hours (86400 seconds)
        time.sleep(86400)

def start_background_scheduler():
    """Start the background scheduler in a separate thread."""
    print("🔧 Starting background scheduler thread...")
    thread = threading.Thread(target=background_scheduler)
    thread.daemon = True
    thread.start()
    print("✅ Background scheduler thread started successfully")
    print("📝 Background scheduler will log to console every 24 hours")
    print("🔄 First update cycle will start immediately...")

if __name__ == "__main__":
    print("🚀 Starting background scheduler...")
    start_background_scheduler()
    print("✅ Background scheduler started")
    
    # Keep the script running
    try:
        while True:
            time.sleep(60)  # Sleep for 1 minute
    except KeyboardInterrupt:
        print("\n🛑 Background scheduler stopped")
