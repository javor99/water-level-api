#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""
Background Scheduler for Water Level System
Automatically updates 30-day history and predictions for all stations every 2 hours.
Enhanced with detailed logging and exponential backoff retry logic.
"""

import os
import sys
import sqlite3
import threading
import time
import subprocess
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError as RequestsConnectionError
import pandas as pd
from services.email_service import send_water_level_alert

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

def fetch_water_daily(station_id: str, past_days: int, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch daily water level data for a station from Vandah API with exponential backoff retry.
    
    Args:
        station_id: Station ID to fetch data for
        past_days: Number of days of historical data to fetch
        max_retries: Maximum number of retry attempts (default: 5)
    
    Returns:
        DataFrame with daily water level data, or empty DataFrame on failure
    """
    to_time = datetime.now().replace(microsecond=0)
    from_time = to_time - timedelta(days=past_days)

    url = "https://vandah.miljoeportal.dk/api/water-levels"
    params = {
        "stationId": station_id,
        "from": from_time.strftime("%Y-%m-%dT%H:%MZ"),
        "to": to_time.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "json",
    }
    
    # Retryable HTTP status codes
    retryable_status_codes = {403, 500, 502, 503, 504}
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            
            # If successful, process and return data
            if r.status_code == 200:
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
                
                if attempt > 0:
                    print(f"    ✅ Successfully fetched data for {station_id} after {attempt} retry(ies)")
                return daily
            
            # Check if status code is retryable
            elif r.status_code in retryable_status_codes:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff: 2^attempt seconds, max 30 seconds
                    wait_time = min(2 ** attempt, 30)
                    print(f"    ⚠️  API returned {r.status_code} for {station_id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    r.raise_for_status()  # Raise exception on final attempt
            else:
                # Non-retryable status code, raise immediately
                r.raise_for_status()
                
        except (Timeout, RequestsConnectionError) as e:
            # Network errors - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                print(f"    ⚠️  Network error for {station_id} ({type(e).__name__}), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"    ❌ Error fetching water data for {station_id} after {max_retries} attempts: {e}")
                return pd.DataFrame()
                
        except RequestException as e:
            # Other request errors - retry if retryable status code
            if hasattr(e.response, 'status_code') and e.response.status_code in retryable_status_codes:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)
                    print(f"    ⚠️  Request error for {station_id} (HTTP {e.response.status_code}), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"    ❌ Error fetching water data for {station_id} after {max_retries} attempts: {e}")
                    return pd.DataFrame()
            else:
                # Non-retryable error, fail immediately
                print(f"    ❌ Error fetching water data for {station_id}: {e}")
                return pd.DataFrame()
                
        except Exception as e:
            # Unexpected errors - don't retry
            print(f"    ❌ Unexpected error fetching water data for {station_id}: {e}")
            return pd.DataFrame()
    
    # If we exhausted all retries
    print(f"    ❌ Failed to fetch water data for {station_id} after {max_retries} attempts")
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
            'python3', 'utilities/predict_unseen_station.py',
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
        print(f"    🔔 Checking alerts for {station_name} ({station_id})...")
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
            print(f"    ⚠️  No predictions found for {station_name} - skipping alert check")
            conn.close()
            return False
        
        current_prediction = latest_prediction['predicted_water_level_cm']
        
        # Get the min/max values from min_max_values table
        cursor.execute("""
            SELECT min_level_cm, max_level_cm
            FROM min_max_values 
            WHERE station_id = ?
        """, (station_id,))
        
        minmax_result = cursor.fetchone()
        
        if not minmax_result or not minmax_result['min_level_cm'] or not minmax_result['max_level_cm']:
            conn.close()
            return False
        
        min_level = minmax_result['min_level_cm']
        max_level = minmax_result['max_level_cm']
        
        # Get all active subscriptions for this station
        cursor.execute("""
            SELECT user_email, threshold_percentage, alert_type, last_alert_sent_at
            FROM station_subscriptions 
            WHERE station_id = ? AND is_active = 1
        """, (station_id,))
        
        subscriptions = cursor.fetchall()
        
        if not subscriptions:
            print(f"    ℹ️  No active subscriptions for {station_name} - no alerts to check")
            conn.close()
            return False
        
        print(f"    📋 Found {len(subscriptions)} active subscription(s) for {station_name}")
        
        # Check each subscription and send alerts if needed
        alerts_sent = 0
        now = datetime.now()
        
        for subscription in subscriptions:
            user_email = subscription['user_email']
            threshold_percentage = subscription['threshold_percentage']
            # sqlite3.Row doesn't support .get(), use dict-style access with fallback
            alert_type = subscription['alert_type'] if subscription['alert_type'] else 'above'
            last_alert_sent_at = subscription['last_alert_sent_at'] if subscription['last_alert_sent_at'] else None
            
            # Calculate threshold as percentage between min and max
            threshold_level = min_level + (max_level - min_level) * threshold_percentage
            
            # Check if alert condition is met based on alert_type
            alert_triggered = False
            if alert_type == 'above':
                # Flood alert: prediction exceeds threshold
                alert_triggered = current_prediction >= threshold_level
                alert_msg = f"exceeds (above)"
            elif alert_type == 'below':
                # Drain/low water alert: prediction falls below threshold
                alert_triggered = current_prediction <= threshold_level
                alert_msg = f"falls below"
            
            if alert_triggered:
                # Check if 24 hours have passed since last alert (or if no alert was ever sent)
                should_send_alert = True
                if last_alert_sent_at:
                    try:
                        last_alert_time = datetime.strptime(last_alert_sent_at, '%Y-%m-%d %H:%M:%S')
                        hours_since_last_alert = (now - last_alert_time).total_seconds() / 3600
                        if hours_since_last_alert < 24:
                            should_send_alert = False
                            print(f"    ⏸️  Skipping alert to {user_email}: Last alert sent {hours_since_last_alert:.1f} hours ago (minimum 24h required)")
                    except (ValueError, TypeError):
                        # If date parsing fails, send alert anyway
                        pass
                
                if should_send_alert:
                    print(f"    🚨 ALERT ({alert_type.upper()}): {station_name} prediction ({current_prediction:.2f}cm) {alert_msg} threshold ({threshold_percentage*100:.0f}% = {threshold_level:.2f}cm)")
                    
                    # Send alert email
                    if send_water_level_alert(
                        user_email=user_email,
                        station_name=station_name,
                        station_id=station_id,
                        current_prediction=current_prediction,
                        min_level=min_level,
                        max_level=max_level,
                        threshold_percentage=threshold_percentage,
                        alert_type=alert_type
                    ):
                        # Update last_alert_sent_at timestamp
                        cursor.execute("""
                            UPDATE station_subscriptions 
                            SET last_alert_sent_at = ?
                            WHERE user_email = ? AND station_id = ? AND alert_type = ?
                        """, (now.strftime('%Y-%m-%d %H:%M:%S'), user_email, station_id, alert_type))
                        conn.commit()
                        
                        alerts_sent += 1
                        print(f"    📧 Alert email sent to {user_email} (next alert in 24 hours)")
                    else:
                        print(f"    ❌ Failed to send alert email to {user_email}")
        
        conn.close()
        
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
    """Background scheduler that runs every 2 hours."""
    # Create log file for background scheduler
    log_file = open("background_scheduler.log", "a")
    
    def log_message(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        print(log_line.strip())
        log_file.write(log_line)
        log_file.flush()
    
    log_message("🚀 Background scheduler started - updating every 2 hours")
    log_message("📅 Next update scheduled in 2 hours...")
    
    while True:
        try:
            print(f"\n⏰ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scheduled update cycle...")
            update_all_stations()
            print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Update cycle completed successfully")
        except Exception as e:
            print(f"❌ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in background scheduler: {e}")
        
        # Wait 2 hours (7200 seconds)
        time.sleep(7200)

def start_background_scheduler():
    """Start the background scheduler in a separate thread."""
    print("🔧 Starting background scheduler thread...")
    thread = threading.Thread(target=background_scheduler)
    thread.daemon = True
    thread.start()
    print("✅ Background scheduler thread started successfully")
    print("📝 Background scheduler will log to console every 2 hours")
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
