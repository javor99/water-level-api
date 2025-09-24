#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Predictions for All Stations and Update Database
This script combines prediction generation and database updates in one process.
Automatically tries different time windows (60 days, then 90 days) if insufficient data.
"""

import os
import sys
import json
import sqlite3
import subprocess
import pandas as pd
from datetime import datetime
import time

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def load_station_coordinates():
    """Load station coordinates from JSON file."""
    with open('station_coordinates.json', 'r') as f:
        stations = json.load(f)
    return stations

def run_prediction_for_station(station_id, lat, lon, station_name):
    """Run prediction for a single station using the predict_unseen_station.py script.
    Tries 60 days first, then 90 days if insufficient data."""
    print(f"🔮 Running predictions for station {station_id} ({station_name})")
    
    # Try different time windows
    time_windows = [60, 90]
    
    for past_days in time_windows:
        print(f"  📅 Trying {past_days} days of historical data...")
        
        try:
            # Run the prediction script with current time window
            cmd = [
                'python3', 'predict_unseen_station.py',
                '--vandah_id', station_id,
                '--lat', str(lat),
                '--lon', str(lon),
                '--unseen_strategy', 'nearest',
                '--anchor', 'adjust',
                '--past_days', str(past_days)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ Predictions generated successfully with {past_days} days of data")
                return True
            else:
                # Check if it's a data insufficiency error
                error_msg = result.stderr.lower()
                if "not enough rows" in error_msg or "sequence" in error_msg:
                    print(f"  ⚠️  Insufficient data with {past_days} days")
                    if past_days == 90:  # This was our last attempt
                        print(f"  ❌ Cannot generate predictions - insufficient data even with 90 days")
                        return False
                    else:
                        print(f"  🔄 Trying with more historical data...")
                        continue
                else:
                    print(f"  ❌ Prediction failed: {result.stderr}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Prediction timed out for station {station_id}")
            return False
        except Exception as e:
            print(f"  ❌ Error running prediction: {e}")
            return False
    
    return False

def save_predictions_to_db(station_id):
    """Save predictions from CSV file to database."""
    csv_path = f'predictions/predictions_{station_id}_unseen.csv'
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  Prediction CSV not found: {csv_path}")
        return False
    
    try:
        # Read the predictions CSV
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
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            records_inserted += 1
        
        conn.commit()
        conn.close()
        
        print(f"  💾 Saved {records_inserted} predictions to database")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving predictions to database: {e}")
        return False

def update_water_levels_and_history():
    """Update current water levels and 30-day historical data."""
    print("📊 Updating water levels and historical data...")
    
    try:
        # Run the database update script
        result = subprocess.run(['python3', 'update_30_days.py'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("  ✅ Water levels and historical data updated successfully")
            return True
        else:
            print(f"  ❌ Database update failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⏰ Database update timed out")
        return False
    except Exception as e:
        print(f"  ❌ Error updating database: {e}")
        return False

def get_database_summary():
    """Get summary of database contents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Count records
    cursor.execute('SELECT COUNT(*) FROM predictions')
    pred_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM water_levels')
    water_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM last_30_days_historical')
    hist_count = cursor.fetchone()[0]
    
    # Get latest prediction date
    cursor.execute('SELECT MAX(prediction_date) FROM predictions')
    latest_pred = cursor.fetchone()[0]
    
    # Get station summary
    cursor.execute('''
        SELECT s.station_id, s.name, w.water_level_cm, w.measurement_date,
               s.min_level_cm, s.max_level_cm, s.last_30_days_min_cm, s.last_30_days_max_cm
        FROM stations s
        LEFT JOIN water_levels w ON s.station_id = w.station_id
        ORDER BY s.station_id
    ''')
    
    stations = cursor.fetchall()
    
    conn.close()
    
    return {
        'predictions': pred_count,
        'water_levels': water_count,
        'historical': hist_count,
        'latest_prediction': latest_pred,
        'stations': stations
    }

def main():
    """Main function to run predictions and update database."""
    print("🚀 RUNNING PREDICTIONS AND UPDATING DATABASE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Load station coordinates
    stations = load_station_coordinates()
    print(f"Found {len(stations)} stations to process")
    print()
    
    # Step 1: Run predictions for all stations
    print("STEP 1: Generating predictions for all stations")
    print("-" * 50)
    
    successful_predictions = 0
    failed_predictions = 0
    insufficient_data = 0
    
    for station in stations:
        station_id = station['id']
        station_name = station['name']
        lat = station['lat']
        lon = station['lon']
        
        print(f"[{station_id}] {station_name}")
        
        if run_prediction_for_station(station_id, lat, lon, station_name):
            successful_predictions += 1
        else:
            failed_predictions += 1
            insufficient_data += 1
        
        # Small delay between stations
        time.sleep(2)
        print()
    
    print("STEP 2: Saving predictions to database")
    print("-" * 50)
    
    successful_db_saves = 0
    failed_db_saves = 0
    
    for station in stations:
        station_id = station['id']
        station_name = station['name']
        
        print(f"[{station_id}] {station_name}")
        
        if save_predictions_to_db(station_id):
            successful_db_saves += 1
        else:
            failed_db_saves += 1
        
        print()
    
    print("STEP 3: Updating water levels and historical data")
    print("-" * 50)
    
    if update_water_levels_and_history():
        print("✅ All database updates completed successfully")
    else:
        print("❌ Some database updates failed")
    
    print()
    print("=" * 60)
    print("📊 FINAL SUMMARY")
    print(f"Completed at: {datetime.now()}")
    print()
    print("Prediction Generation:")
    print(f"  Successful: {successful_predictions}/{len(stations)}")
    print(f"  Failed (insufficient data): {insufficient_data}/{len(stations)}")
    print(f"  Total failed: {failed_predictions}/{len(stations)}")
    print()
    print("Database Saves:")
    print(f"  Successful: {successful_db_saves}/{len(stations)}")
    print(f"  Failed: {failed_db_saves}/{len(stations)}")
    print()
    
    # Show final database summary
    summary = get_database_summary()
    print("Database Contents:")
    print(f"  Predictions: {summary['predictions']}")
    print(f"  Water levels: {summary['water_levels']}")
    print(f"  Historical data: {summary['historical']}")
    print(f"  Latest prediction: {summary['latest_prediction']}")
    print()
    
    print("Station Data Summary:")
    for station in summary['stations']:
        print(f"  {station['station_id']} ({station['name']}):")
        print(f"    Current: {station['water_level_cm']:.2f} cm ({station['measurement_date']})")
        print(f"    Historical range: {station['min_level_cm']:.2f} - {station['max_level_cm']:.2f} cm")
        print(f"    30-day range: {station['last_30_days_min_cm']:.2f} - {station['last_30_days_max_cm']:.2f} cm")
    
    # Show data sufficiency summary
    if insufficient_data > 0:
        print()
        print("⚠️  DATA SUFFICIENCY WARNINGS:")
        print(f"  {insufficient_data} stations had insufficient data even with 90 days of history")
        print("  These stations may need:")
        print("    - Longer historical data collection")
        print("    - Data gap filling")
        print("    - Alternative prediction methods")

if __name__ == "__main__":
    main()
