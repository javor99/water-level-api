#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update New Station Data
Comprehensive script to update 30-day history, min/max values, and predictions for a newly added water level station.
"""

import os
import sys
import sqlite3
import pandas as pd
import json
import requests
import subprocess
from datetime import datetime, timedelta
import argparse

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

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
            print(f"  ⚠️  No water level data available for station {station_id}")
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
        print(f"  ❌ Error fetching water data: {e}")
        return pd.DataFrame()

def fetch_historical_water_data(station_id: str, years_back: int = 5) -> list:
    """Fetch historical water level data for min/max calculation - single request."""
    print(f"  📊 Fetching {years_back} years of historical data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    try:
        url = "https://vandah.miljoeportal.dk/api/water-levels"
        params = {
            "stationId": station_id,
            "from": start_date.strftime("%Y-%m-%dT00:00Z"),
            "to": end_date.strftime("%Y-%m-%dT00:00Z"),
            "format": "json",
        }
        
        print(f"    📡 Single request: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        r = requests.get(url, params=params, timeout=60)  # Increased timeout for large request
        r.raise_for_status()
        raw = r.json()
        
        all_data = []
        if raw and raw[0].get("results"):
            for result in raw[0]["results"]:
                all_data.append({
                    'timestamp': result['measurementDateTime'][:10],  # YYYY-MM-DD
                    'level_cm': result['result']
                })
            print(f"    ✅ Fetched {len(all_data)} records")
        else:
            print(f"    ⚠️  No historical data available")
        
        return all_data
        
    except Exception as e:
        print(f"    ❌ Error fetching historical data: {e}")
        return []

def update_30_day_history(station_id: str, station_name: str):
    """Update 30-day historical data for a station."""
    print(f"📅 Updating 30-day history for {station_id} ({station_name})")
    
    try:
        # Fetch water data for last 30 days
        water_data = fetch_water_daily(station_id, 40)
        
        if water_data.empty:
            print(f"  ⚠️  No water data available for 30-day history")
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
        
        print(f"  ✅ Updated {records_inserted} records in 30-day history")
        return True
        
    except Exception as e:
        print(f"  ❌ Error updating 30-day history: {e}")
        return False

def calculate_and_update_minmax(station_id: str, station_name: str):
    """Calculate and update min/max values for a station."""
    print(f"📊 Calculating min/max values for {station_id} ({station_name})")
    
    try:
        # Fetch historical data
        historical_data = fetch_historical_water_data(station_id, 5)
        
        if not historical_data:
            print(f"  ⚠️  No historical data available for min/max calculation")
            return False
        
        # Calculate min/max from historical data
        water_levels = [record['level_cm'] for record in historical_data]
        overall_min_cm = min(water_levels)  # ✅ CORRECT NAME
        overall_max_cm = max(water_levels)  # ✅ CORRECT NAME
        min_level_cm = overall_min_cm / 100.0
        max_level_cm = overall_max_cm / 100.0
        
        # Calculate 30-day min/max
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_data = [
            record for record in historical_data 
            if datetime.strptime(record['timestamp'], '%Y-%m-%d') >= thirty_days_ago
        ]
        
        if recent_data:
            recent_levels = [record['level_cm'] for record in recent_data]
            last_30_days_min_cm = min(recent_levels)
            last_30_days_max_cm = max(recent_levels)
            last_30_days_min_m = last_30_days_min_cm / 100.0
            last_30_days_max_m = last_30_days_max_cm / 100.0
        else:
            # Fallback to overall min/max if no recent data
            last_30_days_min_cm = overall_min_cm  # ✅ Use correct variable
            last_30_days_max_cm = overall_max_cm  # ✅ Use correct variable
            last_30_days_min_m = min_level_cm
            last_30_days_max_m = max_level_cm
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE stations 
            SET last_30_days_min_cm = ?, last_30_days_max_cm = ?, 
                last_30_days_min_m = ?, last_30_days_max_m = ?
            WHERE station_id = ?
        """, (
            last_30_days_min_cm, last_30_days_max_cm,
            last_30_days_min_m, last_30_days_max_m,
            station_id
        ))
        
        # Also update min_max_values table with overall historical values
        cursor.execute("""
            INSERT OR REPLACE INTO min_max_values 
            (station_id, min_level_cm, max_level_cm, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            station_id, overall_min_cm, overall_max_cm, 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
        
        print(f"  ✅ Updated min/max values:")
        print(f"    Historical: {overall_min_cm:.2f} - {overall_max_cm:.2f} cm")
        print(f"    30-day: {last_30_days_min_cm:.2f} - {last_30_days_max_cm:.2f} cm")
        return True
        
    except Exception as e:
        print(f"  ❌ Error calculating min/max: {e}")
        return False

def run_predictions(station_id: str, station_name: str, latitude: float, longitude: float):
    """Run predictions for a station."""
    print(f"🔮 Running predictions for {station_id} ({station_name})")
    
    try:
        # Only try 40 days - no retry logic
        past_days = 40
        print(f"  📅 Trying {past_days} days of historical data...")
        
        # Run the prediction script
        cmd = [
            'python3', 'predict_unseen_station.py',
            '--vandah_id', station_id,
            '--lat', str(latitude),
            '--lon', str(longitude),
            '--unseen_strategy', 'nearest',
            '--anchor', 'none',
            '--past_days', str(past_days)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"  ✅ Predictions generated successfully with {past_days} days of data")
            
            # Save predictions to database
            if save_predictions_to_db(station_id):
                return True
            else:
                return False
        else:
            # Check if it's a data insufficiency error
            error_msg = result.stderr.lower()
            if "not enough rows" in error_msg or "sequence" in error_msg:
                print(f"  ❌ Cannot generate predictions - insufficient consecutive data in last {past_days} days")
                print(f"  📊 Need 40 consecutive days ending today for predictions")
                return False
            else:
                print(f"  ❌ Prediction failed: {result.stderr}")
                return False
        
    except Exception as e:
        print(f"  ❌ Error in prediction process: {e}")
        return False

def save_predictions_to_db(station_id: str):
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
        
        # Insert new predictions with correct column mapping
        records_inserted = 0
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO predictions
                (station_id, prediction_date, predicted_water_level_cm,
                 change_from_last_cm, forecast_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                station_id,
                row['date'],  # CSV has 'date' column
                row['predicted_water_level_cm'],  # CSV has this column name
                row['change_from_last_daily_mean_cm'],  # CSV has this column name
                row['date'],  # Use same date for forecast_date
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

def get_station_info(station_id: str):
    """Get station information from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT station_id, name, latitude, longitude, location_type, station_owner, municipality_id
        FROM stations 
        WHERE station_id = ?
    """, (station_id,))
    
    station = cursor.fetchone()
    conn.close()
    return dict(station) if station else None

def update_current_water_level(station_id: str, station_name: str) -> bool:
    """Update current water level with the most recent measurement from 30-day history."""
    try:
        print(f"  📊 Updating current water level for {station_name}...")
        
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
            print(f"    ⚠️  No 30-day history data found for station {station_id}")
            conn.close()
            return False
        
        # Insert or update current water level
        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, level_cm, level_cm, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['level_cm'],
            latest_measurement['level_cm'],
            latest_measurement['timestamp'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"    ✅ Current water level updated: {latest_measurement['level_cm']:.2f} cm ({latest_measurement['timestamp']})")
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to update current water level: {e}")
        return False

def update_new_station_data(station_id: str):
    """Main function to update all data for a new station."""
    print("🚀 UPDATING NEW STATION DATA")
    print("=" * 60)
    print(f"Station ID: {station_id}")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Get station information
    station_info = get_station_info(station_id)
    if not station_info:
        print(f"❌ Station {station_id} not found in database")
        return False
    
    station_name = station_info['name']
    latitude = station_info['latitude']
    longitude = station_info['longitude']
    
    print(f"Station: {station_name}")
    print(f"Location: {latitude}, {longitude}")
    print()
    
    # Track results
    results = {
        '30_day_history': False,
        'current_water_level': False,
        'minmax_calculation': False,
        'predictions': False
    }
    
    # Step 1: Update 30-day history
    print("STEP 1: Updating 30-day historical data")
    print("-" * 40)
    results['30_day_history'] = update_30_day_history(station_id, station_name)
    print()
    
    # Step 2: Update current water level
    print("STEP 2: Updating current water level")
    print("-" * 40)
    results['current_water_level'] = update_current_water_level(station_id, station_name)
    print()
    
    # Step 3: Calculate and update min/max values
    print("STEP 3: Calculating min/max values")
    print("-" * 40)
    results['minmax_calculation'] = calculate_and_update_minmax(station_id, station_name)
    print()
    
    # Step 4: Run predictions
    print("STEP 4: Generating predictions")
    print("-" * 40)
    results['predictions'] = run_predictions(station_id, station_name, latitude, longitude)
    print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY")
    print(f"Completed at: {datetime.now()}")
    print()
    print("Results:")
    print(f"  ✅ 30-day history: {'SUCCESS' if results['30_day_history'] else 'FAILED'}")
    print(f"  ✅ Current water level: {'SUCCESS' if results['current_water_level'] else 'FAILED'}")
    print(f"  ✅ Min/Max calculation: {'SUCCESS' if results['minmax_calculation'] else 'FAILED'}")
    print(f"  ✅ Predictions: {'SUCCESS' if results['predictions'] else 'FAILED'}")
    print()
    
    # Show final station data
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.station_id, s.name, 
               s.last_30_days_min_cm, s.last_30_days_max_cm,
               mm.min_level_cm, mm.max_level_cm,
               COUNT(h.station_id) as history_count,
               COUNT(p.station_id) as prediction_count
        FROM stations s
        LEFT JOIN min_max_values mm ON s.station_id = mm.station_id
        LEFT JOIN last_30_days_historical h ON s.station_id = h.station_id
        LEFT JOIN predictions p ON s.station_id = p.station_id
        WHERE s.station_id = ?
        GROUP BY s.station_id
    """, (station_id,))
    
    station_data = cursor.fetchone()
    conn.close()
    
    if station_data:
        print("Final Station Data:")
        print(f"  Station: {station_data['name']}")
        min_level = station_data['last_30_days_min_cm'] if station_data['last_30_days_min_cm'] is not None else 0
        max_level = station_data['last_30_days_max_cm'] if station_data['last_30_days_max_cm'] is not None else 0
        last_30_min = station_data['last_30_days_min_cm'] if station_data['last_30_days_min_cm'] is not None else 0
        last_30_max = station_data['last_30_days_max_cm'] if station_data['last_30_days_max_cm'] is not None else 0
        
        print(f"  Historical min/max: {min_level:.2f} - {max_level:.2f} cm")
        print(f"  30-day min/max: {last_30_min:.2f} - {last_30_max:.2f} cm")
        print(f"  Historical records: {station_data['history_count']}")
        print(f"  Prediction records: {station_data['prediction_count']}")
    
    success_count = sum(results.values())
    print(f"\n🎯 Overall Success: {success_count}/4 operations completed")
    
    return success_count == 4

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Update data for a newly added water level station')
    parser.add_argument('station_id', help='Station ID to update')
    parser.add_argument('--skip-predictions', action='store_true', 
                       help='Skip prediction generation (faster for testing)')
    
    args = parser.parse_args()
    
    if args.skip_predictions:
        print("⚠️  Skipping predictions (--skip-predictions flag used)")
        # Modify the function to skip predictions
        original_run_predictions = run_predictions
        def skip_predictions(*args, **kwargs):
            print("🔮 Skipping predictions as requested")
            return True
        globals()['run_predictions'] = skip_predictions
    
    success = update_new_station_data(args.station_id)
    
    if success:
        print("\n🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some operations failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
