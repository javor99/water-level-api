#!/usr/bin/env python3
"""
Water Level Data Fetcher
Comprehensive script to fetch water levels, min/max values, and historical data
"""

import requests
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from email_service import send_water_level_alert, send_subscription_confirmation

class WaterLevelDataFetcher:
    def __init__(self, api_base_url="https://6fe4a9afdb88.ngrok-free.app", db_path="water_levels.db"):
        self.api_base_url = api_base_url
        self.db_path = db_path
        self.token = None
    
    def login(self, email="admin@admin.com", password="12345678"):
        """Login to get authentication token"""
        try:
            response = requests.post(f"{self.api_base_url}/auth/login", json={
                "email": email,
                "password": password
            })
            if response.status_code == 200:
                data = response.json()
                self.token = data.get('token')
                print(f"✅ Logged in as {data.get('user', {}).get('email')}")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def get_current_water_levels(self):
        """Get current water levels for all stations via API"""
        try:
            response = requests.get(f"{self.api_base_url}/water-levels")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {data.get('count', 0)} current water level readings")
                return data.get('water_levels', [])
            else:
                print(f"❌ API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ API error: {e}")
            return []
    
    def get_predictions(self, station_id=None):
        """Get water level predictions"""
        try:
            url = f"{self.api_base_url}/predictions"
            if station_id:
                url += f"/{station_id}"
            
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {data.get('count', 0)} predictions")
                return data.get('predictions', [])
            else:
                print(f"❌ API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ API error: {e}")
            return []
    
    def get_historical_data(self, station_id=None, days=30):
        """Get historical water level data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if station_id:
                query = """
                SELECT h.station_id, s.name as station_name, h.measurement_date, 
                       h.water_level_cm, h.water_level_m
                FROM last_30_days_historical h
                JOIN stations s ON h.station_id = s.station_id
                WHERE h.station_id = ? AND h.measurement_date >= date('now', '-{} days')
                ORDER BY h.measurement_date DESC
                """.format(days)
                df = pd.read_sql_query(query, conn, params=(station_id,))
            else:
                query = """
                SELECT h.station_id, s.name as station_name, h.measurement_date, 
                       h.water_level_cm, h.water_level_m
                FROM last_30_days_historical h
                JOIN stations s ON h.station_id = s.station_id
                WHERE h.measurement_date >= date('now', '-{} days')
                ORDER BY h.station_id, h.measurement_date DESC
                """.format(days)
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            print(f"✅ Retrieved {len(df)} historical records")
            return df
        except Exception as e:
            print(f"❌ Database error: {e}")
            return pd.DataFrame()
    
    def get_minmax_values(self, station_id=None):
        """Get min/max values for stations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if station_id:
                query = """
                SELECT 
                    h.station_id,
                    s.name,
                    MIN(h.water_level_cm) as min_cm,
                    MAX(h.water_level_cm) as max_cm,
                    MIN(h.water_level_m) as min_m,
                    MAX(h.water_level_m) as max_m,
                    COUNT(*) as measurements
                FROM last_30_days_historical h
                JOIN stations s ON h.station_id = s.station_id
                WHERE h.station_id = ?
                GROUP BY h.station_id, s.name
                """
                cursor.execute(query, (station_id,))
            else:
                query = """
                SELECT 
                    h.station_id,
                    s.name,
                    MIN(h.water_level_cm) as min_cm,
                    MAX(h.water_level_cm) as max_cm,
                    MIN(h.water_level_m) as min_m,
                    MAX(h.water_level_m) as max_m,
                    COUNT(*) as measurements
                FROM last_30_days_historical h
                JOIN stations s ON h.station_id = s.station_id
                GROUP BY h.station_id, s.name
                ORDER BY s.name
                """
                cursor.execute(query)
            
            results = cursor.fetchall()
            conn.close()
            
            minmax_data = []
            for row in results:
                minmax_data.append({
                    'station_id': row[0],
                    'station_name': row[1],
                    'min_level_cm': row[2],
                    'max_level_cm': row[3],
                    'min_level_m': row[4],
                    'max_level_m': row[5],
                    'total_measurements': row[6]
                })
            
            print(f"✅ Retrieved min/max data for {len(minmax_data)} stations")
            return minmax_data
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def get_station_info(self):
        """Get all station information"""
        try:
            response = requests.get(f"{self.api_base_url}/stations")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {data.get('count', 0)} stations")
                return data.get('stations', [])
            else:
                print(f"❌ API error: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ API error: {e}")
            return []
    
    def print_summary(self):
        """Print a comprehensive summary of all water level data"""
        print("\n" + "="*60)
        print("🌊 WATER LEVEL DATA SUMMARY")
        print("="*60)
        
        # Current water levels
        print("\n📊 CURRENT WATER LEVELS:")
        current_levels = self.get_current_water_levels()
        for level in current_levels:
            print(f"  {level['station_id']}: {level['name']}")
            print(f"    Current: {level['water_level_cm']:.1f} cm ({level['water_level_m']:.2f} m)")
            print(f"    Date: {level['measurement_date']}")
            print()
        
        # Min/Max values
        print("\n📈 MIN/MAX VALUES (Last 30 Days):")
        minmax_data = self.get_minmax_values()
        for data in minmax_data:
            print(f"  {data['station_name']} ({data['station_id']}):")
            print(f"    Min: {data['min_level_cm']:.1f} cm ({data['min_level_m']:.2f} m)")
            print(f"    Max: {data['max_level_cm']:.1f} cm ({data['max_level_m']:.2f} m)")
            print(f"    Measurements: {data['total_measurements']}")
            print()
        
        # Historical data sample
        print("\n📅 RECENT HISTORICAL DATA (Last 5 Days):")
        historical = self.get_historical_data(days=5)
        if not historical.empty:
            for station_id in historical['station_id'].unique():
                station_data = historical[historical['station_id'] == station_id]
                station_name = station_data['station_name'].iloc[0]
                print(f"  {station_name} ({station_id}):")
                for _, row in station_data.head(3).iterrows():
                    print(f"    {row['measurement_date']}: {row['water_level_cm']:.1f} cm")
                print()
        
        # Predictions sample
        print("\n🔮 PREDICTIONS SAMPLE:")
        predictions = self.get_predictions()
        if predictions:
            for pred in predictions[:3]:
                print(f"  {pred['station_id']}: {pred['name']}")
                print(f"    Predicted: {pred['predicted_water_level_cm']:.1f} cm")
                print(f"    Forecast: {pred['forecast_date']}")
                print(f"    Change: {pred['change_from_last_cm']:.1f} cm")
                print()

def main():
    """Main function to demonstrate the data fetcher"""
    fetcher = WaterLevelDataFetcher()
    
    # Login (optional, for admin endpoints)
    fetcher.login()
    
    # Print comprehensive summary
    fetcher.print_summary()
    
    # Example: Get specific station data
    print("\n" + "="*60)
    print("🎯 SPECIFIC STATION EXAMPLE: 70000864 (Hove å, Tostholm bro)")
    print("="*60)
    
    station_id = "70000864"
    
    # Historical data for specific station
    historical = fetcher.get_historical_data(station_id, days=7)
    if not historical.empty:
        print(f"\n📈 Last 7 days of data for {station_id}:")
        for _, row in historical.iterrows():
            print(f"  {row['measurement_date']}: {row['water_level_cm']:.1f} cm ({row['water_level_m']:.2f} m)")
    
    # Min/Max for specific station
    minmax = fetcher.get_minmax_values(station_id)
    if minmax:
        data = minmax[0]
        print(f"\n📊 Min/Max for {data['station_name']}:")
        print(f"  Min: {data['min_level_cm']:.1f} cm ({data['min_level_m']:.2f} m)")
        print(f"  Max: {data['max_level_cm']:.1f} cm ({data['max_level_m']:.2f} m)")
        print(f"  Total measurements: {data['total_measurements']}")

if __name__ == "__main__":
    main()
