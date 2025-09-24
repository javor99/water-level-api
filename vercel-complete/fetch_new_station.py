#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch New Station Data by ID
Fetches all data about a new water level station and adds it to the database
"""

import sqlite3
import requests
import json
from datetime import datetime
import sys

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    return conn

def fetch_station_info_from_api(station_id):
    """Fetch station information from the external API."""
    try:
        # API endpoint for station information
        api_url = f"https://vandah.miljoeportal.dk/api/stations/{station_id}"
        
        print(f"📡 Fetching station information from API...")
        
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"❌ API returned status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching station data: {e}")
        return None

def convert_coordinates(utm_x, utm_y):
    """Convert UTM coordinates to lat/lon."""
    try:
        from pyproj import Transformer
        
        # UTM Zone 32N (SRID 25832) to WGS84 (SRID 4326)
        transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)
        
        return lat, lon
    except ImportError:
        print("⚠️  pyproj not available, using approximate conversion")
        # Approximate conversion (less accurate)
        lat = utm_y / 111320.0
        lon = utm_x / (111320.0 * 0.866)
        return lat, lon
    except Exception as e:
        print(f"⚠️  Coordinate conversion failed: {e}")
        return None, None

def add_station_to_database(station_data, municipality_id=1):
    """Add the new station to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Extract station information
        station_id = station_data.get('id', '')
        name = station_data.get('name', '')
        description = station_data.get('description', '')
        
        # Handle coordinates
        geometry = station_data.get('geometry', {})
        coordinates = geometry.get('coordinates', [])
        
        if len(coordinates) >= 2:
            utm_x, utm_y = coordinates[0], coordinates[1]
            lat, lon = convert_coordinates(utm_x, utm_y)
        else:
            lat, lon = None, None
        
        # Get additional properties
        properties = station_data.get('properties', {})
        elevation = properties.get('elevation', None)
        status = properties.get('status', 'active')
        
        # Insert station into database
        cursor.execute('''
            INSERT OR REPLACE INTO stations 
            (station_id, name, description, latitude, longitude, elevation, 
             municipality_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (
            station_id,
            name,
            description,
            lat,
            lon,
            elevation,
            municipality_id,
            status
        ))
        
        conn.commit()
        print(f"✅ Station added to database with ID: {station_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding station to database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def fetch_initial_water_levels(station_id):
    """Fetch initial water level data for the new station."""
    try:
        # Fetch recent water levels
        api_url = f"https://vandah.miljoeportal.dk/api/waterlevels/{station_id}"
        
        # Get last 7 days of data
        end_date = datetime.now()
        start_date = end_date.replace(day=end_date.day-7)
        
        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'format': 'json'
        }
        
        print(f"📊 Fetching initial water level data...")
        
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"⚠️  Could not fetch water level data (status {response.status_code})")
            return None
            
    except Exception as e:
        print(f"⚠️  Error fetching water level data: {e}")
        return None

def save_initial_water_levels(station_id, water_level_data):
    """Save initial water level data to the database."""
    if not water_level_data or 'data' not in water_level_data:
        return False
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS last_30_days_historical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                measurement_date TIMESTAMP NOT NULL,
                water_level_cm REAL NOT NULL,
                water_level_m REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(station_id, measurement_date)
            )
        ''')
        
        # Insert water level data
        records_inserted = 0
        for record in water_level_data['data']:
            if 'value' in record and record['value'] is not None:
                try:
                    level_cm = float(record['value'])
                    level_m = level_cm / 100
                    measurement_date = record['date']
                    
                    cursor.execute('''
                        INSERT OR IGNORE INTO last_30_days_historical 
                        (station_id, measurement_date, water_level_cm, water_level_m)
                        VALUES (?, ?, ?, ?)
                    ''', (station_id, measurement_date, level_cm, level_m))
                    
                    records_inserted += 1
                    
                except (ValueError, KeyError):
                    continue
        
        conn.commit()
        print(f"💾 Saved {records_inserted} initial water level records")
        return records_inserted > 0
        
    except Exception as e:
        print(f"❌ Error saving water level data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def fetch_new_station(station_id, municipality_id=1):
    """Fetch all data for a new station and add it to the database."""
    print(f"🆕 FETCHING NEW STATION DATA")
    print("=" * 50)
    print(f"Station ID: {station_id}")
    print(f"Municipality ID: {municipality_id}")
    print(f"Started at: {datetime.now()}")
    print()
    
    # Fetch station information
    station_data = fetch_station_info_from_api(station_id)
    
    if not station_data:
        print("❌ Could not fetch station information")
        return False
    
    print("✅ Station information fetched successfully")
    print(f"   Name: {station_data.get('name', 'Unknown')}")
    print(f"   Description: {station_data.get('description', 'No description')}")
    
    # Add station to database
    if not add_station_to_database(station_data, municipality_id):
        print("❌ Failed to add station to database")
        return False
    
    # Fetch initial water level data
    water_level_data = fetch_initial_water_levels(station_id)
    
    if water_level_data:
        save_initial_water_levels(station_id, water_level_data)
    
    print()
    print("=" * 50)
    print("✅ NEW STATION SUCCESSFULLY ADDED!")
    print(f"Station ID: {station_id}")
    print(f"Name: {station_data.get('name', 'Unknown')}")
    print("The station is now available in the API endpoints")
    
    return True

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python3 fetch_new_station.py <station_id> [municipality_id]")
        print("Example: python3 fetch_new_station.py 70000864 1")
        sys.exit(1)
    
    station_id = sys.argv[1]
    municipality_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    fetch_new_station(station_id, municipality_id)

if __name__ == "__main__":
    main()
