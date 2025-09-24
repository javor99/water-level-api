#!/usr/bin/env python3
"""
Create a new station with automatic metadata fetching from Vandah API
"""

import requests
import json
import sys
from pyproj import Transformer

def fetch_station_metadata(station_id):
    """Fetch station metadata from Vandah API and convert coordinates."""
    try:
        url = "https://vandah.miljoeportal.dk/api/stations?format=json"
        print(f"📡 Fetching station metadata for {station_id}...")
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            stations = response.json()
            
            # Find the specific station
            station = None
            for s in stations:
                if s.get('stationId') == station_id:
                    station = s
                    break
            
            if station:
                # Convert UTM coordinates to lat/lon
                transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
                location = station['location']
                x, y = location['x'], location['y']
                lon, lat = transformer.transform(x, y)
                
                metadata = {
                    'station_id': station['stationId'],
                    'name': station['name'],
                    'latitude': lat,
                    'longitude': lon,
                    'location_type': station['locationType'],
                    'station_owner': station.get('stationOwnerName', ''),
                    'description': station.get('description', '')
                }
                
                print(f"✅ Found station: {metadata['name']}")
                print(f"📍 Coordinates: {lat:.6f}, {lon:.6f}")
                print(f"🏢 Owner: {metadata['station_owner']}")
                print(f"🏷️ Type: {metadata['location_type']}")
                
                return metadata
            else:
                print(f"❌ Station {station_id} not found in Vandah API")
                return None
                
        else:
            print(f"❌ API returned status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching station data: {e}")
        return None

def create_station_via_api(station_data, municipality_id=1):
    """Create station via the local API."""
    try:
        response = requests.post("http://localhost:5001/stations", 
                               json=station_data, 
                               headers={"Content-Type": "application/json"})
        
        if response.status_code == 201:
            return response.json()
        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error creating station: {e}")
        return None

def update_station_data(station_id):
    """Update station data using the existing script."""
    import subprocess
    try:
        print(f"🔄 Updating station data for {station_id}...")
        result = subprocess.run(['python3', 'update_new_station_data.py', station_id], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Station data updated successfully")
            return True
        else:
            print(f"❌ Data update failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Data update timed out")
        return False
    except Exception as e:
        print(f"❌ Error updating data: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_station_auto.py <station_id> [municipality_id]")
        print("Example: python3 create_station_auto.py 70001001 1")
        sys.exit(1)
    
    station_id = sys.argv[1]
    municipality_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("🚀 AUTOMATIC STATION CREATION")
    print("=" * 50)
    print(f"Station ID: {station_id}")
    print(f"Municipality ID: {municipality_id}")
    print("")
    
    # Step 1: Fetch station metadata from Vandah API
    metadata = fetch_station_metadata(station_id)
    if not metadata:
        print("❌ Cannot proceed without station metadata")
        sys.exit(1)
    
    print("")
    
    # Step 2: Create station via local API
    print("Step 2: Creating station via local API...")
    station_data = {
        "station_id": metadata['station_id'],
        "name": metadata['name'],
        "latitude": metadata['latitude'],
        "longitude": metadata['longitude'],
        "location_type": metadata['location_type'].lower(),
        "station_owner": metadata['station_owner'],
        "municipality_id": municipality_id
    }
    
    result = create_station_via_api(station_data, municipality_id)
    if not result:
        print("❌ Failed to create station")
        sys.exit(1)
    
    print("✅ Station created successfully!")
    print("")
    
    # Step 3: Update station data
    print("Step 3: Updating station data (30-day history, min/max, predictions)...")
    print("This may take 1-3 minutes...")
    
    success = update_station_data(station_id)
    if not success:
        print("⚠️ Station created but data update failed")
        print("You can run manually: python3 update_new_station_data.py", station_id)
        sys.exit(1)
    
    print("")
    print("🎉 STATION SETUP COMPLETED!")
    print("=" * 50)
    print(f"Station: {metadata['name']}")
    print(f"ID: {station_id}")
    print(f"Coordinates: {metadata['latitude']:.6f}, {metadata['longitude']:.6f}")
    print(f"Owner: {metadata['station_owner']}")
    print(f"Type: {metadata['location_type']}")
    print("")
    print("📊 Available endpoints:")
    print(f"  - Station info: http://localhost:5001/stations/{station_id}")
    print(f"  - Water levels: http://localhost:5001/water-levels/{station_id}")
    print(f"  - Predictions: http://localhost:5001/predictions/{station_id}")

if __name__ == "__main__":
    main()
