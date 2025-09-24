#!/usr/bin/env python3
"""
Fetch station metadata from Vandah API
"""

import requests
import json
from pyproj import Transformer

def fetch_station_metadata(station_id):
    """Fetch station metadata from Vandah API and convert coordinates."""
    try:
        # Try to get station from the stations list
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 fetch_station_metadata.py <station_id>")
        sys.exit(1)
    
    station_id = sys.argv[1]
    metadata = fetch_station_metadata(station_id)
    
    if metadata:
        print("\n📋 Station Metadata:")
        print(json.dumps(metadata, indent=2))
    else:
        print("❌ Failed to fetch station metadata")
        sys.exit(1)
