#!/usr/bin/env python3
"""
Modify the stations API endpoint to auto-fetch metadata from Vandah API
"""

import re

def fetch_station_metadata_function():
    """Generate the fetch_station_metadata function code."""
    return '''
def fetch_station_metadata_from_vandah(station_id):
    """Fetch station metadata from Vandah API and convert coordinates."""
    try:
        import requests
        from pyproj import Transformer
        
        url = "https://vandah.miljoeportal.dk/api/stations?format=json"
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
                
                return {
                    'name': station['name'],
                    'latitude': lat,
                    'longitude': lon,
                    'location_type': station['locationType'].lower(),
                    'station_owner': station.get('stationOwnerName', ''),
                    'description': station.get('description', '')
                }
            else:
                return None
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching station metadata: {e}")
        return None
'''

def modify_stations_endpoint():
    """Modify the stations endpoint to auto-fetch metadata."""
    
    # Read the current file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Add the fetch function after the imports
    import_section = content.find('from flask import')
    if import_section != -1:
        # Find the end of imports
        lines = content[:import_section].split('\n')
        import_end = len('\n'.join(lines))
        
        # Insert the fetch function after imports
        new_content = content[:import_end] + '\n' + fetch_station_metadata_function() + '\n' + content[import_end:]
        content = new_content
    
    # Modify the POST stations endpoint
    # Find the current validation logic
    old_validation = '''required_fields = ["station_id", "name"]
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            station_id = data["station_id"].strip()
            name = data["name"].strip()
            latitude = data.get("latitude")
            longitude = data.get("longitude")
            location_type = data.get("location_type", "stream").strip()
            station_owner = data.get("station_owner", "").strip()
            municipality_id = data.get("municipality_id")
            
            if not station_id or not name:
                return jsonify({"error": "Station ID and name cannot be empty"}), 400'''
    
    new_validation = '''# Check required fields
            if "station_id" not in data:
                return jsonify({"error": "Missing required field: station_id"}), 400
            
            station_id = data["station_id"].strip()
            municipality_id = data.get("municipality_id")
            
            if not station_id:
                return jsonify({"error": "Station ID cannot be empty"}), 400
            
            # Auto-fetch metadata if only station_id and municipality_id provided
            if "name" not in data or not data.get("name"):
                print(f"Auto-fetching metadata for station {station_id}...")
                metadata = fetch_station_metadata_from_vandah(station_id)
                if metadata:
                    name = metadata['name']
                    latitude = metadata['latitude']
                    longitude = metadata['longitude']
                    location_type = metadata['location_type']
                    station_owner = metadata['station_owner']
                    print(f"✅ Fetched: {name} at {latitude:.6f}, {longitude:.6f}")
                else:
                    return jsonify({"error": f"Could not fetch metadata for station {station_id} from Vandah API"}), 400
            else:
                # Use provided data
                name = data["name"].strip()
                latitude = data.get("latitude")
                longitude = data.get("longitude")
                location_type = data.get("location_type", "stream").strip()
                station_owner = data.get("station_owner", "").strip()
            
            if not name:
                return jsonify({"error": "Station name cannot be empty"}), 400'''
    
    # Replace the validation logic
    content = content.replace(old_validation, new_validation)
    
    # Write the modified content
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("✅ Modified stations API endpoint to auto-fetch metadata")

if __name__ == "__main__":
    modify_stations_endpoint()
