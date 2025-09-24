#!/usr/bin/env python3
"""
Simple fix to make the API synchronous - just modify the existing stations endpoint
"""

def fix_api():
    # Read the file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Add the fetch function after imports
    fetch_function = '''
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
    
    # Find where to insert the functions (after imports)
    import_end = content.find('from flask import')
    if import_end != -1:
        # Find the end of imports
        lines = content[:import_end].split('\n')
        import_end = len('\n'.join(lines))
        
        # Insert the functions after imports
        content = content[:import_end] + '\n' + fetch_function + '\n' + content[import_end:]
    
    # Modify the stations endpoint to auto-fetch metadata
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
    
    # Replace the background update with synchronous execution
    old_background = '''            # Background data update temporarily disabled'''
    new_sync = '''            # Run data update synchronously and wait for completion
            print(f"🔄 Starting data update for station {station_id}...")
            try:
                import subprocess
                result = subprocess.run([
                    'python3', 'update_new_station_data.py', station_id
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ Data update completed for station {station_id}")
                    data_update_status = "completed"
                    data_update_message = "30-day history, min/max values, and predictions have been generated successfully"
                else:
                    print(f"❌ Data update failed for station {station_id}: {result.stderr}")
                    data_update_status = "failed"
                    data_update_message = f"Data update failed: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ Data update timed out for station {station_id}")
                data_update_status = "timeout"
                data_update_message = "Data update timed out after 5 minutes"
            except Exception as e:
                print(f"❌ Error in data update for station {station_id}: {e}")
                data_update_status = "error"
                data_update_message = f"Data update error: {str(e)}"
            
            # Get updated station info with all data
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get station details
            cursor.execute("""
                SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner, s.municipality_id, m.name as municipality_name
                FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id
                WHERE s.station_id = ?
            """, (station_id,))
            station_row = cursor.fetchone()
            
            # Get min/max values
            cursor.execute("SELECT min_level_cm, max_level_cm FROM last_30_days_historical WHERE station_id = ?", (station_id,))
            minmax_row = cursor.fetchone()
            
            # Get prediction count
            cursor.execute("SELECT COUNT(*) as prediction_count FROM predictions WHERE station_id = ?", (station_id,))
            prediction_row = cursor.fetchone()
            
            # Get 30-day data count
            cursor.execute("SELECT COUNT(*) as history_count FROM last_30_days_historical WHERE station_id = ?", (station_id,))
            history_row = cursor.fetchone()
            
            conn.close()
            
            # Build station info
            station_info = {
                "station_id": station_row['station_id'],
                "name": station_row['name'],
                "latitude": station_row['latitude'],
                "longitude": station_row['longitude'],
                "location_type": station_row['location_type'],
                "station_owner": station_row['station_owner'],
                "municipality_id": station_row['municipality_id'],
                "municipality_name": station_row['municipality_name'],
                "weather_station_info": get_weather_station_info()
            }
            
            # Add data summary
            data_summary = {
                "status": data_update_status,
                "message": data_update_message,
                "30_day_history_count": history_row['history_count'] if history_row else 0,
                "predictions_count": prediction_row['prediction_count'] if prediction_row else 0,
                "min_max_available": minmax_row is not None,
                "min_level_cm": minmax_row['min_level_cm'] if minmax_row else None,
                "max_level_cm": minmax_row['max_level_cm'] if minmax_row else None
            }
            
            return jsonify({
                "message": "Station created successfully with complete data.",
                "station": station_info,
                "data_summary": data_summary
            }), 201'''
    
    # Replace the background execution with synchronous execution
    content = content.replace(old_background, new_sync)
    
    # Write the modified content
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("✅ Modified API to run synchronously with auto-fetching metadata")

if __name__ == "__main__":
    fix_api()
