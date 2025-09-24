#!/usr/bin/env python3
"""
Modify the API to wait for all data processing to complete before returning response
"""

def modify_api():
    # Read the file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Replace the background thread with synchronous execution
    old_background = '''        # Start background data update
        import threading
        thread = threading.Thread(target=run_station_data_update, args=(station_id, name))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Station created successfully. Data update started in background.",
            "station": {
                "station_id": station_id,
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "location_type": location_type,
                "station_owner": station_owner,
                "municipality_id": municipality_id
            },
            "data_update": {
                "status": "started",
                "message": "30-day history, min/max values, and predictions are being updated in the background"
            }
        }), 201'''
    
    new_synchronous = '''        # Run data update synchronously and wait for completion
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
    content = content.replace(old_background, new_synchronous)
    
    # Write the modified content
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("✅ Modified API to run synchronously and return complete station info")

if __name__ == "__main__":
    modify_api()
