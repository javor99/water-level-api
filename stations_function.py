@app.route('/stations', methods=['GET', 'POST'])
def stations():
    """Handle both GET (list stations) and POST (create station) requests."""
    if request.method == 'POST':
        # Create new station and trigger automatic data update
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            required_fields = ["station_id", "name"]
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
                return jsonify({"error": "Station ID and name cannot be empty"}), 400
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check if station already exists
            cursor.execute("SELECT station_id FROM stations WHERE station_id = ?", (station_id,))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": "Station with this ID already exists"}), 409
            
            # Insert the new station
            cursor.execute("""
                INSERT INTO stations 
                (station_id, name, latitude, longitude, location_type, station_owner, municipality_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (station_id, name, latitude, longitude, location_type, station_owner, municipality_id))
            
            conn.commit()
            conn.close()
            
            # Start background data update immediately
            print(f"🚀 Starting automatic data update for new station {station_id}")
            update_thread = threading.Thread(
                target=run_station_data_update, 
                args=(station_id, name),
                daemon=True
            )
            update_thread.start()
            
            return jsonify({
                "message": "Station created successfully. Automatic data update started immediately.",
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
                    "message": "30-day history, min/max values, and predictions are being updated automatically in the background"
                }
            }), 201
            
        except Exception as e:
            return jsonify({"error": f"Failed to create station: {str(e)}"}), 500
    
    # GET request - list all stations
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner, s.municipality_id, m.name as municipality_name
        FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id
        ORDER BY s.name
    """)
    
    weather_info = get_weather_station_info()
    
    stations = []
    for row in cursor.fetchall():
        station_data = {
            "station_id": row['station_id'],
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "location_type": row['location_type'],
            "station_owner": row["station_owner"],
            "municipality_id": row["municipality_id"],
            "municipality_name": row["municipality_name"],
            "weather_station_info": weather_info
        }
        stations.append(station_data)
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(stations),
        "stations": stations
    })
