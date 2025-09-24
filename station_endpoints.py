# ===== STATION CRUD ENDPOINTS =====

@app.route('/stations', methods=['POST'])
def create_station():
    """Create a new water level station."""
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
        
        cursor.execute("""
            INSERT INTO stations 
            (station_id, name, latitude, longitude, location_type, station_owner, municipality_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (station_id, name, latitude, longitude, location_type, station_owner, municipality_id))
        
        station_id = cursor.lastrowid if cursor.lastrowid else station_id
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Station created successfully",
            "station": {
                "station_id": station_id,
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "location_type": location_type,
                "station_owner": station_owner,
                "municipality_id": municipality_id
            }
        }), 201
        
    except Exception as e:
        return jsonify({"error": f"Failed to create station: {str(e)}"}), 500

@app.route('/stations/<station_id>', methods=['PUT'])
def update_station(station_id):
    """Update station details."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("""
            SELECT station_id, name, latitude, longitude, location_type, station_owner, municipality_id
            FROM stations WHERE station_id = ?
        """, (station_id,))
        
        station = cursor.fetchone()
        if not station:
            conn.close()
            return jsonify({"error": "Station not found"}), 404
        
        # Update fields
        name = data.get("name", station["name"]).strip()
        latitude = data.get("latitude", station["latitude"])
        longitude = data.get("longitude", station["longitude"])
        location_type = data.get("location_type", station["location_type"]).strip()
        station_owner = data.get("station_owner", station["station_owner"] or "").strip()
        municipality_id = data.get("municipality_id", station["municipality_id"])
        
        if not name:
            return jsonify({"error": "Station name cannot be empty"}), 400
        
        # Check if new station_id conflicts (if changing station_id)
        new_station_id = data.get("station_id", station_id).strip()
        if new_station_id != station_id and new_station_id:
            cursor.execute("SELECT station_id FROM stations WHERE station_id = ? AND station_id != ?", 
                         (new_station_id, station_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": "Station with this ID already exists"}), 409
        
        cursor.execute("""
            UPDATE stations 
            SET station_id = ?, name = ?, latitude = ?, longitude = ?, 
                location_type = ?, station_owner = ?, municipality_id = ?
            WHERE station_id = ?
        """, (new_station_id, name, latitude, longitude, location_type, 
              station_owner, municipality_id, station_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Station updated successfully",
            "station": {
                "station_id": new_station_id,
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "location_type": location_type,
                "station_owner": station_owner,
                "municipality_id": municipality_id
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update station: {str(e)}"}), 500

@app.route('/stations/<station_id>', methods=['DELETE'])
def delete_station(station_id):
    """Delete a station."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("SELECT station_id, name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        
        if not station:
            conn.close()
            return jsonify({"error": "Station not found"}), 404
        
        # Check for related data
        cursor.execute("SELECT COUNT(*) FROM water_levels WHERE station_id = ?", (station_id,))
        water_level_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE station_id = ?", (station_id,))
        prediction_count = cursor.fetchone()[0]
        
        if water_level_count > 0 or prediction_count > 0:
            conn.close()
            return jsonify({
                "error": f"Cannot delete station. It has {water_level_count} water level records and {prediction_count} prediction records. Please delete related data first."
            }), 409
        
        cursor.execute("DELETE FROM stations WHERE station_id = ?", (station_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Station '{station['name']}' deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to delete station: {str(e)}"}), 500

@app.route('/stations/<station_id>/data', methods=['DELETE'])
def delete_station_data(station_id):
    """Delete all data (water levels and predictions) for a station."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("SELECT station_id, name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        
        if not station:
            conn.close()
            return jsonify({"error": "Station not found"}), 404
        
        # Delete related data
        cursor.execute("DELETE FROM water_levels WHERE station_id = ?", (station_id,))
        water_levels_deleted = cursor.rowcount
        
        cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
        predictions_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Station data deleted successfully",
            "station_name": station["name"],
            "water_levels_deleted": water_levels_deleted,
            "predictions_deleted": predictions_deleted
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to delete station data: {str(e)}"}), 500
