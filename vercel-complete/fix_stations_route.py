#!/usr/bin/env python3

# Read the current server file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Replace the stations route to handle both GET and POST
old_route = '''@app.route('/stations')
def get_stations():'''

new_route = '''@app.route('/stations', methods=['GET', 'POST'])
def stations():
    """Handle both GET (list stations) and POST (create station) requests."""
    if request.method == 'POST':
        # Create new station
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
            
            # Start background data update
            update_thread = threading.Thread(
                target=run_station_data_update, 
                args=(station_id, name),
                daemon=True
            )
            update_thread.start()
            
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
            }), 201
            
        except Exception as e:
            return jsonify({"error": f"Failed to create station: {str(e)}"}), 500
    
    # GET request - list stations
def get_stations():'''

# Apply the fix
content = content.replace(old_route, new_route)

# Remove the duplicate POST route
lines = content.split('\n')
new_lines = []
skip_lines = False
for i, line in enumerate(lines):
    if "@app.route('/stations', methods=['POST'])" in line:
        skip_lines = True
    elif skip_lines and (line.startswith('@app.route') or line.startswith('def ') or line.startswith('if __name__')):
        skip_lines = False
        new_lines.append(line)
    elif not skip_lines:
        new_lines.append(line)

content = '\n'.join(new_lines)

# Write the fixed content
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("Fixed the stations route to handle both GET and POST methods")
