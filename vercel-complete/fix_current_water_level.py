#!/usr/bin/env python3

# Read the current server file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Replace the water levels endpoint with the correct logic
old_endpoint = '''@app.route('/water-levels')
def get_water_levels():
    """Get current water levels for all stations with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all stations first
    cursor.execute("SELECT station_id, name, latitude, longitude FROM stations ORDER BY name")
    stations = cursor.fetchall()
    
    water_levels = []
    for station in stations:
        station_id = station['station_id']
        
        # Try to get latest from last_30_days_historical first (most recent data)
        cursor.execute("""
            SELECT water_level_cm, water_level_m, measurement_date, 'last_30_days_historical' as source
            FROM last_30_days_historical 
            WHERE station_id = ? 
            ORDER BY measurement_date DESC 
            LIMIT 1
        """, (station_id,))
        
        latest = cursor.fetchone()
        
        # If no data in last_30_days_historical, try water_levels table
        if not latest:
            cursor.execute("""
                SELECT water_level_cm, water_level_m, measurement_date, 'water_levels' as source
                FROM water_levels 
                WHERE station_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (station_id,))
            latest = cursor.fetchone()
        
        # If we have data, add to results
        if latest:
            water_data = {
                "station_id": station_id,
                "name": station['name'],
                "latitude": station['latitude'],
                "longitude": station['longitude'],
                "measurement_date": latest['measurement_date'],
                "water_level_cm": latest['water_level_cm'],
                "water_level_m": latest['water_level_m'],
                "data_source": latest['source'],
                "weather_station_info": weather_info
            }
            water_levels.append(water_data)
    
    # water_levels list is now built in the query section above
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(water_levels),
        "water_levels": water_levels
    })'''

new_endpoint = '''@app.route('/water-levels')
def get_water_levels():
    """Get current water levels for all stations with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get current water levels from latest readings in last_30_days_historical
    cursor.execute("""
        SELECT s.station_id, s.name, s.latitude, s.longitude, 
               h.water_level_cm, h.water_level_m, h.measurement_date
        FROM stations s
        INNER JOIN (
            SELECT station_id, water_level_cm, water_level_m, measurement_date,
                   ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY measurement_date DESC) as rn
            FROM last_30_days_historical
        ) h ON s.station_id = h.station_id AND h.rn = 1
        ORDER BY s.name
    """)
    
    weather_info = get_weather_station_info()
    
    water_levels = []
    for row in cursor.fetchall():
        water_data = {
            "station_id": row['station_id'],
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "measurement_date": row['measurement_date'],
            "water_level_cm": row['water_level_cm'],
            "water_level_m": row['water_level_m'],
            "data_source": "last_30_days_historical",
            "weather_station_info": weather_info
        }
        water_levels.append(water_data)
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(water_levels),
        "water_levels": water_levels
    })'''

# Apply the fix
content = content.replace(old_endpoint, new_endpoint)

# Write the fixed content
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Fixed water levels endpoint to always use latest reading from last_30_days_historical")
