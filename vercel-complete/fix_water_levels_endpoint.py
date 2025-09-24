#!/usr/bin/env python3

# Read the current server file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Replace the water levels endpoint with a fixed version
old_endpoint = '''@app.route('/water-levels')
def get_water_levels():
    """Get current water levels for all stations with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT w1.station_id, s.name, s.latitude, s.longitude, w1.water_level_cm, w1.water_level_m, w1.measurement_date
        FROM water_levels w1
        INNER JOIN (
            SELECT station_id, MAX(created_at) as max_created_at
            FROM water_levels
            GROUP BY station_id
        ) w2 ON w1.station_id = w2.station_id AND w1.created_at = w2.max_created_at
        JOIN stations s ON w1.station_id = s.station_id
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
            "weather_station_info": weather_info
        }
        water_levels.append(water_data)
    
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
    
    # Get latest water level from either water_levels or last_30_days_historical tables
    cursor.execute("""
        WITH latest_water_levels AS (
            SELECT station_id, water_level_cm, water_level_m, measurement_date, created_at, 'water_levels' as source
            FROM water_levels w1
            WHERE w1.created_at = (
                SELECT MAX(created_at) FROM water_levels w2 WHERE w2.station_id = w1.station_id
            )
        ),
        latest_30day AS (
            SELECT station_id, water_level_cm, water_level_m, measurement_date, created_at, 'last_30_days_historical' as source
            FROM last_30_days_historical h1
            WHERE h1.measurement_date = (
                SELECT MAX(measurement_date) FROM last_30_days_historical h2 WHERE h2.station_id = h1.station_id
            )
        ),
        combined_levels AS (
            SELECT * FROM latest_water_levels
            UNION ALL
            SELECT * FROM latest_30day
        ),
        latest_per_station AS (
            SELECT station_id, water_level_cm, water_level_m, measurement_date, source,
                   ROW_NUMBER() OVER (PARTITION BY station_id ORDER BY measurement_date DESC, created_at DESC) as rn
            FROM combined_levels
        )
        SELECT l.station_id, s.name, s.latitude, s.longitude, l.water_level_cm, l.water_level_m, l.measurement_date, l.source
        FROM latest_per_station l
        JOIN stations s ON l.station_id = s.station_id
        WHERE l.rn = 1
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
            "data_source": row['source'],
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

print("✅ Fixed the water levels endpoint to get latest readings from both tables")
