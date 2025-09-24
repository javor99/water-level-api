#!/usr/bin/env python3

# Read the current server file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Replace with a simpler, more efficient version
old_query = '''    cursor.execute("""
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
    """)'''

new_query = '''    # Get all stations first
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
            water_levels.append(water_data)'''

# Apply the fix
content = content.replace(old_query, new_query)

# Also remove the loop that processes the results since we're now building the list directly
old_loop = '''    water_levels = []
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
        water_levels.append(water_data)'''

new_loop = '''    # water_levels list is now built in the query section above'''

content = content.replace(old_loop, new_loop)

# Write the fixed content
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Fixed the water levels endpoint with a simpler, more efficient query")
