def update_current_water_level(station_id: str, station_name: str) -> bool:
    """Update current water level with the most recent measurement from 30-day history."""
    try:
        print(f"  📊 Updating current water level for {station_name}...")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the most recent measurement from 30-day history
        cursor.execute("""
            SELECT water_level_cm, water_level_m, measurement_date
            FROM last_30_days_historical 
            WHERE station_id = ? 
            ORDER BY measurement_date DESC 
            LIMIT 1
        """, (station_id,))
        
        latest_measurement = cursor.fetchone()
        
        if not latest_measurement:
            print(f"    ⚠️  No 30-day history data found for station {station_id}")
            conn.close()
            return False
        
        # Insert or update current water level
        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, water_level_cm, water_level_m, measurement_date, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['water_level_cm'],
            latest_measurement['water_level_m'],
            latest_measurement['measurement_date'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"    ✅ Current water level updated: {latest_measurement['water_level_cm']:.2f} cm ({latest_measurement['measurement_date']})")
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to update current water level: {e}")
        return False
