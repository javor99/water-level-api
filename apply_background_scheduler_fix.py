#!/usr/bin/env python3
"""Apply background scheduler water_levels INSERT fix to main file"""

# Read the file
with open('background_scheduler.py', 'r') as f:
    content = f.read()

# Fix the INSERT statement to use correct columns and include timestamp
old_insert = '''        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, water_level_cm, water_level_m, measurement_date, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['water_level_cm'],
            latest_measurement['water_level_m'],
            latest_measurement['measurement_date'],
            datetime.now().isoformat()
        ))'''

new_insert = '''        cursor.execute("""
            INSERT OR REPLACE INTO water_levels 
            (station_id, level_cm, timestamp, water_level_cm, water_level_m, measurement_date, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            station_id,
            latest_measurement['water_level_cm'],
            latest_measurement['measurement_date'],
            latest_measurement['water_level_cm'],
            latest_measurement['water_level_m'],
            latest_measurement['measurement_date'],
            datetime.now().isoformat()
        ))'''

if old_insert in content:
    content = content.replace(old_insert, new_insert)
    print("✅ Applied background scheduler water_levels INSERT fix")
else:
    print("❌ Could not find the INSERT statement to fix")

# Write the fixed content back
with open('background_scheduler.py', 'w') as f:
    f.write(content)

print("✅ Background scheduler now uses correct water_levels columns and includes timestamp")
