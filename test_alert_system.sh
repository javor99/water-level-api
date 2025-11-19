#!/bin/bash
# Test script to manually trigger alert checking

echo "=========================================="
echo "🧪 Testing Alert System"
echo "=========================================="
echo ""

# Check if database exists
if [ ! -f "water_levels.db" ]; then
    echo "❌ Database file not found!"
    exit 1
fi

echo "1. Checking subscriptions..."
python3 << 'PYEOF'
import sqlite3

conn = sqlite3.connect('water_levels.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get a station with subscriptions
cursor.execute("""
    SELECT DISTINCT ss.station_id, s.name, ss.user_email, ss.threshold_percentage, ss.alert_type
    FROM station_subscriptions ss
    JOIN stations s ON ss.station_id = s.station_id
    WHERE ss.is_active = 1
    LIMIT 1
""")

station = cursor.fetchone()
if station:
    print(f"   Station: {station['name']} ({station['station_id']})")
    print(f"   Subscriber: {station['user_email']}")
    print(f"   Threshold: {station['threshold_percentage']*100}% ({station['alert_type']})")
    
    # Check prediction
    cursor.execute("""
        SELECT predicted_water_level_cm FROM predictions 
        WHERE station_id = ? ORDER BY prediction_date DESC LIMIT 1
    """, (station['station_id'],))
    pred = cursor.fetchone()
    
    # Check min/max
    cursor.execute("SELECT min_level_cm, max_level_cm FROM min_max_values WHERE station_id = ?", (station['station_id'],))
    mm = cursor.fetchone()
    
    if pred and mm:
        prediction = pred['predicted_water_level_cm']
        min_level = mm['min_level_cm']
        max_level = mm['max_level_cm']
        threshold = min_level + (max_level - min_level) * station['threshold_percentage']
        
        print(f"   Prediction: {prediction:.2f} cm")
        print(f"   Min/Max: {min_level:.2f} / {max_level:.2f} cm")
        print(f"   Threshold: {threshold:.2f} cm")
        
        if station['alert_type'] == 'above':
            crossed = prediction >= threshold
        else:
            crossed = prediction <= threshold
        
        print(f"   Threshold crossed: {'✅ YES' if crossed else '❌ NO'}")
    else:
        print("   ⚠️  Missing prediction or min/max values")
else:
    print("   ❌ No active subscriptions found")

conn.close()
PYEOF

echo ""
echo "2. Manually triggering alert check..."
echo "   (This will test if alert checking works)"
echo ""

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from background_scheduler import check_and_send_alerts_for_station

# Test with station 70000355 (the one we modified)
print("   Testing station 70000355...")
try:
    result = check_and_send_alerts_for_station('70000355', '26.82, Århus Å')
    print(f"   ✅ Alert check completed: {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "3. Checking if last_alert_sent_at was updated..."
python3 << 'PYEOF'
import sqlite3
from datetime import datetime

conn = sqlite3.connect('water_levels.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
    SELECT user_email, station_id, alert_type, last_alert_sent_at
    FROM station_subscriptions
    WHERE station_id = '70000355' AND is_active = 1
""")

subs = cursor.fetchall()
if subs:
    print("   Subscription status:")
    for sub in subs:
        last_alert = sub['last_alert_sent_at'] if sub['last_alert_sent_at'] else 'None'
        print(f"     {sub['user_email']} ({sub['alert_type']}): {last_alert}")
        if last_alert != 'None':
            print(f"       ✅ Alert was sent!")
else:
    print("   No subscriptions found")

conn.close()
PYEOF

echo ""
echo "=========================================="
echo "✅ Test complete!"
echo "=========================================="

