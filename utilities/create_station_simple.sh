#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.
# Simple script to create a new water level station with automatic data update

if [ $# -lt 2 ]; then
    echo "Usage: $0 <station_id> <station_name> [municipality_id] [latitude] [longitude]"
    echo "Example: $0 70001001 'New Station' 1 55.5 12.0"
    exit 1
fi

STATION_ID=$1
STATION_NAME=$2
MUNICIPALITY_ID=${3:-1}  # Default to municipality 1
LATITUDE=${4:-55.5}      # Default coordinates
LONGITUDE=${5:-12.0}

echo "🚀 Creating new station: $STATION_NAME (ID: $STATION_ID)"
echo "📍 Municipality: $MUNICIPALITY_ID"
echo "🌍 Coordinates: $LATITUDE, $LONGITUDE"
echo ""

# Step 1: Create the station
echo "Step 1: Creating station via API..."
RESPONSE=$(curl -s -X POST "http://localhost:5001/stations" \
  -H "Content-Type: application/json" \
  -d "{
    \"station_id\": \"$STATION_ID\",
    \"name\": \"$STATION_NAME\",
    \"latitude\": $LATITUDE,
    \"longitude\": $LONGITUDE,
    \"municipality_id\": $MUNICIPALITY_ID
  }")

echo "$RESPONSE" | jq '.'

# Check if creation was successful
if echo "$RESPONSE" | jq -e '.message' > /dev/null; then
    echo ""
    echo "✅ Station created successfully!"
    echo ""
    echo "Step 2: Updating station data (30-day history, min/max, predictions)..."
    echo "This may take 1-3 minutes..."
    
    # Step 2: Update station data
    python3 update_new_station_data.py $STATION_ID
    
    echo ""
    echo "Step 3: Verifying station data..."
    
    # Step 3: Verify the data
    echo "📊 Station info:"
    curl -s "http://localhost:5001/stations/$STATION_ID" | jq '.station | {station_id, name, municipality_name, latitude, longitude}'
    
    echo ""
    echo "📈 Water level history (latest 3 days):"
    curl -s "http://localhost:5001/water-levels/$STATION_ID" | jq '.history[:3]'
    
    echo ""
    echo "🔮 Predictions (next 3 days):"
    curl -s "http://localhost:5001/predictions/$STATION_ID" | jq '.predictions[:3]'
    
    echo ""
    echo "🎉 Station setup completed successfully!"
    echo "You can now access:"
    echo "  - Station info: http://localhost:5001/stations/$STATION_ID"
    echo "  - Water levels: http://localhost:5001/water-levels/$STATION_ID"
    echo "  - Predictions: http://localhost:5001/predictions/$STATION_ID"
    
else
    echo "❌ Failed to create station:"
    echo "$RESPONSE" | jq '.error'
    exit 1
fi
