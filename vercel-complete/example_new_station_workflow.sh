#!/bin/bash

# Example workflow for adding a new water level station and updating its data

echo "🚀 NEW STATION WORKFLOW EXAMPLE"
echo "================================"

# Step 1: Add new station via API
echo "Step 1: Adding new station via API..."
curl -X POST "http://localhost:5001/stations" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "70000999",
    "name": "Example New Station",
    "latitude": 55.500,
    "longitude": 12.000,
    "location_type": "stream",
    "station_owner": "Example Organization",
    "municipality_id": 1
  }'

echo -e "\n"

# Step 2: Update all data for the new station
echo "Step 2: Updating station data (30-day history, min/max, predictions)..."
python3 update_new_station_data.py 70000999

echo -e "\n"

# Step 3: Verify the station data
echo "Step 3: Verifying station data..."
curl -X GET "http://localhost:5001/stations/70000999" 2>/dev/null | jq '.'

echo -e "\n"

# Step 4: Check predictions
echo "Step 4: Checking predictions..."
curl -X GET "http://localhost:5001/predictions/70000999" 2>/dev/null | jq '.'

echo -e "\n"
echo "✅ Workflow completed!"
