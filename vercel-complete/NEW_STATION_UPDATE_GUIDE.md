# New Station Data Update Guide

## Overview
The `update_new_station_data.py` script automatically updates all necessary data for a newly added water level station, including:
- 30-day historical water level data
- Min/max water level values (historical and 30-day)
- 7-day water level predictions

## Usage

### Basic Usage
```bash
python3 update_new_station_data.py <station_id>
```

### Examples
```bash
# Update data for station 70000864
python3 update_new_station_data.py 70000864

# Update data for station 70000865 (skip predictions for faster testing)
python3 update_new_station_data.py 70000865 --skip-predictions
```

## What the Script Does

### Step 1: 30-Day Historical Data
- Fetches the last 30 days of water level data from Vandah API
- Stores daily averages in the `last_30_days_historical` table
- Replaces any existing data for the station

### Step 2: Min/Max Calculation
- Fetches 5 years of historical water level data
- Calculates overall min/max values
- Calculates 30-day min/max values
- Updates the `stations` table with these values

### Step 3: Predictions
- Runs the prediction model using `predict_unseen_station.py`
- Tries 60 days of data first, then 90 days if insufficient
- Saves 7-day predictions to the `predictions` table
- Uses the "nearest" strategy for unseen stations

## Prerequisites

1. **Station must exist in database** - The station must already be added to the `stations` table
2. **Required dependencies** - pandas, requests, torch, numpy
3. **Prediction model** - The transformer model must be available in `models/` directory
4. **Internet connection** - For fetching data from Vandah API

## Output

The script provides detailed progress information:
```
🚀 UPDATING NEW STATION DATA
============================================================
Station ID: 70000864
Started at: 2024-01-15 14:30:00

Station: Borup station
Location: 55.483, 11.967

STEP 1: Updating 30-day historical data
----------------------------------------
📅 Updating 30-day history for 70000864 (Borup station)
  ✅ Updated 30 records in 30-day history

STEP 2: Calculating min/max values
----------------------------------------
📊 Calculating min/max values for 70000864 (Borup station)
  📊 Fetching 5 years of historical data...
  ✅ Updated min/max values:
    Historical: 25.50 - 180.20 cm
    30-day: 45.30 - 120.80 cm

STEP 3: Generating predictions
----------------------------------------
🔮 Running predictions for 70000864 (Borup station)
  📅 Trying 60 days of historical data...
  ✅ Predictions generated successfully with 60 days of data
  💾 Saved 7 predictions to database

============================================================
📊 SUMMARY
Completed at: 2024-01-15 14:35:00

Results:
  ✅ 30-day history: SUCCESS
  ✅ Min/Max calculation: SUCCESS
  ✅ Predictions: SUCCESS

Final Station Data:
  Station: Borup station
  Historical min/max: 25.50 - 180.20 cm
  30-day min/max: 45.30 - 120.80 cm
  Historical records: 30
  Prediction records: 7

🎯 Overall Success: 3/3 operations completed
🎉 All operations completed successfully!
```

## Error Handling

The script handles various error conditions:
- **No water data available** - Warns but continues with other operations
- **Insufficient data for predictions** - Tries different time windows
- **API timeouts** - Provides clear error messages
- **Database errors** - Reports specific issues

## Integration with Station CRUD

This script is designed to be used after adding a new station via the API:

```bash
# 1. Add new station via API
curl -X POST "http://localhost:5001/stations" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "70000999",
    "name": "New Test Station",
    "latitude": 55.500,
    "longitude": 12.000,
    "location_type": "stream",
    "station_owner": "Test Organization",
    "municipality_id": 1
  }'

# 2. Update all data for the new station
python3 update_new_station_data.py 70000999
```

## Troubleshooting

### Common Issues

1. **Station not found**
   - Ensure the station exists in the database
   - Check the station_id is correct

2. **No water data available**
   - The station might not have data in Vandah API
   - Check if the station_id is valid in the Vandah system

3. **Prediction failures**
   - Insufficient historical data (less than 60 days)
   - Model file not found
   - Dependencies not installed

4. **Database errors**
   - Check database permissions
   - Ensure tables exist (run municipality_db_update.py if needed)

### Debug Mode
For more detailed output, you can modify the script to add debug logging or run individual functions separately.

## Performance Notes

- **30-day history**: Usually takes 5-10 seconds
- **Min/max calculation**: Takes 5-10 seconds (single request for 5 years of data)
- **Predictions**: Can take 1-3 minutes depending on data availability
- **Total time: Typically 1-3 minutes per station (much faster with single request)

Use `--skip-predictions` flag for faster testing when you only need historical data and min/max values.
