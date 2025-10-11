# Missing Data Interpolation Guide

## Overview

The prediction system now automatically handles missing water level data using **linear interpolation** when generating predictions. This ensures more reliable predictions even when there are small gaps in historical data.

## How It Works

### Automatic Missing Data Detection

When generating predictions, the system:

1. **Checks the last 40 days** of water level data for missing dates
2. **Counts the number of missing days** in that period
3. **Applies one of two actions:**
   - If missing days ≤ 3: **Interpolates** the missing values
   - If missing days > 3: **Raises an error** (insufficient data quality)

### Interpolation Method

The system uses **linear interpolation** which:
- Fills missing values by drawing a straight line between known data points
- Provides smooth, realistic estimates for short gaps
- Preserves the general trend of water level changes

### Configuration Parameters

The interpolation behavior can be customized with these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_missing` | 3 | Maximum number of missing days allowed for interpolation |
| `lookback_days` | 40 | Number of recent days to check for missing data |

## Code Integration

### In `predict_unseen_station.py`

The interpolation is automatically applied when fetching water data:

```python
def fetch_water_daily(vandah_station_id: str, past_days: int) -> pd.DataFrame:
    """Fetch water level data and interpolate missing days if needed."""
    # ... fetch data from API ...
    
    # Check and interpolate missing days (max 3 missing days in last 40 days)
    daily = check_and_interpolate_missing_days(daily, max_missing=3, lookback_days=40)
    
    return daily
```

### Function Signature

```python
def check_and_interpolate_missing_days(
    daily_df: pd.DataFrame, 
    max_missing: int = 3, 
    lookback_days: int = 40
) -> pd.DataFrame:
    """
    Check if there are missing days in the last lookback_days.
    If missing days <= max_missing, interpolate them. Otherwise, raise an error.
    
    Args:
        daily_df: DataFrame with 'date' and 'water_level_cm' columns
        max_missing: Maximum number of missing days allowed for interpolation (default: 3)
        lookback_days: Number of days to look back for checking missing data (default: 40)
    
    Returns:
        DataFrame with interpolated values if applicable
        
    Raises:
        RuntimeError: If missing days exceed max_missing threshold
    """
```

## Example Scenarios

### ✅ Scenario 1: No Missing Data
```
Input:  40 consecutive days of data
Action: No interpolation needed
Result: ✅ Predictions generated successfully
```

### ✅ Scenario 2: 1 Missing Day
```
Input:  39 days of data, 1 day missing within last 40 days
Action: Linear interpolation fills the 1 missing day
Result: ✅ Predictions generated successfully
Output: "🔧 Interpolating 1 missing days using linear interpolation..."
```

### ✅ Scenario 3: 3 Missing Days (At Threshold)
```
Input:  37 days of data, 3 days missing within last 40 days
Action: Linear interpolation fills all 3 missing days
Result: ✅ Predictions generated successfully
Output: "🔧 Interpolating 3 missing days using linear interpolation..."
```

### ❌ Scenario 4: 4+ Missing Days (Exceeds Threshold)
```
Input:  36 days of data, 4 days missing within last 40 days
Action: Error raised - too many gaps for reliable interpolation
Result: ❌ Prediction generation fails
Error:  "Too many missing days in the last 40 days: 4 missing days found.
         Maximum allowed for interpolation: 3 days."
```

## Console Output Examples

### Successful Interpolation
```
📊 Found 2 missing days in the last 40 days: [datetime.date(2024, 01, 15), datetime.date(2024, 01, 20)]
🔧 Interpolating 2 missing days using linear interpolation...
✅ Successfully interpolated 2 missing days
```

### No Missing Data
```
✅ No missing days in the last 40 days
```

### Too Many Missing Days
```
📊 Found 5 missing days in the last 40 days: [datetime.date(2024, 01, 15), datetime.date(2024, 01, 17), ...]
❌ Too many missing days in the last 40 days: 5 missing days found.
   Maximum allowed for interpolation: 3 days.
   Missing dates: [datetime.date(2024, 01, 15), datetime.date(2024, 01, 17), ...]
   Cannot generate reliable predictions with this much missing data.
```

## Testing

Run the test script to see the interpolation in action:

```bash
python3 test_interpolation.py
```

This script demonstrates:
- ✅ No missing data (baseline)
- ✅ 1 missing day (interpolates successfully)
- ✅ 3 missing days (at limit, interpolates)
- ❌ 4 missing days (exceeds limit, fails)
- ✅ Scattered missing days (interpolates)
- ✅ Consecutive missing days (interpolates)

## Benefits

1. **Improved Reliability**: Predictions can be generated even with minor data gaps
2. **Data Quality Control**: Automatically rejects predictions when data quality is too poor
3. **Transparency**: Clear logging shows exactly what interpolation occurred
4. **Configurable**: Thresholds can be adjusted based on requirements

## Why 3 Days?

The default threshold of 3 missing days (7.5% of 40 days) balances:
- **Reliability**: Short gaps can be interpolated without significant error
- **Safety**: Longer gaps would introduce too much uncertainty
- **Practicality**: Handles common data collection issues (weekend gaps, sensor maintenance)

## Customization

To change the interpolation parameters, modify the call in `predict_unseen_station.py`:

```python
# More strict: allow only 1 missing day
daily = check_and_interpolate_missing_days(daily, max_missing=1, lookback_days=40)

# More lenient: allow 5 missing days
daily = check_and_interpolate_missing_days(daily, max_missing=5, lookback_days=40)

# Shorter lookback period (last 30 days)
daily = check_and_interpolate_missing_days(daily, max_missing=3, lookback_days=30)
```

## Integration with Existing System

The interpolation is **automatically applied** in:

1. **`predict_unseen_station.py`**: When generating predictions for a single station
2. **`run_predictions_and_update_db.py`**: When generating predictions for all stations
3. **`background_scheduler.py`**: During scheduled automatic updates

No changes needed to existing workflows - interpolation happens transparently!

## Technical Details

### Algorithm Steps

1. **Convert dates** to datetime format
2. **Sort by date** to ensure chronological order
3. **Extract lookback period** (last N days)
4. **Generate complete date range** for the lookback period
5. **Identify missing dates** by comparing actual vs. expected dates
6. **Count missing days** and check against threshold
7. **If acceptable**: Create complete date range and merge with actual data
8. **Apply pandas interpolate()** with method='linear' and limit_direction='both'
9. **Combine** interpolated lookback period with earlier data
10. **Return** complete DataFrame with no gaps

### Interpolation Formula

For a missing day between day `i-1` and day `i+1`:

```
value[i] = value[i-1] + (value[i+1] - value[i-1]) × 0.5
```

For multiple consecutive missing days, this becomes:

```
value[i] = value[start] + (value[end] - value[start]) × (i - start) / (end - start)
```

Where `start` and `end` are the indices of known data points surrounding the gap.

## Best Practices

1. **Monitor interpolation frequency**: Regular interpolation may indicate data collection issues
2. **Review interpolated predictions**: More critical for high-stakes decisions
3. **Maintain data quality**: Interpolation is a fallback, not a primary solution
4. **Adjust thresholds carefully**: Consider your specific use case and accuracy requirements

## Future Enhancements

Potential improvements to consider:

- **Spline interpolation**: More sophisticated curve fitting for longer gaps
- **Pattern-based interpolation**: Use historical patterns (e.g., seasonal trends)
- **Quality metrics**: Report confidence scores for interpolated predictions
- **Selective interpolation**: Different thresholds for different stations based on data reliability
- **Gap analysis**: Detailed reporting on missing data patterns

## Troubleshooting

### Problem: Predictions fail with "Too many missing days"
**Solution**: 
- Check data source for outages
- Extend data collection period
- Consider manual data entry for critical gaps
- Temporarily increase `max_missing` threshold (with caution)

### Problem: Interpolated predictions seem inaccurate
**Solution**:
- Verify source data quality
- Reduce `max_missing` to stricter threshold
- Check for systematic issues (e.g., sensor drift)
- Consider alternative interpolation methods

### Problem: No interpolation occurring when expected
**Solution**:
- Verify missing data is within the `lookback_days` window
- Check console output for diagnostic messages
- Ensure data format is correct (date column as date objects)

## Support

For issues or questions about the interpolation feature:
1. Check console output for detailed error messages
2. Run `test_interpolation.py` to verify functionality
3. Review this guide for configuration options
4. Examine `predict_unseen_station.py` for implementation details

