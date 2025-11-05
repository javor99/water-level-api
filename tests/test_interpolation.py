#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to demonstrate the missing data interpolation functionality.
This shows how the system handles missing days in water level data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import the interpolation function
sys.path.insert(0, os.path.dirname(__file__))
from predict_unseen_station import check_and_interpolate_missing_days


def create_test_data_with_gaps(days=50, missing_indices=None):
    """
    Create test water level data with specified missing days.
    
    Args:
        days: Total number of days
        missing_indices: List of day indices to remove (0-based)
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    
    # Create complete date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate water level data (with some variation)
    np.random.seed(42)
    base_level = 150.0  # cm
    variation = np.sin(np.linspace(0, 4*np.pi, days)) * 20  # Seasonal variation
    noise = np.random.normal(0, 5, days)  # Random noise
    water_levels = base_level + variation + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates.date,
        'water_level_cm': water_levels
    })
    
    # Remove specified days to create gaps
    if missing_indices:
        missing_indices = [i for i in missing_indices if 0 <= i < days]
        df = df.drop(missing_indices).reset_index(drop=True)
    
    return df


def test_case(name, days=50, missing_indices=None, lookback_days=40, max_missing=3):
    """Run a single test case."""
    print(f"\n{'='*70}")
    print(f"TEST CASE: {name}")
    print(f"{'='*70}")
    
    # Create test data
    df = create_test_data_with_gaps(days=days, missing_indices=missing_indices)
    
    print(f"\nOriginal data:")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    if missing_indices:
        print(f"  Missing day indices (from end): {missing_indices}")
    
    try:
        # Apply interpolation
        result_df = check_and_interpolate_missing_days(
            df, 
            max_missing=max_missing, 
            lookback_days=lookback_days
        )
        
        print(f"\nResult:")
        print(f"  Total rows after interpolation: {len(result_df)}")
        print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        
        # Show a sample of the data
        print(f"\nLast 10 rows of interpolated data:")
        print(result_df.tail(10).to_string(index=False))
        
        return True
        
    except RuntimeError as e:
        print(f"\n❌ ERROR: {e}")
        return False


def main():
    """Run all test cases."""
    print("🧪 TESTING MISSING DATA INTERPOLATION FUNCTIONALITY")
    print("="*70)
    
    # Test Case 1: No missing data
    test_case(
        name="No Missing Data (Perfect case)",
        days=50,
        missing_indices=None,
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 2: 1 missing day (should interpolate)
    test_case(
        name="1 Missing Day (Should interpolate)",
        days=50,
        missing_indices=[45],  # Missing 5 days from the end
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 3: 3 missing days (at the limit, should interpolate)
    test_case(
        name="3 Missing Days (At limit - should interpolate)",
        days=50,
        missing_indices=[48, 45, 42],  # Missing 3 days
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 4: 4 missing days (exceeds limit, should fail)
    test_case(
        name="4 Missing Days (Exceeds limit - should fail)",
        days=50,
        missing_indices=[49, 47, 44, 41],  # Missing 4 days
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 5: Missing days scattered throughout lookback period
    test_case(
        name="Scattered Missing Days (2 days - should interpolate)",
        days=50,
        missing_indices=[49, 35],  # Missing days at different positions
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 6: Missing consecutive days
    test_case(
        name="Consecutive Missing Days (2 days - should interpolate)",
        days=50,
        missing_indices=[48, 47],  # Missing 2 consecutive days
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 7: Missing days at the boundary of lookback period
    test_case(
        name="Missing Days at Boundary (1 day - should interpolate)",
        days=50,
        missing_indices=[10],  # Right at the edge of 40-day lookback
        lookback_days=40,
        max_missing=3
    )
    
    # Test Case 8: Custom parameters - strict limit
    test_case(
        name="Custom Parameters - Max 1 Missing Day",
        days=50,
        missing_indices=[48, 45],  # 2 missing days
        lookback_days=40,
        max_missing=1  # Only allow 1 missing day
    )
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS COMPLETED")
    print(f"{'='*70}")
    
    # Summary
    print("\n📋 SUMMARY:")
    print("The interpolation function:")
    print("  ✓ Checks the last N days (default: 40 days) for missing data")
    print("  ✓ Counts missing days in that period")
    print("  ✓ If missing days ≤ threshold (default: 3), performs linear interpolation")
    print("  ✓ If missing days > threshold, raises an error")
    print("  ✓ Uses pandas linear interpolation for smooth estimates")
    print("\nThis ensures predictions are only generated when data quality is sufficient.")


if __name__ == "__main__":
    main()

