#!/usr/bin/env python3
import pandas as pd
import sqlite3
from datetime import datetime

def save_predictions_to_db_fixed(station_id: str):
    """Save predictions from CSV file to database with correct column mapping."""
    csv_path = f'predictions/predictions_{station_id}_unseen.csv'
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  Prediction CSV not found: {csv_path}")
        return False
    
    try:
        # Read the predictions CSV
        df = pd.read_csv(csv_path)
        print(f"  📊 CSV columns: {list(df.columns)}")
        print(f"  📊 CSV data shape: {df.shape}")
        
        conn = sqlite3.connect('water_levels.db')
        cursor = conn.cursor()
        
        # Delete existing predictions for this station
        cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
        
        # Insert new predictions with correct column mapping
        records_inserted = 0
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO predictions
                (station_id, prediction_date, predicted_water_level_cm, predicted_water_level_m,
                 change_from_last_cm, forecast_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                station_id,
                row['date'],  # CSV has 'date' column
                row['predicted_water_level_cm'],
                row['predicted_water_level_m'],
                row['change_from_last_daily_mean_cm'],  # CSV has this column name
                row['date'],  # Use same date for forecast_date
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            records_inserted += 1
        
        conn.commit()
        conn.close()
        
        print(f"  💾 Saved {records_inserted} predictions to database")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving predictions to database: {e}")
        return False

# Test the fix
print("🔧 Fixing predictions database insertion for station 70000600...")
result = save_predictions_to_db_fixed("70000600")
if result:
    print("✅ Predictions successfully saved to database!")
else:
    print("❌ Failed to save predictions")
