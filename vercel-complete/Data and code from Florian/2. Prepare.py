#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config_loader import load_config
from sklearn.preprocessing import StandardScaler, RobustScaler

import os, glob


# In[9]:


CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']


# In[10]:


# Function to reconstruct absolute water levels from a series of changes
def reconstruct_water_levels(changes, starting_level):
    """
    Reconstruct absolute water levels from predicted changes
    
    Args:
        changes: Array of predicted changes in water level
        starting_level: The last known actual water level
        
    Returns:
        Array of absolute water levels
    """
    water_levels = np.zeros(len(changes))
    current_level = starting_level
    
    for i in range(len(changes)):
        # Add the change to the previous level
        current_level = current_level + changes[i]
        water_levels[i] = current_level
        
    return water_levels


# In[11]:


def normalize_daily_features(df):
    """Appropriate normalization for each feature type"""
    
    # Copy to avoid modifying original
    df_norm = df.copy()
    
    # 1. Log transform rainfall (handles zeros and skewness)
    rain_cols = [col for col in df.columns if 'rainfall' in col.lower()]
    for col in rain_cols:
        df_norm[col] = np.log1p(df[col])  # log(1+x)
    
    # 2. RobustScaler for rainfall (after log transform)
    robust_scaler = RobustScaler()
    df_norm[rain_cols] = robust_scaler.fit_transform(df_norm[rain_cols])
    
    # 3. StandardScaler for most features
    standard_cols = ['Pressure', 'Temp', 'Wind_speed', 'Sun', 
                     'water_level', 'Humidity'] + \
                    [col for col in df.columns if 'lag' in col or 'ma' in col]
    
    standard_scaler = StandardScaler()
    df_norm[standard_cols] = standard_scaler.fit_transform(df[standard_cols])
    
    # 4. Keep Season as is (for embedding)
    # Don't normalize cyclical features (sin/cos already -1 to 1)
    
    return df_norm, {'robust': robust_scaler, 'standard': standard_scaler}


# In[12]:


def prepare_file(input_file, output_file):
    df = pd.read_csv(input_file)
    ## Drop rows where WaterHeight has no value
    length_before = len(df)
    df = df.dropna(subset=['water_level'])
    length_after = len(df)

    print(f"Length before droping {length_before}")
    print(f"Length after droping {length_after}")

    df = df.reset_index()
    df = df.rename(columns={"observed" : "date"})
    df = df.drop(columns=['index', 'stationId'])
    df['date'] = pd.to_datetime(df['date'])

    # Create a change-based feature by calculating the difference between consecutive days
    df['water_level_change'] = df['water_level'].diff()
    # Fill the first row's NaN with 0
    df['water_level_change'] = df['water_level_change'].fillna(0)

    starting_level = df['water_level'].iloc[0]
    df['water_level_absolute'] = reconstruct_water_levels(df['water_level_change'].values, starting_level)

    df['date'] = pd.to_datetime(df['date'])
    df['Season'] = (df['date'].dt.month % 12 + 3) // 3  # 1=Winter, 2=Spring, 3=Summer, 4=Fall


    ## GET WATER LEVEL LAGS
    # Previous water levels are the strongest predictors
    df['water_level_lag_1d'] = df['water_level'].shift(1)
    df['water_level_lag_2d'] = df['water_level'].shift(2)
    df['water_level_lag_3d'] = df['water_level'].shift(3)
    df['water_level_lag_7d'] = df['water_level'].shift(7)
    df['water_level_lag_14d'] = df['water_level'].shift(14)


    # Cumulative rainfall is more important than single-day
    df['rainfall_sum_3d'] = df['Rainfall_day'].rolling(3).sum()
    df['rainfall_sum_7d'] = df['Rainfall_day'].rolling(7).sum()
    df['rainfall_sum_14d'] = df['Rainfall_day'].rolling(14).sum()
    df['rainfall_sum_30d'] = df['Rainfall_day'].rolling(30).sum()

    # Antecedent Precipitation Index (exponentially weighted)
    df['API'] = 0.0
    k = 0.9  # decay factor
    for i in range(1, len(df)):
        df.iloc[i, -1] = df.iloc[i-1, -1] * k + df['Rainfall_day'].iloc[i]


    # Pressure changes predict weather 3-4 days ahead
    df['pressure_change_1d'] = df['Pressure'].diff(1)
    df['pressure_change_3d'] = df['Pressure'].diff(3)
    df['pressure_trend_5d'] = df['Pressure'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )


    df['Rainfall_day_4day_lagged'] = df['Rainfall_day'].rolling(4).sum().shift(1)
    df['Temp_4day_lagged'] = df['Temp'].rolling(4).mean().shift(1)

    # Save to File

    df.to_csv(output_file, index=False)
    print(df.columns)
    print(f"file saved to {output_file}")


# In[13]:


def prepare_files(base_path, weather_station_name, water_stations):
    processed_count = 0
    
    for water_station in water_stations:
        # Find CSV files in the folder
        csv_files = glob.glob(os.path.join(base_path, f"Combined{weather_station_name}{water_station}.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {base_path}")
            continue
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {water_station}, processing all of them")
        
        # Process each CSV file
        for combined_file in csv_files:
            output_file_path = os.path.join(base_path, f'preprocessed_{weather_station_name}_{water_station}.csv')
            prepare_file(combined_file, output_file_path)
    


# In[14]:


from config_loader import load_mappings


weather_station_Id = '06170'

water_stations = ['Sengelose', 'Hove', 'Vaerebro', 'Borup', 'Himmelev', 'Hove_2', 'Ledreborg']
base_path = f"{base_path_data}\\1_combined_weather_water_level\\"

stations = ['06104','06069', '06068', '06109', '06126', '06056', '06102', '06074', '06072', '06082'] # Billund, Foulum, Isenvad, Askov, Arslev, Mejrup(Stora), Horsens, Aarhus Syd, Odum, Borris


input_water_levels_base = f"{base_path_data}\\0_1_water_levels_processed\\"
input_weather_base = f"{base_path_data}\\0_2_combined_weather\\"
output_folder_base = f"{base_path_data}\\1_combined_weather_water_level\\"

station_mappings = {
    '06170': ['Borup', 'Himmelev', 'Hove_2', 'Ledreborg'], # Roskilde
    '06104': [], # Billund
    '06069': [], # Foulum
    '06068': ['Mollebaek', 'Aresvad'], # Isenvad
    '06109': [], # Askov
    '06126': [], # Arslev
    # '06056': [], # Mejrup(Stora)
    '06102': ['Bygholm', 'Gudena', 'Gesager'], # Horsens
    '06074': [], # Aarhus Syd
    '06072': [], # Odum
    '06082': ['Skjern', 'Kirkea', 'Hoven', 'Karstoft', 'Vorgod'], # Borris
}

station_mappings = load_mappings()

for weather_station_Id, water_stations in station_mappings.items():
    combined_weather = f"{input_weather_base}processed_weather_{weather_station_Id}.csv"
    prepare_files(base_path, weather_station_Id, water_stations)

