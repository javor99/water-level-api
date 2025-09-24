#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import json
from importlib import reload

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csvhelp
import os, glob
from config_loader import load_config, load_mappings


# In[7]:


def combine_csv_files(combined_weather, vand_file_path, output_file_path, columns_to_keep = None, columns_to_drop = None):
    # Step 2: Merge with the additional file using date as index
    try:

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file_path)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Reading combined weather file {combined_weather}...")
        weather_df = pd.read_csv(combined_weather)
        weather_df['observed']=pd.to_datetime(weather_df['observed'])

        print(f"Reading vand file {vand_file_path}...")
        vand_df = pd.read_csv(vand_file_path)
        vand_df['observed']=pd.to_datetime(vand_df['observed'])
        
        # Check if both dataframes have a 'date' column
        if 'observed' not in weather_df.columns:
            raise ValueError("Original dataframe does not have a 'observed' column")
        if 'observed' not in vand_df.columns:
            raise ValueError("Additional file does not have a 'observed' column")
        
        # Set date as index for both dataframes
        # weather_df.set_index('observed', inplace=True)
        # vand_df.set_index('observed', inplace=True)
        
        # Get min and max
        min_date = vand_df['observed'].min()
        max_date = vand_df['observed'].max()

        print(f"Date range in df2: {min_date} to {max_date}")

        original_len = len(weather_df)
    
        # Filter df1 to only include dates within df2's range
        filtered_df1 = weather_df[(weather_df['observed'] >= min_date) & (weather_df['observed'] <= max_date)]
        
        print(f"Original df1 had {original_len} rows")
        print(f"Filtered df1 has {len(filtered_df1)} rows")
        print(f"Removed {original_len - len(filtered_df1)} rows outside the date range")

        # Merge the dataframes
        final_df = filtered_df1.merge(vand_df, how='outer')
        final_df = final_df.dropna(subset=['stationId'])
        
        if columns_to_drop is not None:
            final_df = final_df.drop(columns=columns_to_drop, errors='ignore')

        if columns_to_keep is not None:
            final_df = final_df[columns_to_keep]
        
        # Save the final result
        print(final_df)
        final_df.to_csv(output_file_path, index = False)
        print(f"Final combined file saved as {output_file_path}")
        
        # Reset index for return value
        final_df.reset_index(inplace=True)
        return final_df
        
    except Exception as e:
        print(f"Error processing additional file: {e}")
        raise


# In[8]:


def process_folders(combined_weather, weather_station_name, water_stations=None, input_base_path=".", output_base_path=".", columns_to_keep = None, columns_to_drop = None):
    # If no folders specified, find all directories in base_path
    if water_stations is None:
        water_stations = [d for d in os.listdir(input_base_path) 
                  if os.path.isdir(os.path.join(input_base_path, d)) 
                  and not d.startswith('.')]  # Exclude hidden folders
        print(f"Auto-detected folders: {water_stations}")
    
    processed_count = 0
    
    for water_station in water_stations:
        folder_path = os.path.join(input_base_path, water_station)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping {water_station} - not a directory")
            continue
        
        # Find CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "water_height_days.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {water_station}")
            continue
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {water_station}, processing all of them")
        
        # Process each CSV file
        for water_station_filename in csv_files:
            # Create output filename
            # output_file_path = os.path.join(output_base_path, water_station, f'Combined{weather_station_name}{water_station}.csv')
            output_file_path = os.path.join(output_base_path, f'Combined{weather_station_name}{water_station}.csv')
            
            print(f"\nProcessing: {water_station_filename}")
            combine_csv_files(combined_weather, water_station_filename, output_file_path, columns_to_keep, columns_to_drop)
            processed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {processed_count}")


# In[9]:


CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']


# In[10]:


stationId = '06170'
# stations = ['06104','06069', '06068', '06109', '06126', '06056', '06102', '06074', '06072', '06082'] # Billund, Foulum, Isenvad, Askov, Arslev, Mejrup(Stora), Horsens, Aarhus Syd, Odum, Borris
stations = ['06104','06069', '06068', '06109', '06126', '06102', '06074', '06072', '06082'] # Billund, Foulum, Isenvad, Askov, Arslev, Horsens, Aarhus Syd, Odum, Borris
# waterStation = 'Main'
# waterStation = 'Hove'
waterStation = 'Vaerebro'

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

# columns_to_keep = ['stationId', 'observed', 'Rainfall_day', 'water_level']
columns_to_drop = ['ks mrk.']

for stationId, water_stations in station_mappings.items():
    combined_weather = f"{input_weather_base}processed_weather_{stationId}.csv"
    process_folders(combined_weather, stationId, water_stations, input_water_levels_base, output_folder_base, columns_to_drop=columns_to_drop)


# In[ ]:




