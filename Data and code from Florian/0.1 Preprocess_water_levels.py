#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os, glob
from config_loader import load_config, load_mappings


# In[53]:


CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']


# # Process original file with weird formating into something more understandable

# In[54]:


def process_file_to_vand(input_file, output_file, input_encoding='latin-1', output_encoding='utf-8'):

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create if there's a directory path
        os.makedirs(output_dir, exist_ok=True)

    # Flag to track if we've found the target word
    found_target = False
    target_word = 'Dato (DK normaltid)'
    
    try:
        # Open the input file with a more forgiving encoding
        with open(input_file, 'r', encoding=input_encoding, errors='replace') as infile:
            with open(output_file, 'w', encoding=output_encoding) as outfile:
                for line in infile:
                    # Check if we've already found the target or if this line contains it
                    if found_target or target_word in line:
                        # Set the flag to True since we've found the target
                        found_target = True
                        
                        # Replace semicolons with commas and write to output file
                        modified_line = line.replace(';', ',')
                        outfile.write(modified_line)
        
        print(f"Processing complete. Output saved to {output_file}")
        if not found_target:
            print(f"Warning: Target word '{target_word}' was not found in the file.")
            
    except Exception as e:
        print(f"An error occurred: {e}")


# In[55]:


def process_folders(folders=None, base_path="."):
    """
    Process CSV files in specified folders or auto-detect folders.
    
    Args:
        folders: List of folder names. If None, will auto-detect.
        base_path: Base directory where folders are located
        output_dir: Directory where processed files will be saved
    """
    # If no folders specified, find all directories in base_path
    if folders is None:
        folders = [d for d in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, d)) 
                  and not d.startswith('.')]  # Exclude hidden folders
        print(f"Auto-detected folders: {folders}")
    
    processed_count = 0
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder} - not a directory")
            continue
        
        # Find CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*Vandstand*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {folder}")
            continue
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {folder}, processing all of them")
        
        # Process each CSV file
        for csv_file in csv_files:
            # Create output filename
            csv_filename = os.path.basename(csv_file)
            output_file = os.path.join(base_path, folder, 'water_height_days.csv')
            
            print(f"\nProcessing: {csv_file}")
            process_file_to_vand(csv_file, output_file)
            processed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {processed_count}")


# In[56]:


def process_mappings(station_mappings, input_folder_base, weather_stations_to_include = None):
    for weather_station, water_level_stations in station_mappings.items():
        if weather_stations_to_include == None:
            process_folders(water_level_stations, input_folder_base)
        elif weather_station in weather_stations_to_include:
            print(f"Processing water stations for weather station [{weather_station}]")
            process_folders(water_level_stations, input_folder_base)


# In[57]:


# station = 'Main'
# station = 'Hove'
# station = 'Vaerebro'
# station = 'Vaerebro'
# water_level_stations = ['Borup', 'Himmelev', 'Hove_2', 'Ledreborg']

weather_stations_to_include = ['06068', '06102', '06082']
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
input_folder_base = f"{base_path_data}\\0_1_water_levels_raw\\"
process_mappings(station_mappings, input_folder_base)


# # Preprocess water

# In[58]:


def visualise_station(df, station_name):
    # Visualize water levels based on change
    plt.figure(figsize=(10, 6))
    plt.plot(df['observed'], df['water_level'], label=f'{station_name}')

    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    plt.title(f'Water Level: {station_name}')
    plt.xlabel('Date')
    plt.ylabel('Water Level (m)')
    plt.legend()
    plt.grid(True)
    # plt.savefig('original_water_levels.png')


# In[59]:


def preprocess_vand(input_file, output_file, date_format):
    directory = os.path.dirname(output_file)
    
    # If the directory path is not empty and doesn't exist, create it
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    print(f"Reading vand file {input_file}...")
    input_df = pd.read_csv(input_file)

    # Rename columns by index
    # Assuming column 0 is the date column and column 1 is the water level column
    column_mapping = {
        input_df.columns[0]: "observed",
        input_df.columns[1]: "water_level"
    }
    input_df = input_df.rename(columns=column_mapping)

    print(input_df.index)
    #date format
    #07-03-2015 00:00

    try:
        input_df['observed'] = pd.to_datetime(input_df['observed'], format=date_format)
        print(f"Successfully parsed dates with format: {date_format}")
    except:
        # If that fails, try the alternative format
        # Define your alternative format here
        alternative_format = '%d-%m-%Y %H:%M'  # Change this to your second format
        try:
            input_df['observed'] = pd.to_datetime(input_df['observed'], format=alternative_format)
            print(f"Successfully parsed dates with alternative format: {alternative_format}")
        except:
            # If both fail, let pandas infer the format
            print("Both formats failed, letting pandas infer the date format...")
            input_df['observed'] = pd.to_datetime(input_df['observed'])

    # Check if we have multiple readings per day
    input_df['date'] = input_df['observed'].dt.date
    readings_per_day = input_df.groupby('date').size()
    
    if readings_per_day.max() > 1:
        print(f"Found multiple readings per day (max: {readings_per_day.max()}). Aggregating to daily averages...")
        # Aggregate to daily data using mean
        input_df = input_df.groupby('date').agg({
            'water_level': 'mean'
        }).reset_index()
        input_df.rename(columns={'date': 'observed'}, inplace=True)
        # Convert date back to datetime
        input_df['observed'] = pd.to_datetime(input_df['observed'])
    else:
        print("Data already appears to be daily.")
        # Drop the temporary date column
        input_df.drop('date', axis=1, inplace=True)

    input_df['observed'] = input_df['observed'].dt.strftime('%Y-%m-%d')

    # Important to fill missing NA values with something
    
    # input_df['water_level'] = input_df['water_level'].interpolate(method='linear')
    # input_df.index = input_df['observed']
    input_df.to_csv(output_file, index=False)
    print(f"Final combined file saved as {output_file}")
    return input_df
    


# In[60]:


def process_folders_for_pre_processing(folders=None, input_base_path=".", output_base_path = '.', date_format = '%d-%m-%Y'):
    """
    Process CSV files in specified folders or auto-detect folders.
    
    Args:
        folders: List of folder names. If None, will auto-detect.
        base_path: Base directory where folders are located
        output_dir: Directory where processed files will be saved
    """
    # If no folders specified, find all directories in base_path
    if folders is None:
        folders = [d for d in os.listdir(input_base_path) 
                  if os.path.isdir(os.path.join(input_base_path, d)) 
                  and not d.startswith('.')]  # Exclude hidden folders
        print(f"Auto-detected folders: {folders}")
    
    processed_count = 0
    
    for folder in folders:
        folder_path = os.path.join(input_base_path, folder)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder} - not a directory")
            continue
        
        # Find CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "water_height_days.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {folder}")
            continue
        
        if len(csv_files) > 1:
            print(f"Warning: Multiple CSV files found in {folder}, processing all of them")
        
        # Process each CSV file
        for csv_file in csv_files:
            # Create output filename
            csv_filename = os.path.basename(csv_file)
            output_file = os.path.join(output_base_path, folder, 'water_height_days.csv')
            
            print(f"\nProcessing: {csv_file}")
            processed_df = preprocess_vand(csv_file, output_file, date_format)
            visualise_station(processed_df, folder)
            processed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Total files processed: {processed_count}")


# In[61]:


def preprocess_mappings(station_mappings, input_folder_base, output_folder_base, date_format,  weather_stations_to_include = None):
    for weather_station, water_level_stations in station_mappings.items():
        if weather_stations_to_include == None:
            print(f"Preprocessing water stations for weather station [{weather_station}]")
            process_folders_for_pre_processing(water_level_stations, input_folder_base, output_folder_base, date_format)
            
        elif weather_station in weather_stations_to_include:
            print(f"Preprocessing water stations for weather station [{weather_station}]")
            process_folders_for_pre_processing(water_level_stations, input_folder_base, output_folder_base, date_format)


# In[62]:


# date_format = '%d-%m-%Y %H:%M'
date_format = '%d-%m-%Y'

output_folder_base = f"{base_path_data}\\0_1_water_levels_intermediate\\"

preprocess_mappings(station_mappings, input_folder_base, output_folder_base, date_format)


# # HANDLE NAs

# In[66]:


from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import Water_levels_NA_handler
reload(Water_levels_NA_handler)


# In[68]:


input_folder_base = f"{base_path_data}/0_1_water_levels_intermediate/"
output_folder_base = f"{base_path_data}/0_1_water_levels_processed/"

CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']


water_stations, station_files_in, station_files_out = [], [], []
for weather_station, water_stations_names in station_mappings.items():
    for water_station_name in water_stations_names:
        if('train' in water_station_name or 'test' in water_station_name):
            continue
        water_stations.append(water_station_name)
        file_path_in = f"{input_folder_base}{water_station_name}/water_height_days.csv"
        station_files_in.append(file_path_in)
        file_path_out = f"{output_folder_base}{water_station_name}/water_height_days.csv"
        station_files_out.append(file_path_out) 
        Water_levels_NA_handler.preprocess_water_station(file_path_in, water_station_name, file_path_out, max_interpolate_days = 10)   


# In[ ]:




