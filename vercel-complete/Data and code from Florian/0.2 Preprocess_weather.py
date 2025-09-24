#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config_loader import load_config

import os, glob


# In[2]:


CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']


# In[3]:


stationId = '06170' # Roskilde
stationId = '06104' # Billund
combined_weather = f"{base_path_data}\\0_2_combined_weather\\combined_weather{stationId}.csv" # Total Weather for a station combined into one CSV file


# In[4]:


def process_stations(base_weather_dir, stations, output_dir):
    for station in stations:
        file_path = f"{base_weather_dir}combined_station_{station}.csv"
        print(f"Processing: {file_path}")
        df = pd.read_csv(file_path)
        df["observed"] = pd.to_datetime(df["observed"])
        print(df.columns)

        # Aggregate by day
        daily_rain = (
        df.groupby(['stationId', df['observed'].dt.date])
        [['precip_past10min', 'sun_last10min_glob']]
        .sum().reset_index())
        daily_rain.columns = ['stationId','observed', 'Rainfall_day', 'Sun']
        print(daily_rain)


        # Mean by day
        daily_vars = (
        df.groupby(['stationId', df['observed'].dt.date])
        [['cloud_cover','humidity_past1h', 'pressure', 'temp_mean_past1h', 'temp_min_past1h', 'temp_soil_mean_past1h', 'wind_speed_past1h']]
        .mean().reset_index())
        daily_vars.columns = ['stationId','observed', 'Cloud_cover', 'Humidity', 'Pressure', "Temp", 'Temp_min_mean', 'Temp_soil_mean', 'Wind_speed']
        print(daily_vars)

        # Max by day
        daily_max = (
        df.groupby(['stationId', df['observed'].dt.date])
        [['temp_max_past12h', 'temp_grass_max_past1h', 'temp_soil_max_past1h']]
        .max().reset_index())
        daily_max.columns = ['stationId','observed', 'Temp_max', 'Temp_grass_max', 'Temp_soil_max']
        print(daily_max)

        daily_vars = daily_vars.drop(columns='stationId')
        daily_max = daily_max.drop(columns='stationId')
        df_combined = pd.merge(daily_rain, daily_vars, on='observed', how = 'left')
        df_combined = pd.merge(df_combined, daily_max, on='observed', how = 'left')
        print(df_combined)

        processed_weather = f"{output_dir}\\processed_weather{stationId}.csv" # Total Weather for a station combined into one CSV file

        df_combined.to_csv(processed_weather, index=False)


# In[10]:


def process_stations_safe(base_weather_dir, stations, output_dir):
    """
    Process weather station data with safe column handling
    """
    # Define expected columns for each aggregation type
    COLUMNS_SUM = ['precip_past10min', 'sun_last10min_glob']
    COLUMNS_MEAN = ['cloud_cover', 'humidity_past1h', 'pressure', 'temp_mean_past1h', 
                    'temp_min_past1h', 'temp_soil_mean_past1h', 'wind_speed_past1h']
    COLUMNS_MAX = ['temp_max_past12h', 'temp_grass_max_past1h', 'temp_soil_max_past1h']
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Track processing status
    processed_stations = []
    skipped_stations = []
    failed_stations = []
    
    for station in stations:
        file_path = f"{base_weather_dir}combined_station_{station}.csv"
        print(f"\n{'='*60}")
        print(f"Processing station {station}")
        print(f"File: {file_path}")
        print(f"{'='*60}")
        
        try:
            df = pd.read_csv(file_path)
            df["observed"] = pd.to_datetime(df["observed"])
            
            print(f"\n📊 DataFrame shape: {df.shape}")
            print(f"📋 Available columns: {', '.join(df.columns.tolist())}")
            
            # Check if precipitation column exists - if not, skip this file
            if 'precip_past10min' not in df.columns:
                print(f"\n❌ SKIPPING station {station}: Missing required precipitation column 'precip_past10min'")
                print(f"   This file will not be processed.")
                skipped_stations.append((station, "Missing precipitation column"))
                continue
            
            # Check which columns are available
            available_sum = [col for col in COLUMNS_SUM if col in df.columns]
            available_mean = [col for col in COLUMNS_MEAN if col in df.columns]
            available_max = [col for col in COLUMNS_MAX if col in df.columns]
            
            # Report missing columns
            missing_sum = [col for col in COLUMNS_SUM if col not in df.columns]
            missing_mean = [col for col in COLUMNS_MEAN if col not in df.columns]
            missing_max = [col for col in COLUMNS_MAX if col not in df.columns]
            
            if missing_sum or missing_mean or missing_max:
                print("\n⚠️  Missing columns:")
                if missing_sum:
                    print(f"  - Sum aggregation: {', '.join(missing_sum)}")
                if missing_mean:
                    print(f"  - Mean aggregation: {', '.join(missing_mean)}")
                if missing_max:
                    print(f"  - Max aggregation: {', '.join(missing_max)}")
            
            # Process available columns
            print("\n✅ Processing available columns:")
            
            # Aggregate by day - SUM
            if available_sum:
                print(f"  - Sum columns: {', '.join(available_sum)}")
                daily_rain = (
                    df.groupby(['stationId', df['observed'].dt.date])
                    [available_sum]
                    .sum().reset_index()
                )
                # Rename columns dynamically
                new_names = ['stationId', 'observed']
                for col in available_sum:
                    if col == 'precip_past10min':
                        new_names.append('Rainfall_day')
                    elif col == 'sun_last10min_glob':
                        new_names.append('Sun')
                    else:
                        new_names.append(f"{col}_sum")  # Generic naming for unexpected columns
                daily_rain.columns = new_names
            else:
                print("  ⚠️  No columns available for sum aggregation")
                # Create empty dataframe with basic columns
                daily_rain = pd.DataFrame({
                    'stationId': df['stationId'].unique()[0] if 'stationId' in df.columns else station,
                    'observed': df.groupby(df['observed'].dt.date).size().index
                })
            
            # Mean by day
            if available_mean:
                print(f"  - Mean columns: {', '.join(available_mean)}")
                daily_vars = (
                    df.groupby(['stationId', df['observed'].dt.date])
                    [available_mean]
                    .mean().reset_index()
                )
                # Rename columns dynamically
                rename_map = {
                    'cloud_cover': 'Cloud_cover',
                    'humidity_past1h': 'Humidity',
                    'pressure': 'Pressure',
                    'temp_mean_past1h': 'Temp',
                    'temp_min_past1h': 'Temp_min_mean',
                    'temp_soil_mean_past1h': 'Temp_soil_mean',
                    'wind_speed_past1h': 'Wind_speed'
                }
                daily_vars.columns = ['stationId', 'observed'] + [
                    rename_map.get(col, f"{col}_mean") for col in available_mean
                ]
            else:
                print("  ⚠️  No columns available for mean aggregation")
                daily_vars = None
            
            # Max by day
            if available_max:
                print(f"  - Max columns: {', '.join(available_max)}")
                daily_max = (
                    df.groupby(['stationId', df['observed'].dt.date])
                    [available_max]
                    .max().reset_index()
                )
                # Rename columns dynamically
                rename_map = {
                    'temp_max_past12h': 'Temp_max',
                    'temp_grass_max_past1h': 'Temp_grass_max',
                    'temp_soil_max_past1h': 'Temp_soil_max'
                }
                daily_max.columns = ['stationId', 'observed'] + [
                    rename_map.get(col, f"{col}_max") for col in available_max
                ]
            else:
                print("  ⚠️  No columns available for max aggregation")
                daily_max = None
            
            # Combine dataframes
            print("\n🔄 Combining aggregated data...")
            df_combined = daily_rain.copy()
            
            if daily_vars is not None:
                daily_vars = daily_vars.drop(columns='stationId')
                df_combined = pd.merge(df_combined, daily_vars, on='observed', how='left')
            
            if daily_max is not None:
                daily_max = daily_max.drop(columns='stationId')
                df_combined = pd.merge(df_combined, daily_max, on='observed', how='left')
            
            print(f"\n📊 Combined dataframe shape: {df_combined.shape}")
            print(f"📋 Final columns: {', '.join(df_combined.columns.tolist())}")
            
            # Save processed data
            processed_weather = os.path.join(output_dir, f"processed_weather_{station}.csv")
            df_combined.to_csv(processed_weather, index=False)
            print(f"\n✅ Saved processed data to: {processed_weather}")
            
            # Display first few rows
            print("\n📄 First 5 rows of processed data:")
            print(df_combined.head())
            
            processed_stations.append(station)
            
        except FileNotFoundError:
            print(f"\n❌ Error: File not found - {file_path}")
            failed_stations.append((station, "File not found"))
            continue
        except Exception as e:
            print(f"\n❌ Error processing station {station}: {str(e)}")
            failed_stations.append((station, str(e)))
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully processed: {len(processed_stations)} stations")
    if processed_stations:
        for station in processed_stations:
            print(f"   - Station {station}")
    
    if skipped_stations:
        print(f"\n⚠️  Skipped: {len(skipped_stations)} stations")
        for station, reason in skipped_stations:
            print(f"   - Station {station}: {reason}")
    
    if failed_stations:
        print(f"\n❌ Failed: {len(failed_stations)} stations")
        for station, reason in failed_stations:
            print(f"   - Station {station}: {reason}")
    
    print(f"\n📁 Output directory: {output_dir}")
    print("="*60)


def check_columns_availability(base_weather_dir, stations):
    """
    Utility function to check which columns are available across all stations
    """
    print("\n" + "="*60)
    print("CHECKING COLUMN AVAILABILITY ACROSS STATIONS")
    print("="*60)
    
    all_columns = set()
    station_columns = {}
    stations_with_precip = []
    stations_without_precip = []
    
    for station in stations:
        file_path = f"{base_weather_dir}combined_station_{station}.csv"
        try:
            df = pd.read_csv(file_path, nrows=1)  # Read only header
            columns = set(df.columns)
            station_columns[station] = columns
            all_columns.update(columns)
            
            # Check for precipitation column
            if 'precip_past10min' in columns:
                stations_with_precip.append(station)
            else:
                stations_without_precip.append(station)
                
        except Exception as e:
            print(f"⚠️  Could not read station {station}: {e}")
            station_columns[station] = set()
    
    # Report precipitation column status
    print(f"\n🌧️  Precipitation column ('precip_past10min') status:")
    print(f"✅ Stations WITH precipitation: {len(stations_with_precip)}")
    for station in stations_with_precip:
        print(f"   - Station {station}")
    
    if stations_without_precip:
        print(f"\n❌ Stations WITHOUT precipitation (will be skipped): {len(stations_without_precip)}")
        for station in stations_without_precip:
            print(f"   - Station {station}")
    
    # Find common columns
    if station_columns:
        common_columns = set.intersection(*station_columns.values())
        print(f"\n✅ Common columns across all stations ({len(common_columns)}):")
        for col in sorted(common_columns):
            print(f"  - {col}")
    
    # Find unique columns per station
    print("\n📊 Unique columns by station:")
    for station, columns in station_columns.items():
        unique = columns - common_columns if station_columns else columns
        if unique:
            print(f"\n  Station {station} unique columns:")
            for col in sorted(unique):
                print(f"    - {col}")
    
    return station_columns, common_columns, stations_with_precip, stations_without_precip


# In[11]:


base_weather_dir = f"{base_path_data}\\WeatherDataAllCSVMultiProcess_MultiStation\\"
stations = ['06104','06069', '06068', '06109', '06126', '06056', '06102', '06074', '06072', '06082']
output_dir = f"{base_path_data}\\0_2_combined_weather\\"

station_cols, common_cols, has_precip, no_precip = check_columns_availability(base_weather_dir, stations)


# In[12]:


process_stations_safe(base_weather_dir, stations, output_dir)


# In[13]:


def print_station_data(df, cols = None):
    if cols != None:
        for col in cols:
            plt.figure(figsize=(10, 4))
            plt.plot(df['observed'], df[col], label=col)
            # Format the x-axis to show dates nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
            plt.title(f"{df['stationId'][0]}{col}")
            plt.xlabel('Date')
            plt.ylabel('Unit')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        for col in df.columns:
            if (col == 'stationId' or col == 'observed'):
                continue

            plt.figure(figsize=(10, 4))
            plt.plot(df['observed'], df[col], label=col)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
            plt.title(f"{df['stationId'][0]}{col}")
            plt.xlabel('Date')
            plt.ylabel('Unit')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


# In[14]:


def print_all_stations(base_weather_dir, stations):
    for station in stations:
        df = pd.read_csv(f"{base_weather_dir}processed_weather_{station}.csv")
        print_station_data(df)


# In[15]:


output_dir = f"{base_path_data}\\0_2_combined_weather\\"
print_all_stations(output_dir, stations)


# In[ ]:




