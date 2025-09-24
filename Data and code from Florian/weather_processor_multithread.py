import os
import multiprocessing as mp
from functools import partial
import json
import pandas as pd
import time
import datetime
from config_loader import load_config



#Station name: Roskilde Airport
# stationId = '06170'

# Most data
# Stationsnavn: Sjælsmark
# stationId =  '06188'
stationId = '06104' # Billund
stations = ['06104','06069', '06068', '06109', '06126', '06056', '06102', '06074', '06072', '06082'] # Billund, Foulum, Isenvad, Askov, Arslev, Mejrup(Stora), Horsens, Aarhus Syd, Odum, Borris

CONFIG = load_config()
print(CONFIG['base_path_data'])
base_path_data = CONFIG['base_path_data']

def safe_to_datetime(date_series):
    """
    Convert ISO format date strings to datetime objects with error handling.
    Handles both formats:
    - With microseconds: "2023-07-08T02:07:59.229922Z"
    - Without microseconds: "2023-12-17T15:09:32Z"
    
    Parameters:
    - date_series: pandas Series containing date strings
    
    Returns:
    - pandas Series with datetime objects
    """
    try:
        # Let pandas infer the format automatically
        return pd.to_datetime(date_series, errors='coerce')
    except Exception as e:
        print(f"Error with automatic format detection: {e}")
        
        # Try explicit formats as fallback
        try:
            # Try custom converter to handle multiple formats
            def parse_datetime(dt_str):
                try:
                    if '.' in dt_str:  # Has microseconds
                        return pd.to_datetime(dt_str, format='%Y-%m-%dT%H:%M:%S.%f%z')
                    else:  # No microseconds
                        return pd.to_datetime(dt_str, format='%Y-%m-%dT%H:%M:%S%z')
                except:
                    return pd.NaT
                    
            return date_series.apply(parse_datetime)
        except Exception as e:
            print(f"All conversion attempts failed: {e}")
            return date_series
        

def log_execution_time(timing_log_file, method_name, file_count,elapsed_time_seconds, elapsed_time_minutes, station, cores):
    """
    Log execution time to a text file that can be read and appended to later
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the file with headers if it doesn't exist
    if not os.path.exists(timing_log_file):
        with open(timing_log_file, 'w') as f:
            f.write("Timestamp,Method,Files_Processed,Execution_Time_Seconds,Execution_Time_Minutes,Station,Cores\n")
    
    # Append the timing information
    with open(timing_log_file, 'a') as f:
        f.write(f"{timestamp},{method_name},{file_count},{elapsed_time_seconds:.4f},{elapsed_time_minutes:.4f},{station},{cores}\n")
    
    print(f"Timing information logged to {timing_log_file}")



def process_content(file, stations_list):
    """
    Process file content and return data frames grouped by station
    """
    geojson_data = []
    for line in file:
        geojson_data.append(json.loads(line))
    
    records = []
    for feature in geojson_data:
        record = {
            'id': feature['id'],
            'created': feature['properties']['created'],
            'observed': feature['properties']['observed'],
            'parameterId': feature['properties']['parameterId'],
            'stationId': feature['properties']['stationId'],
            'value': feature['properties']['value']
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    df['created'] = safe_to_datetime(df['created'])
    df['observed'] = safe_to_datetime(df['observed'])

    # Filter for all stations in the list
    filtered_df = df[df['stationId'].isin(stations_list)]
    
    # Group by stationId and create pivoted dataframes
    station_dfs = {}
    for station_id in stations_list:
        station_df = filtered_df[filtered_df['stationId'] == station_id]
        
        if not station_df.empty:
            pivoted_df = station_df.pivot_table(
                index=['stationId', 'observed'],
                columns='parameterId',
                values='value',
                aggfunc='first'
            )
            pivoted_df = pivoted_df.reset_index()
            station_dfs[station_id] = pivoted_df
    
    return station_dfs


def process_file(file_path, output_dir, stations_list):
    """
    Process a single file and save results for each station to separate files.
    """
    try:
        process_id = mp.current_process().name
        print(f"Process {process_id} starting to process: {file_path}")
        
        # Read the file
        with open(file_path, 'r') as file:
            # Process the content for all stations
            station_dfs = process_content(file, stations_list)
        
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        
        # Save each station's data to a separate file
        saved_files = []
        for station_id, df in station_dfs.items():
            station_output_dir = os.path.join(output_dir, f"station_{station_id}")
            if not os.path.exists(station_output_dir):
                os.makedirs(station_output_dir)
            
            output_path = os.path.join(station_output_dir, f"{filename_no_ext}_{station_id}.csv")
            df.to_csv(output_path, index=False)
            saved_files.append((station_id, output_path))
            
        return file_path, True, saved_files  # Success with list of saved files
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, False, []  # Failure
    

def combine_station_files(output_dir, stations_list):
    """
    Combine all files for each station into a single CSV file
    """
    print("\nCombining files by station...")
    
    for station_id in stations_list:
        station_dir = os.path.join(output_dir, f"station_{station_id}")
        
        if not os.path.exists(station_dir):
            print(f"No data found for station {station_id}")
            continue
            
        # Get all CSV files for this station
        csv_files = [f for f in os.listdir(station_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"No CSV files found for station {station_id}")
            continue
        
        print(f"Combining {len(csv_files)} files for station {station_id}...")
        
        # Read all CSV files and combine
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(station_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if dfs:
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by observed time
            if 'observed' in combined_df.columns:
                combined_df['observed'] = pd.to_datetime(combined_df['observed'])
                combined_df = combined_df.sort_values('observed')
            
            # Save combined file
            combined_path = os.path.join(output_dir, f"combined_station_{station_id}.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"Saved combined data for station {station_id} to {combined_path}")
            
            # Optionally, clean up individual files
            # for csv_file in csv_files:
            #     os.remove(os.path.join(station_dir, csv_file))
            # os.rmdir(station_dir)




def main():
    # Configuration
    input_dir = f"{base_path_data}\\WeatherDataAll"  # Replace with your input folder path
    output_dir = f"{base_path_data}\\WeatherDataAllCSVMultiProcess_MultiStation"  # Replace with your output folder path
    timing_log_file = f"{base_path_data}processing_times_multistation.csv"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all files in the input directory
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                 if os.path.isfile(os.path.join(input_dir, f))]
    
    print(f"Found {len(file_paths)} files to process")
    print(f"Processing for stations: {', '.join(stations)}")
    
    # Number of cores to use
    num_cores = 24  # You could also use: min(24, mp.cpu_count())
    
    start_time = time.time()

    # Create a pool of workers
    pool = mp.Pool(processes=num_cores)
    
    print("Starting parallel processing...")
    
    # Create a partial function with fixed output_dir and stations
    process_func = partial(process_file, output_dir=output_dir, stations_list=stations)
    
    # Process files in parallel and collect results
    results = pool.map(process_func, file_paths)
    
    # Close the pool
    pool.close()
    pool.join()
    
    processing_time_seconds = time.time() - start_time
    processing_time_minutes = processing_time_seconds / 60

    # Log timing for the parallel processing phase
    log_execution_time(timing_log_file, "parallel_processing", len(results), 
                      processing_time_seconds, processing_time_minutes, 
                      f"Multiple_Stations_{len(stations)}", num_cores)
    
    # Summary of processing
    successful = [path for path, success, _ in results if success]
    failed = [path for path, success, _ in results if not success]
    
    print(f"\nParallel processing complete!")
    print(f"Successfully processed: {len(successful)} files")
    print(f"Failed to process: {len(failed)} files")
    print(f"Time taken: {processing_time_minutes:.2f} minutes")
    
    # Now combine files by station
    combine_start_time = time.time()
    combine_station_files(output_dir, stations)
    
    combine_time_seconds = time.time() - combine_start_time
    combine_time_minutes = combine_time_seconds / 60
    
    # Log timing for the combining phase
    log_execution_time(timing_log_file, "combine_files", len(stations), 
                      combine_time_seconds, combine_time_minutes, 
                      f"Combine_{len(stations)}_Stations", 1)
    
    total_time_seconds = time.time() - start_time
    total_time_minutes = total_time_seconds / 60
    
    print(f"\nTotal processing time: {total_time_minutes:.2f} minutes")
    print(f"Output files saved to: {output_dir}")

if __name__ == "__main__":
    # This is important for Windows to avoid recursive process spawning
    main()