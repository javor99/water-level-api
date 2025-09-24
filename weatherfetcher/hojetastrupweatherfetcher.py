import requests
import pandas as pd

# Fetch historical hourly weather data from Open-Meteo API
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 55.65,
    "longitude": 12.27,
    "past_days": 10,  # Fetch 10 days of historical data
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "Europe/Copenhagen"
}

response = requests.get(url, params=params)
data = response.json()

# Convert hourly data to DataFrame
df = pd.DataFrame({
    'time': data['hourly']['time'],
    'temperature_2m': data['hourly']['temperature_2m'],  # already in °C
    'relative_humidity_2m': data['hourly']['relative_humidity_2m'],  # in %
    'wind_speed_10m': data['hourly']['wind_speed_10m']    # in km/h
})

# Convert time to datetime and extract date
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date

# Convert wind speed from km/h to m/s (1 km/h = 0.27778 m/s)
df['wind_speed_10m'] = df['wind_speed_10m'] * 0.27778

# Aggregate data by date
daily_df = df.groupby('date').agg({
    'temperature_2m': 'mean',    # Daily average temperature in °C
    'relative_humidity_2m': 'mean',  # Daily average relative humidity in %
    'wind_speed_10m': 'mean'     # Daily average wind speed in m/s
}).reset_index()

# Rename columns for final output
daily_df.rename(columns={
    'temperature_2m': 'temperature-2m',
    'relative_humidity_2m': 'relative-humidity-2m',
    'wind_speed_10m': 'wind-speed-10m'
}, inplace=True)

print(daily_df)

