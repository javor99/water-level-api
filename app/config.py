"""Application configuration."""
import os

# Security - Fixed SECRET_KEY so tokens survive server restarts
SECRET_KEY = os.environ.get('SECRET_KEY', 'water-level-api-secret-key-fixed-do-not-change-in-production')
JWT_EXPIRATION_HOURS = 24

# Database
DB_PATH = "water_levels.db"

# Server
HOST = "0.0.0.0"
PORT = 5001

# Weather Station Info
WEATHER_STATION_INFO = {
    "weather_station_id": "copenhagen_meteorological",
    "weather_station_name": "Copenhagen Meteorological Station",
    "weather_station_latitude": 55.681,
    "weather_station_longitude": 12.285095,
    "weather_station_elevation": 19.0,
    "weather_data_source": "Open-Meteo API",
    "weather_api_url": "https://api.open-meteo.com/v1/forecast",
    "weather_model": "DMI HARMONIE AROME",
    "weather_timezone": "Europe/Copenhagen",
    "weather_timezone_abbreviation": "GMT+2",
    "weather_coverage": "All water level stations use weather data from this single Copenhagen location",
    "weather_update_frequency": "Every 3 hours",
    "weather_forecast_length": "Up to 10 days"
}

