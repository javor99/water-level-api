import requests
from pyproj import Transformer

# Define the transformer from UTM Zone 32N (EPSG:25832) to WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Fetch station data from the API
url = "https://vandah.miljoeportal.dk/api/stations?format=json"
response = requests.get(url)
stations = response.json()

# Filter for Høje Taastrup kommune
hoje_taastrup_stations = [
    station for station in stations if station.get('stationOwnerName') == "Høje Taastrup kommune"
]

# Process each station
for station in hoje_taastrup_stations:
    location = station['location']
    x, y = location['x'], location['y']
    lon, lat = transformer.transform(x, y)

    print(f"Station: {station['name']}")
    print(f"Station ID: {station['stationId']}")
    print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}\n")

