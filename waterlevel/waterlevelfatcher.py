import requests
from pyproj import Transformer
from datetime import datetime, timedelta
from collections import defaultdict

API_BASE = "https://vandah.miljoeportal.dk/api"
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

def get_hoje_taastrup_stations():
    url = f"{API_BASE}/stations?format=json"
    response = requests.get(url)
    response.raise_for_status()
    stations = response.json()
    return [s for s in stations if s['stationOwnerName'] == "Høje Taastrup kommune"]

def fetch_water_levels(station_id, from_time, to_time):
    url = (
        f"{API_BASE}/water-levels?"
        f"stationId={station_id}"
        f"&from={from_time}"
        f"&to={to_time}"
        f"&format=json"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def aggregate_daily_average(results):
    daily_data = defaultdict(list)
    for record in results:
        date = record['measurementDateTime'][:10]  # Extract YYYY-MM-DD
        daily_data[date].append(record['result'])

    daily_averages = [
        {"date": date, "water_level": round(sum(values) / len(values), 2)}
        for date, values in sorted(daily_data.items())
    ]

    return daily_averages

def main():
    stations = get_hoje_taastrup_stations()

    to_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    from_time = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%dT%H:%MZ")

    for station in stations:
        station_id = station['stationId']
        operator_station_id = station.get('operatorStationId') or 'N/A'
        x, y = station['location']['x'], station['location']['y']
        lon, lat = transformer.transform(x, y)

        print(f"\nStation: {station['name']}")
        print(f"Station ID: {station_id}")
        print(f"Operator Station ID: {operator_station_id}")
        print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")

        water_data = fetch_water_levels(station_id, from_time, to_time)

        if not water_data or not water_data[0]['results']:
            print("  No water level data found in this period.")
            continue

        daily_averages = aggregate_daily_average(water_data[0]['results'])

        print("Daily Average Water Levels (Last 30 Days):")
        for entry in daily_averages:
            print(f"  {entry['date']}: {entry['water_level']} cm")

if __name__ == "__main__":
    main()

