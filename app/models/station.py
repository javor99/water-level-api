# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Station model and operations."""
import requests
from pyproj import Transformer


def validate_station_exists_in_vandah(station_id):
    """Check if station exists in Vandah API and return metadata if found."""
    try:
        url = "https://vandah.miljoeportal.dk/api/stations?format=json"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            stations = response.json()
            
            # Find the specific station
            for s in stations:
                if s.get('stationId') == station_id:
                    # Convert UTM coordinates to lat/lon
                    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
                    location = s['location']
                    x, y = location['x'], location['y']
                    lon, lat = transformer.transform(x, y)
                    
                    return {
                        'exists': True,
                        'metadata': {
                            'name': s.get('name', ''),
                            'latitude': lat,
                            'longitude': lon,
                            'location_type': s['locationType'].lower(),
                            'station_owner': s.get('stationOwnerName', ''),
                            'description': s.get('description', '')
                        }
                    }
            
            return {'exists': False, 'metadata': None}
        else:
            return {'exists': False, 'error': f'Vandah API returned status {response.status_code}'}
            
    except Exception as e:
        return {'exists': False, 'error': f'Failed to validate station in Vandah: {str(e)}'}


def get_weather_station_info():
    """Get the actual weather station information used for all water level stations."""
    from app.config import WEATHER_STATION_INFO
    return WEATHER_STATION_INFO



