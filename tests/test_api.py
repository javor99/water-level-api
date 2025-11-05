#!/usr/bin/env python3
import requests
import time

def test_api():
    base_url = "http://localhost:5001"
    
    print("🧪 Testing API Endpoints")
    print("=" * 50)
    
    # Test 1: Create municipality
    print("1. Creating municipality 'Struer'...")
    try:
        response = requests.post(f"{base_url}/municipalities", 
                               json={
                                   "name": "Struer", 
                                   "region": "Midtjylland", 
                                   "population": 22000, 
                                   "area_km2": 250.0
                               }, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            print("   ✅ Municipality created successfully")
            municipality_data = response.json()
            print(f"   Municipality ID: {municipality_data.get('municipality', {}).get('id', 'N/A')}")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 2: Create station
    print("2. Creating station '70000600'...")
    try:
        response = requests.post(f"{base_url}/stations", 
                               json={
                                   "station_id": "70000600", 
                                   "name": "Struer Water Station", 
                                   "latitude": 56.5, 
                                   "longitude": 8.6,
                                   "location_type": "stream",
                                   "station_owner": "Struer Municipality",
                                   "municipality_id": 1
                               }, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 201:
            print("   ✅ Station created successfully")
            station_data = response.json()
            print(f"   Station: {station_data.get('station', {}).get('name', 'N/A')}")
            if 'data_update' in station_data:
                print(f"   🔄 Data update: {station_data['data_update']['status']}")
        else:
            print(f"   ❌ Failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test 3: Check if station exists
    print("3. Checking if station exists...")
    try:
        response = requests.get(f"{base_url}/stations/70000600", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Station found")
        else:
            print(f"   ❌ Station not found: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_api()
