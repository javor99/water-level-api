# Water Level API - CRUD Operations Guide

**Role limitations have been removed - all endpoints are now publicly accessible without authentication requirements.**

## Base URL
```
http://localhost:5001
```

## Municipality CRUD Operations

### 1. List All Municipalities
```bash
curl -X GET "http://localhost:5001/municipalities"
```

**Response:**
```json
{
  "success": true,
  "count": 5,
  "municipalities": [
    {
      "id": 1,
      "name": "Copenhagen",
      "region": "Capital Region",
      "population": 650000,
      "area_km2": 86.4,
      "description": "Capital city of Denmark",
      "created_at": "2024-01-01 10:00:00",
      "created_by": "admin@admin.com",
      "updated_at": "2024-01-01 10:00:00",
      "updated_by": "admin@admin.com",
      "station_count": 3
    }
  ]
}
```

### 2. Get Specific Municipality
```bash
curl -X GET "http://localhost:5001/municipalities/1"
```

**Response:**
```json
{
  "success": true,
  "municipality": {
    "id": 1,
    "name": "Copenhagen",
    "region": "Capital Region",
    "population": 650000,
    "area_km2": 86.4,
    "description": "Capital city of Denmark",
    "created_at": "2024-01-01 10:00:00",
    "created_by": "admin@admin.com",
    "updated_at": "2024-01-01 10:00:00",
    "updated_by": "admin@admin.com",
    "stations": [
      {
        "station_id": "70000864",
        "name": "Borup station",
        "latitude": 55.483,
        "longitude": 11.967,
        "location_type": "stream",
        "station_owner": "Miljøstyrelsen"
      }
    ]
  }
}
```

### 3. Create New Municipality
```bash
curl -X POST "http://localhost:5001/municipalities" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Aarhus",
    "region": "Central Jutland",
    "population": 350000,
    "area_km2": 468.0,
    "description": "Second largest city in Denmark"
  }'
```

**Response:**
```json
{
  "message": "Municipality created successfully",
  "municipality": {
    "id": 6,
    "name": "Aarhus",
    "region": "Central Jutland",
    "population": 350000,
    "area_km2": 468.0,
    "description": "Second largest city in Denmark",
    "created_by": null
  }
}
```

### 4. Update Municipality
```bash
curl -X PUT "http://localhost:5001/municipalities/6" \
  -H "Content-Type: application/json" \
  -d '{
    "population": 375000,
    "description": "Updated: Second largest city in Denmark with growing population"
  }'
```

**Response:**
```json
{
  "message": "Municipality updated successfully",
  "municipality": {
    "id": 6,
    "name": "Aarhus",
    "region": "Central Jutland",
    "population": 375000,
    "area_km2": 468.0,
    "description": "Updated: Second largest city in Denmark with growing population",
    "updated_by": null
  }
}
```

### 5. Delete Municipality
```bash
curl -X DELETE "http://localhost:5001/municipalities/6"
```

**Response:**
```json
{
  "message": "Municipality 'Aarhus' deleted successfully"
}
```

## User CRUD Operations

### 1. List All Users
```bash
curl -X GET "http://localhost:5001/auth/users"
```

**Response:**
```json
{
  "success": true,
  "count": 3,
  "users": [
    {
      "id": 1,
      "email": "superadmin@superadmin.com",
      "role": "superadmin",
      "created_at": "2024-01-01 10:00:00",
      "last_login": "2024-01-15 14:30:00",
      "is_active": true,
      "created_by": null
    },
    {
      "id": 2,
      "email": "admin@admin.com",
      "role": "admin",
      "created_at": "2024-01-01 10:05:00",
      "last_login": "2024-01-15 12:00:00",
      "is_active": true,
      "created_by": "superadmin@superadmin.com"
    }
  ]
}
```

### 2. Register New User
```bash
curl -X POST "http://localhost:5001/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "password": "securepassword123",
    "role": "user"
  }'
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": 4,
    "email": "john.doe@example.com",
    "role": "user"
  }
}
```

### 3. Update User
```bash
curl -X PUT "http://localhost:5001/auth/users/4" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.smith@example.com",
    "role": "admin",
    "password": "newsecurepassword456"
  }'
```

**Response:**
```json
{
  "message": "User updated successfully"
}
```

### 4. Delete User
```bash
curl -X DELETE "http://localhost:5001/auth/users/4"
```

**Response:**
```json
{
  "message": "User john.smith@example.com deleted successfully",
  "deleted_user": {
    "id": 4,
    "email": "john.smith@example.com",
    "role": "admin"
  }
}
```

## Additional Municipality Operations

### Get Stations by Municipality
```bash
# Get all stations with their municipalities
curl -X GET "http://localhost:5001/municipalities/stations"

# Get stations for specific municipalities
curl -X GET "http://localhost:5001/municipalities/stations?municipality_id=1&municipality_id=2"

# Include weather data
curl -X GET "http://localhost:5001/municipalities/stations?include_weather=true"
```

### Assign Stations to Municipality
```bash
curl -X POST "http://localhost:5001/municipalities/1/stations" \
  -H "Content-Type: application/json" \
  -d '{
    "station_ids": ["70000864", "70000865", "70000923"]
  }'
```

## User Authentication (Optional)

### Login User
```bash
curl -X POST "http://localhost:5001/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@admin.com",
    "password": "12345678"
  }'
```

**Response:**
```json
{
  "message": "Login successful",
  "user": {
    "id": 2,
    "email": "admin@admin.com",
    "role": "admin"
  },
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Verify Token
```bash
curl -X GET "http://localhost:5001/auth/verify" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Python Example Scripts

### Python - Municipality CRUD
```python
import requests
import json

BASE_URL = "http://localhost:5001"

# Create municipality
def create_municipality(name, region=None, population=None):
    data = {"name": name}
    if region:
        data["region"] = region
    if population:
        data["population"] = population
    
    response = requests.post(f"{BASE_URL}/municipalities", json=data)
    return response.json()

# Get all municipalities
def get_municipalities():
    response = requests.get(f"{BASE_URL}/municipalities")
    return response.json()

# Update municipality
def update_municipality(municipality_id, **kwargs):
    response = requests.put(f"{BASE_URL}/municipalities/{municipality_id}", json=kwargs)
    return response.json()

# Delete municipality
def delete_municipality(municipality_id):
    response = requests.delete(f"{BASE_URL}/municipalities/{municipality_id}")
    return response.json()

# Example usage
if __name__ == "__main__":
    # Create a new municipality
    result = create_municipality("Test City", "Test Region", 100000)
    print("Created:", result)
    
    # Get all municipalities
    municipalities = get_municipalities()
    print("All municipalities:", municipalities)
    
    # Update the municipality (assuming ID 1)
    updated = update_municipality(1, population=105000)
    print("Updated:", updated)
```

### Python - User CRUD
```python
import requests
import json

BASE_URL = "http://localhost:5001"

# Register user
def register_user(email, password, role="user"):
    data = {
        "email": email,
        "password": password,
        "role": role
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=data)
    return response.json()

# Get all users
def get_users():
    response = requests.get(f"{BASE_URL}/auth/users")
    return response.json()

# Update user
def update_user(user_id, **kwargs):
    response = requests.put(f"{BASE_URL}/auth/users/{user_id}", json=kwargs)
    return response.json()

# Delete user
def delete_user(user_id):
    response = requests.delete(f"{BASE_URL}/auth/users/{user_id}")
    return response.json()

# Login user
def login_user(email, password):
    data = {"email": email, "password": password}
    response = requests.post(f"{BASE_URL}/auth/login", json=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Register a new user
    result = register_user("test@example.com", "password123", "user")
    print("Registered:", result)
    
    # Login the user
    login_result = login_user("test@example.com", "password123")
    print("Login:", login_result)
    
    # Get all users
    users = get_users()
    print("All users:", users)
```

## Common HTTP Status Codes

- **200**: Success
- **201**: Created successfully
- **400**: Bad request (missing/invalid data)
- **404**: Resource not found
- **409**: Conflict (duplicate email/name)
- **500**: Internal server error

## Required Fields

### Municipality Creation
- **name** (required): Municipality name
- **region** (optional): Geographic region
- **population** (optional): Population count
- **area_km2** (optional): Area in square kilometers
- **description** (optional): Description text

### User Registration
- **email** (required): Valid email address
- **password** (required): Minimum 8 characters
- **role** (optional): "user", "admin", or "superadmin" (defaults to "user")

## Notes

1. **No Authentication Required**: All endpoints are publicly accessible
2. **Data Validation**: The API validates email formats, password lengths, and required fields
3. **Duplicate Prevention**: Cannot create users with duplicate emails or municipalities with duplicate names
4. **Station Dependencies**: Municipalities with assigned stations cannot be deleted until stations are reassigned
5. **Default Users**: The system creates default admin users on startup:
   - superadmin@superadmin.com (password: 12345678)
   - admin@admin.com (password: 12345678)

## Error Handling

All endpoints return JSON responses with error messages:

```json
{
  "error": "Description of what went wrong"
}
```

For successful operations, responses include:
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {...}
}
```

## Water Level Station CRUD Operations

### 1. List All Stations
```bash
curl -X GET "http://localhost:5001/stations"
```

**Response:**
```json
{
  "success": true,
  "count": 10,
  "stations": [
    {
      "station_id": "70000864",
      "name": "Borup station",
      "latitude": 55.483,
      "longitude": 11.967,
      "location_type": "stream",
      "station_owner": "Miljøstyrelsen",
      "municipality_id": 1,
      "municipality_name": "Høje-Taastrup",
      "weather_station_info": {
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
    }
  ]
}
```

### 2. Get Specific Station
```bash
curl -X GET "http://localhost:5001/stations/70000864"
```

**Response:**
```json
{
  "success": true,
  "station": {
    "station_id": "70000864",
    "name": "Borup station",
    "latitude": 55.483,
    "longitude": 11.967,
    "location_type": "stream",
    "station_owner": "Miljøstyrelsen",
    "municipality_id": 1,
    "municipality_name": "Høje-Taastrup",
    "weather_station_info": {
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
  }
}
```

### 3. Create New Station
```bash
curl -X POST "http://localhost:5001/stations" \
  -H "Content-Type: application/json" \
  -d '{
    "station_id": "70000999",
    "name": "New Test Station",
    "latitude": 55.500,
    "longitude": 12.000,
    "location_type": "stream",
    "station_owner": "Test Organization",
    "municipality_id": 1
  }'
```

**Response:**
```json
{
  "message": "Station created successfully",
  "station": {
    "station_id": "70000999",
    "name": "New Test Station",
    "latitude": 55.500,
    "longitude": 12.000,
    "location_type": "stream",
    "station_owner": "Test Organization",
    "municipality_id": 1
  }
}
```

### 4. Update Station
```bash
curl -X PUT "http://localhost:5001/stations/70000999" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Test Station",
    "latitude": 55.510,
    "longitude": 12.010,
    "station_owner": "Updated Organization"
  }'
```

**Response:**
```json
{
  "message": "Station updated successfully",
  "station": {
    "station_id": "70000999",
    "name": "Updated Test Station",
    "latitude": 55.510,
    "longitude": 12.010,
    "location_type": "stream",
    "station_owner": "Updated Organization",
    "municipality_id": 1
  }
}
```

### 5. Delete Station
```bash
curl -X DELETE "http://localhost:5001/stations/70000999"
```

**Response:**
```json
{
  "message": "Station 'Updated Test Station' deleted successfully"
}
```

### 6. Delete Station Data (Water Levels & Predictions)
```bash
curl -X DELETE "http://localhost:5001/stations/70000999/data"
```

**Response:**
```json
{
  "message": "Station data deleted successfully",
  "station_name": "Updated Test Station",
  "water_levels_deleted": 150,
  "predictions_deleted": 30
}
```

## Station Min/Max Value Operations

### 1. Get Station Min/Max Values
```bash
curl -X GET "http://localhost:5001/stations/70000864/minmax"
```

**Response:**
```json
{
  "station_id": "70000864",
  "station_name": "Borup station",
  "min_level_cm": 25.5,
  "max_level_cm": 180.2,
  "min_level_m": 0.255,
  "max_level_m": 1.802
}
```

### 2. Update Station Min/Max Values
```bash
curl -X POST "http://localhost:5001/stations/70000864/minmax" \
  -H "Content-Type: application/json" \
  -d '{
    "min_level_cm": 20.0,
    "max_level_cm": 200.0
  }'
```

**Response:**
```json
{
  "message": "Successfully updated min/max values for station 70000864",
  "station_id": "70000864",
  "station_name": "Borup station",
  "updated_by": null,
  "updated_values": {
    "min_level_cm": 20.0,
    "max_level_cm": 200.0,
    "min_level_m": 0.2,
    "max_level_m": 2.0
  }
}
```

### 3. Bulk Update Min/Max Values
```bash
curl -X POST "http://localhost:5001/stations/minmax/bulk" \
  -H "Content-Type: application/json" \
  -d '{
    "stations": [
      {
        "station_id": "70000864",
        "min_level_cm": 20.0,
        "max_level_cm": 200.0
      },
      {
        "station_id": "70000865",
        "min_level_cm": 15.0,
        "max_level_cm": 180.0
      }
    ]
  }'
```

**Response:**
```json
{
  "message": "Bulk update completed. 2 stations updated successfully.",
  "updated_by": null,
  "updated_stations": [
    {
      "station_id": "70000864",
      "station_name": "Borup station",
      "updated_values": {
        "min_level_cm": 20.0,
        "max_level_cm": 200.0,
        "min_level_m": 0.2,
        "max_level_m": 2.0
      }
    },
    {
      "station_id": "70000865",
      "station_name": "Bygholm station",
      "updated_values": {
        "min_level_cm": 15.0,
        "max_level_cm": 180.0,
        "min_level_m": 0.15,
        "max_level_m": 1.8
      }
    }
  ],
  "total_updated": 2,
  "total_errors": 0
}
```

## Python Example Scripts - Station CRUD

### Python - Station CRUD
```python
import requests
import json

BASE_URL = "http://localhost:5001"

# Create station
def create_station(station_id, name, latitude=None, longitude=None, location_type="stream", station_owner=None, municipality_id=None):
    data = {
        "station_id": station_id,
        "name": name,
        "location_type": location_type
    }
    if latitude:
        data["latitude"] = latitude
    if longitude:
        data["longitude"] = longitude
    if station_owner:
        data["station_owner"] = station_owner
    if municipality_id:
        data["municipality_id"] = municipality_id
    
    response = requests.post(f"{BASE_URL}/stations", json=data)
    return response.json()

# Get all stations
def get_stations():
    response = requests.get(f"{BASE_URL}/stations")
    return response.json()

# Get specific station
def get_station(station_id):
    response = requests.get(f"{BASE_URL}/stations/{station_id}")
    return response.json()

# Update station
def update_station(station_id, **kwargs):
    response = requests.put(f"{BASE_URL}/stations/{station_id}", json=kwargs)
    return response.json()

# Delete station
def delete_station(station_id):
    response = requests.delete(f"{BASE_URL}/stations/{station_id}")
    return response.json()

# Delete station data
def delete_station_data(station_id):
    response = requests.delete(f"{BASE_URL}/stations/{station_id}/data")
    return response.json()

# Update station min/max values
def update_station_minmax(station_id, min_level_cm, max_level_cm):
    data = {
        "min_level_cm": min_level_cm,
        "max_level_cm": max_level_cm
    }
    response = requests.post(f"{BASE_URL}/stations/{station_id}/minmax", json=data)
    return response.json()

# Bulk update min/max values
def bulk_update_minmax(stations_data):
    data = {"stations": stations_data}
    response = requests.post(f"{BASE_URL}/stations/minmax/bulk", json=data)
    return response.json()

# Example usage
if __name__ == "__main__":
    # Create a new station
    result = create_station("70000999", "Test Station", 55.500, 12.000, "stream", "Test Org", 1)
    print("Created:", result)
    
    # Get all stations
    stations = get_stations()
    print("All stations:", stations)
    
    # Update the station
    updated = update_station("70000999", name="Updated Test Station", latitude=55.510)
    print("Updated:", updated)
    
    # Update min/max values
    minmax_result = update_station_minmax("70000999", 10.0, 150.0)
    print("Min/Max updated:", minmax_result)
    
    # Bulk update multiple stations
    bulk_data = [
        {"station_id": "70000864", "min_level_cm": 20.0, "max_level_cm": 200.0},
        {"station_id": "70000865", "min_level_cm": 15.0, "max_level_cm": 180.0}
    ]
    bulk_result = bulk_update_minmax(bulk_data)
    print("Bulk update:", bulk_result)
    
    # Delete station data
    delete_data_result = delete_station_data("70000999")
    print("Data deleted:", delete_data_result)
    
    # Delete the station
    deleted = delete_station("70000999")
    print("Deleted:", deleted)
```

## Station Required Fields

### Station Creation
- **station_id** (required): Unique station identifier
- **name** (required): Station name
- **latitude** (optional): Latitude coordinate
- **longitude** (optional): Longitude coordinate
- **location_type** (optional): Type of location (defaults to "stream")
- **station_owner** (optional): Organization that owns the station
- **municipality_id** (optional): ID of the municipality this station belongs to

### Station Min/Max Update
- **min_level_cm** (required): Minimum water level in centimeters
- **max_level_cm** (required): Maximum water level in centimeters

## Station Data Dependencies

1. **Station Deletion**: Cannot delete a station that has water level or prediction data
2. **Data Deletion**: Use the `/stations/{id}/data` endpoint to delete all related data first
3. **Municipality Assignment**: Stations can be assigned to municipalities via the municipality endpoints
4. **Min/Max Values**: Automatically converted to meters (cm/100) when updated

## Adding Station CRUD to Your Server

To add these station CRUD endpoints to your server, you can:

1. **Copy the endpoints**: Copy the contents of `station_endpoints.py` into your main server file
2. **Import the endpoints**: Add `from station_endpoints import *` to your main server file
3. **Restart the server**: The new endpoints will be available immediately

The station CRUD endpoints are designed to work with the existing database schema and don't require any authentication.
