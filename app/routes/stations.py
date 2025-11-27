# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Station routes."""
from flask import Blueprint, request, jsonify
import subprocess
import threading
import logging
from app.middleware.auth import require_role
from app.utils.database import get_db_connection
from app.utils.jwt_utils import get_user_email_from_jwt
from app.models.station import get_weather_station_info, validate_station_exists_in_vandah

logger = logging.getLogger(__name__)

# Create blueprint - note: no prefix for weather-station route
stations_bp = Blueprint('stations', __name__)


@stations_bp.route('/weather-station')
def get_weather_station():
    """Get the actual weather station information used for all water level stations."""
    weather_info = get_weather_station_info()
    return jsonify({
        "success": True,
        "weather_station": weather_info
    })


@stations_bp.route('/stations', methods=['GET'])
def get_stations():
    """List all stations - no authentication required."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner, s.municipality_id, m.name as municipality_name
        FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id
        ORDER BY s.name
    """)
    
    stations = []
    weather_info = get_weather_station_info()
    for row in cursor.fetchall():
        station_data = {
            "station_id": row['station_id'],
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "location_type": row['location_type'],
            "station_owner": row["station_owner"],
            "municipality_id": row["municipality_id"],
            "municipality_name": row["municipality_name"],
            "weather_station_info": weather_info
        }
        stations.append(station_data)
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(stations),
        "stations": stations
    })


@stations_bp.route('/stations', methods=['POST'])
@require_role('admin')
def create_station():
    """Create new station - admin or superadmin required."""
    logger.info("Station creation request received")
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ["station_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        station_id = data["station_id"].strip()
        name = data.get("name", "")
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        location_type = data.get("location_type", "stream")
        station_owner = data.get("station_owner", "")
        municipality_id = data.get("municipality_id")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station already exists in database
        cursor.execute("SELECT station_id FROM stations WHERE station_id = ?", (station_id,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Station with this ID already exists"}), 409

        # Check if station exists in Vandah system
        vandah_validation = validate_station_exists_in_vandah(station_id)
        if not vandah_validation['exists']:
            conn.close()
            error_msg = f"Station {station_id} does not exist in Vandah system"
            if 'error' in vandah_validation:
                error_msg += f": {vandah_validation['error']}"
            return jsonify({"error": error_msg}), 400

        # Use Vandah metadata if available
        vandah_metadata = vandah_validation['metadata']
        if vandah_metadata:
            name = vandah_metadata.get('name', name)
            latitude = vandah_metadata.get('latitude', latitude)
            longitude = vandah_metadata.get('longitude', longitude)
            location_type = vandah_metadata.get('location_type', location_type)
            station_owner = vandah_metadata.get('station_owner', station_owner)
            description = vandah_metadata.get('description', '')
        
        creator_email = get_user_email_from_jwt()
        
        # Insert the new station
        cursor.execute("""
            INSERT INTO stations 
            (station_id, name, latitude, longitude, location_type, station_owner, municipality_id, created_by, updated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (station_id, name, latitude, longitude, location_type, station_owner, municipality_id, creator_email, creator_email))
        
        conn.commit()
        conn.close()
        
        # Start background data generation
        print(f"🔄 Starting background data generation for station {station_id}...")
        try:
            def run_data_update():
                try:
                    result = subprocess.run([
                        'python3', 'utilities/update_new_station_data.py', station_id
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"✅ Background data generation completed for station {station_id}")
                    else:
                        print(f"❌ Background data generation failed for station {station_id}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ Background data generation timed out for station {station_id}")
                except Exception as e:
                    print(f"❌ Error during background data generation for station {station_id}: {str(e)}")
            
            thread = threading.Thread(target=run_data_update)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"❌ Error starting background data generation for station {station_id}: {str(e)}")
        
        return jsonify({
            "message": "Station created successfully with Vandah metadata. Data generation started in background.",
            "station": {
                "station_id": station_id,
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "location_type": location_type,
                "station_owner": station_owner,
                "municipality_id": municipality_id,
                "created_by": creator_email,
                "description": vandah_metadata.get('description', '') if vandah_metadata else ''
            },
            "vandah_metadata": vandah_metadata if vandah_metadata else None,
            "data_update": {
                "status": "started",
                "message": "Data generation is running in the background. This may take 1-3 minutes."
            }
        }), 201
        
    except Exception as e:
        return jsonify({"error": f"Failed to create station: {str(e)}"}), 500


@stations_bp.route('/stations/<station_id>', methods=['GET'])
def get_station(station_id):
    """Get specific station information with weather data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner, s.municipality_id, m.name as municipality_name
        FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id
        WHERE station_id = ?
    """, (station_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    weather_info = get_weather_station_info()
    if row:
        station_data = {
            "station_id": row['station_id'],
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "location_type": row['location_type'],
            "station_owner": row["station_owner"],
            "municipality_id": row["municipality_id"],
            "municipality_name": row["municipality_name"],
            "weather_station_info": weather_info
        }
        
        return jsonify({
            "success": True,
            "station": station_data
        })
    else:
        return jsonify({"success": False, "error": "Station not found"}), 404


@stations_bp.route('/stations/<station_id>', methods=['DELETE'])
@require_role('admin')
def delete_station(station_id):
    """Delete a station and all its associated data - admin or superadmin required."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        if not station:
            conn.close()
            return jsonify({"error": "Station not found"}), 404
        
        station_name = station["name"]
        
        # Delete all associated data
        cursor.execute("DELETE FROM water_levels WHERE station_id = ?", (station_id,))
        cursor.execute("DELETE FROM predictions WHERE station_id = ?", (station_id,))
        cursor.execute("DELETE FROM past_predictions WHERE station_id = ?", (station_id,))
        cursor.execute("DELETE FROM last_30_days_historical WHERE station_id = ?", (station_id,))
        cursor.execute("DELETE FROM stations WHERE station_id = ?", (station_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Station and all associated data deleted successfully",
            "deleted_station": {
                "station_id": station_id,
                "name": station_name
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to delete station: {str(e)}"}), 500


@stations_bp.route('/stations/<station_id>/minmax', methods=['GET'])
def get_station_minmax(station_id):
    """Get current min/max water level values for a specific station."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT station_id, name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        
        if not station:
            conn.close()
            return jsonify({"error": f"Station {station_id} not found"}), 404
        
        cursor.execute("""
            SELECT min_level_cm, max_level_cm, updated_at
            FROM min_max_values 
            WHERE station_id = ?
        """, (station_id,))
        
        minmax_data = cursor.fetchone()
        conn.close()
        
        if not minmax_data:
            return jsonify({
                "station_id": station[0],
                "station_name": station[1],
                "min_level_cm": None,
                "max_level_cm": None,
                "min_level_m": None,
                "max_level_m": None,
                "updated_at": None
            }), 200
        
        min_level_cm = minmax_data[0]
        max_level_cm = minmax_data[1]
        updated_at = minmax_data[2]
        
        return jsonify({
            "station_id": station[0],
            "station_name": station[1],
            "min_level_cm": min_level_cm,
            "max_level_cm": max_level_cm,
            "min_level_m": min_level_cm / 100.0 if min_level_cm else None,
            "max_level_m": max_level_cm / 100.0 if max_level_cm else None,
            "updated_at": updated_at
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get station min/max values: {str(e)}"}), 500


@stations_bp.route('/stations/<station_id>/minmax', methods=['POST'])
@require_role('admin')
def update_station_minmax(station_id):
    """Update min/max water level values for a specific station (admin/superadmin only)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ['min_level_cm', 'max_level_cm']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        min_level_cm = data['min_level_cm']
        max_level_cm = data['max_level_cm']
        
        if not isinstance(min_level_cm, (int, float)) or not isinstance(max_level_cm, (int, float)):
            return jsonify({"error": "min_level_cm and max_level_cm must be numbers"}), 400
        
        if min_level_cm >= max_level_cm:
            return jsonify({"error": "min_level_cm must be less than max_level_cm"}), 400
        
        min_level_m = min_level_cm / 100.0
        max_level_m = max_level_cm / 100.0
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT s.station_id, s.name FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id WHERE s.station_id = ?", (station_id,))
        station = cursor.fetchone()
        
        if not station:
            conn.close()
            return jsonify({"error": f"Station {station_id} not found"}), 404
        
        cursor.execute("""
            INSERT OR REPLACE INTO min_max_values (station_id, min_level_cm, max_level_cm, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (station_id, min_level_cm, max_level_cm))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Successfully updated min/max values for station {station_id}",
            "station_id": station_id,
            "station_name": station[1],
            "updated_by": "system",
            "updated_values": {
                "min_level_cm": min_level_cm,
                "max_level_cm": max_level_cm,
                "min_level_m": min_level_m,
                "max_level_m": max_level_m
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update station min/max values: {str(e)}"}), 500


@stations_bp.route('/stations/minmax/bulk', methods=['POST'])
@require_role('admin')
def update_bulk_station_minmax():
    """Update min/max water level values for multiple stations at once (admin/superadmin only)."""
    try:
        data = request.get_json()
        
        if not data or 'stations' not in data:
            return jsonify({"error": "No stations data provided. Expected format: {'stations': [{'station_id': '...', 'min_level_cm': ..., 'max_level_cm': ...}]}"}), 400
        
        stations_data = data['stations']
        
        if not isinstance(stations_data, list):
            return jsonify({"error": "stations must be a list"}), 400
        
        results = []
        errors = []
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for i, station_data in enumerate(stations_data):
            try:
                if 'station_id' not in station_data:
                    errors.append(f"Station {i}: Missing station_id")
                    continue
                
                if 'min_level_cm' not in station_data or 'max_level_cm' not in station_data:
                    errors.append(f"Station {station_data.get('station_id', i)}: Missing min_level_cm or max_level_cm")
                    continue
                
                station_id = station_data['station_id']
                min_level_cm = station_data['min_level_cm']
                max_level_cm = station_data['max_level_cm']
                
                if not isinstance(min_level_cm, (int, float)) or not isinstance(max_level_cm, (int, float)):
                    errors.append(f"Station {station_id}: min_level_cm and max_level_cm must be numbers")
                    continue
                
                if min_level_cm >= max_level_cm:
                    errors.append(f"Station {station_id}: min_level_cm must be less than max_level_cm")
                    continue
                
                cursor.execute("SELECT s.name FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id WHERE s.station_id = ?", (station_id,))
                station = cursor.fetchone()
                
                if not station:
                    errors.append(f"Station {station_id}: Not found")
                    continue
                
                min_level_m = min_level_cm / 100.0
                max_level_m = max_level_cm / 100.0
                
                cursor.execute("""
                    UPDATE stations 
                    SET min_level_cm = ?, max_level_cm = ?, min_level_m = ?, max_level_m = ?
                    WHERE station_id = ?
                """, (min_level_cm, max_level_cm, min_level_m, max_level_m, station_id))
                
                results.append({
                    "station_id": station_id,
                    "station_name": station[0],
                    "updated_values": {
                        "min_level_cm": min_level_cm,
                        "max_level_cm": max_level_cm,
                        "min_level_m": min_level_m,
                        "max_level_m": max_level_m
                    }
                })
                
            except Exception as e:
                errors.append(f"Station {station_data.get('station_id', i)}: {str(e)}")
        
        conn.commit()
        conn.close()
        
        response = {
            "message": f"Bulk update completed. {len(results)} stations updated successfully.",
            "updated_by": "system",
            "updated_stations": results,
            "total_updated": len(results),
            "total_errors": len(errors)
        }
        
        if errors:
            response["errors"] = errors
        
        return jsonify(response), 200 if not errors else 207
        
    except Exception as e:
        return jsonify({"error": f"Failed to update stations: {str(e)}"}), 500
