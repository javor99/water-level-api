"""Municipality routes."""
from flask import Blueprint, request, jsonify
from datetime import datetime
from app.middleware.auth import require_role
from app.utils.database import get_db_connection
from app.utils.jwt_utils import get_user_email_from_jwt
from app.models.station import get_weather_station_info
import logging

logger = logging.getLogger(__name__)

municipalities_bp = Blueprint('municipalities', __name__, url_prefix='/municipalities')


@municipalities_bp.route("", methods=["GET"])
def list_municipalities():
    """List all municipalities (public access)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.id, m.name, m.region, m.population, m.area_km2, 
                   m.description, m.created_at, m.created_by, m.updated_at, m.updated_by,
                   COUNT(s.station_id) as station_count
            FROM municipalities m
            LEFT JOIN stations s ON m.id = s.municipality_id
            GROUP BY m.id
            ORDER BY m.name
        """)
        
        municipalities = []
        for row in cursor.fetchall():
            municipalities.append({
                "id": row["id"],
                "name": row["name"],
                "region": row["region"],
                "population": row["population"],
                "area_km2": row["area_km2"],
                "description": row["description"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "updated_at": row["updated_at"],
                "updated_by": row["updated_by"],
                "station_count": row["station_count"]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "count": len(municipalities),
            "municipalities": municipalities
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to list municipalities: {str(e)}"}), 500


@municipalities_bp.route("/<int:municipality_id>", methods=["GET"])
def get_municipality(municipality_id):
    """Get a specific municipality (public access)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT m.id, m.name, m.region, m.population, m.area_km2, 
                   m.description, m.created_at, m.created_by, m.updated_at, m.updated_by
            FROM municipalities m
            WHERE m.id = ?
        """, (municipality_id,))
        
        municipality = cursor.fetchone()
        if not municipality:
            conn.close()
            return jsonify({"error": "Municipality not found"}), 404
        
        # Get stations for this municipality
        cursor.execute("""
            SELECT station_id, name, latitude, longitude, location_type, station_owner
            FROM stations
            WHERE municipality_id = ?
            ORDER BY name
        """, (municipality_id,))
        
        stations = []
        for row in cursor.fetchall():
            stations.append({
                "station_id": row["station_id"],
                "name": row["name"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "location_type": row["location_type"],
                "station_owner": row["station_owner"]
            })
        
        conn.close()
        
        return jsonify({
            "success": True,
            "municipality": {
                "id": municipality["id"],
                "name": municipality["name"],
                "region": municipality["region"],
                "population": municipality["population"],
                "area_km2": municipality["area_km2"],
                "description": municipality["description"],
                "created_at": municipality["created_at"],
                "created_by": municipality["created_by"],
                "updated_at": municipality["updated_at"],
                "updated_by": municipality["updated_by"],
                "stations": stations
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get municipality: {str(e)}"}), 500


@municipalities_bp.route("", methods=["POST"])
@require_role("superadmin")
def create_municipality():
    """Create a new municipality (superadmin only)."""
    logger.info("Municipality creation request received")
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        required_fields = ["name"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        name = data.get("name", "").strip()
        region = data.get("region", "").strip()
        population = data.get("population")
        area_km2 = data.get("area_km2")
        description = data.get("description", "").strip()
        
        if not name:
            return jsonify({"error": "Municipality name cannot be empty"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if municipality already exists
        cursor.execute("SELECT id FROM municipalities WHERE name = ?", (name,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Municipality with this name already exists"}), 409
        
        cursor.execute("""
            INSERT INTO municipalities 
            (name, region, population, area_km2, description, created_by, updated_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, region, population, area_km2, description, 
              get_user_email_from_jwt(), get_user_email_from_jwt(), 
              datetime.now().isoformat(), datetime.now().isoformat()))
        
        municipality_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Municipality created successfully",
            "municipality": {
                "id": municipality_id,
                "name": name,
                "region": region,
                "population": population,
                "area_km2": area_km2,
                "description": description,
                "created_by": get_user_email_from_jwt()
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Municipality creation failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to create municipality: {str(e)}"}), 500


@municipalities_bp.route("/<int:municipality_id>", methods=["PUT"])
def update_municipality(municipality_id):
    """Update a municipality (superadmin only)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if municipality exists
        cursor.execute("SELECT id, name, region, population, area_km2, description FROM municipalities WHERE id = ?", (municipality_id,))
        municipality = cursor.fetchone()
        if not municipality:
            conn.close()
            return jsonify({"error": "Municipality not found"}), 404
        
        # Update fields
        name = data.get("name", municipality["name"]).strip()
        region = data.get("region", municipality["region"] or "").strip()
        population = data.get("population", municipality["population"])
        area_km2 = data.get("area_km2", municipality["area_km2"])
        description = data.get("description", municipality["description"] or "").strip()
        
        if not name:
            return jsonify({"error": "Municipality name cannot be empty"}), 400
        
        # Check if new name conflicts with existing municipality
        if name != municipality["name"]:
            cursor.execute("SELECT id FROM municipalities WHERE name = ? AND id != ?", (name, municipality_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": "Municipality with this name already exists"}), 409
        
        cursor.execute("""
            UPDATE municipalities 
            SET name = ?, region = ?, population = ?, area_km2 = ?, description = ?, 
                updated_at = CURRENT_TIMESTAMP, updated_by = ?
            WHERE id = ?
        """, (name, region, population, area_km2, description, 
              "system", municipality_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Municipality updated successfully",
            "municipality": {
                "id": municipality_id,
                "name": name,
                "region": region,
                "population": population,
                "area_km2": area_km2,
                "description": description,
                "updated_by": "system"
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update municipality: {str(e)}"}), 500


@municipalities_bp.route("/<int:municipality_id>", methods=["DELETE"])
def delete_municipality(municipality_id):
    """Delete a municipality (superadmin only)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if municipality exists
        cursor.execute("SELECT id, name FROM municipalities WHERE id = ?", (municipality_id,))
        municipality = cursor.fetchone()
        if not municipality:
            conn.close()
            return jsonify({"error": "Municipality not found"}), 404
        
        # Check if municipality has stations
        cursor.execute("SELECT COUNT(*) FROM stations WHERE municipality_id = ?", (municipality_id,))
        station_count = cursor.fetchone()[0]
        
        if station_count > 0:
            conn.close()
            return jsonify({
                "error": f"Cannot delete municipality. It has {station_count} associated stations. Please reassign or delete stations first."
            }), 409
        
        cursor.execute("DELETE FROM municipalities WHERE id = ?", (municipality_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Municipality '{municipality['name']}' deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to delete municipality: {str(e)}"}), 500


@municipalities_bp.route("/<int:municipality_id>/stations", methods=["POST"])
def assign_stations_to_municipality(municipality_id):
    """Assign water level stations to a municipality (superadmin only)."""
    try:
        data = request.get_json()
        
        if not data or 'station_ids' not in data:
            return jsonify({"error": "No station_ids provided"}), 400
        
        station_ids = data['station_ids']
        if not isinstance(station_ids, list):
            return jsonify({"error": "station_ids must be a list"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if municipality exists
        cursor.execute("SELECT id, name FROM municipalities WHERE id = ?", (municipality_id,))
        municipality = cursor.fetchone()
        if not municipality:
            conn.close()
            return jsonify({"error": "Municipality not found"}), 404
        
        # Check if all stations exist
        placeholders = ','.join(['?' for _ in station_ids])
        cursor.execute(f"SELECT station_id FROM stations WHERE station_id IN ({placeholders})", station_ids)
        existing_stations = [row['station_id'] for row in cursor.fetchall()]
        
        missing_stations = set(station_ids) - set(existing_stations)
        if missing_stations:
            conn.close()
            return jsonify({"error": f"Stations not found: {list(missing_stations)}"}), 404
        
        # Assign stations to municipality
        assigned_count = 0
        for station_id in station_ids:
            cursor.execute("""
                UPDATE stations 
                SET municipality_id = ? 
                WHERE station_id = ?
            """, (municipality_id, station_id))
            if cursor.rowcount > 0:
                assigned_count += 1
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"Successfully assigned {assigned_count} stations to municipality '{municipality['name']}'",
            "municipality_id": municipality_id,
            "municipality_name": municipality['name'],
            "assigned_stations": assigned_count,
            "station_ids": station_ids
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to assign stations: {str(e)}"}), 500


@municipalities_bp.route("/stations", methods=["GET"])
def get_stations_by_municipalities():
    """Get water level stations by municipalities (all combinations supported)."""
    try:
        # Get query parameters
        municipality_ids = request.args.getlist('municipality_id')
        include_weather = request.args.get('include_weather', 'true').lower() == 'true'
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query based on municipality filter
        if municipality_ids:
            # Filter by specific municipalities
            placeholders = ','.join(['?' for _ in municipality_ids])
            query = f"""
                SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner,
                       s.municipality_id, m.name as municipality_name,
                       s.last_30_days_min_cm, s.last_30_days_max_cm, s.last_30_days_min_m, s.last_30_days_max_m
                FROM stations s
                LEFT JOIN municipalities m ON s.municipality_id = m.id
                WHERE s.municipality_id IN ({placeholders})
                ORDER BY m.name, s.name
            """
            cursor.execute(query, municipality_ids)
        else:
            # Get all stations with municipality info
            query = """
                SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, s.station_owner,
                       s.municipality_id, m.name as municipality_name,
                       s.last_30_days_min_cm, s.last_30_days_max_cm, s.last_30_days_min_m, s.last_30_days_max_m
                FROM stations s
                LEFT JOIN municipalities m ON s.municipality_id = m.id
                ORDER BY m.name, s.name
            """
            cursor.execute(query)
        
        stations = []
        for row in cursor.fetchall():
            station_data = {
                "station_id": row["station_id"],
                "name": row["name"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "location_type": row["location_type"],
                "station_owner": row["station_owner"],
                "municipality_id": row["municipality_id"],
                "municipality_name": row["municipality_name"],
                "last_30_days_min_cm": row["last_30_days_min_cm"],
                "last_30_days_max_cm": row["last_30_days_max_cm"],
                "last_30_days_min_m": row["last_30_days_min_m"],
                "last_30_days_max_m": row["last_30_days_max_m"]
            }
            
            if include_weather:
                station_data["weather_station_info"] = get_weather_station_info()
            
            stations.append(station_data)
        
        conn.close()
        
        return jsonify({
            "success": True,
            "count": len(stations),
            "stations": stations,
            "filters": {
                "municipality_ids": municipality_ids if municipality_ids else "all",
                "include_weather": include_weather
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get stations: {str(e)}"}), 500


@municipalities_bp.route("/weather-stations", methods=["GET"])
def get_weather_stations_by_municipalities():
    """Get weather station information by municipalities."""
    try:
        municipality_ids = request.args.getlist('municipality_id')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get weather station info
        weather_info = get_weather_station_info()
        
        # Get municipality info if filtering
        municipalities = []
        if municipality_ids:
            placeholders = ','.join(['?' for _ in municipality_ids])
            cursor.execute(f"""
                SELECT id, name, region, population, area_km2, description
                FROM municipalities 
                WHERE id IN ({placeholders})
                ORDER BY name
            """, municipality_ids)
            municipalities = [dict(row) for row in cursor.fetchall()]
        else:
            cursor.execute("""
                SELECT id, name, region, population, area_km2, description
                FROM municipalities 
                ORDER BY name
            """)
            municipalities = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "weather_station": weather_info,
            "municipalities": municipalities,
            "filters": {
                "municipality_ids": municipality_ids if municipality_ids else "all"
            },
            "note": "Currently all municipalities use the same weather station (Copenhagen Meteorological Station)"
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get weather stations: {str(e)}"}), 500
