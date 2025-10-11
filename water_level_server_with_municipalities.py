#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Water Level Predictions API Server with Role-Based Authentication
Serves water level data and predictions via REST API
Includes user registration, login, and role-based access control
"""

import os
import logging
import sqlite3
import json
import bcrypt
import jwt
import secrets
import subprocess
import threading
from background_scheduler import start_background_scheduler
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from email_service import send_water_level_alert, send_subscription_confirmation
import requests
from pyproj import Transformer

def require_auth(f):
    """Decorator to require authentication for endpoints."""
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            token = auth_header.split(' ')[1]  # Bearer <token>
        except IndexError:
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.current_user = payload
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

def require_role(required_role):
    """Decorator to require specific role for endpoints."""
    def decorator(f):
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
            
            payload = verify_jwt_token(token)
            if not payload:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            user_role = payload.get('role')
            if user_role != required_role and user_role != 'superadmin':
                return jsonify({'error': f'Insufficient permissions. Required role: {required_role}'}), 403
            
            request.current_user = payload
            return f(*args, **kwargs)
        
        decorated_function.__name__ = f.__name__
        return decorated_function
    return decorator
app = Flask(__name__)
CORS(app)

# Start background scheduler for automatic updates every 5 minutes
print("\n" + "="*50)
print("🚀 STARTING BACKGROUND SCHEDULER...")
print("="*50)
print("🔧 Calling start_background_scheduler()...")
start_background_scheduler()
print("✅ Background scheduler call completed")
print("="*50)

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
JWT_EXPIRATION_HOURS = 24

# Database path
DB_PATH = "water_levels.db"

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_table():
    """Initialize the users table with roles if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    pass  # Don't drop existing table
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_default_users():
    """Create default admin and superadmin users."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if users already exist
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        # Create superadmin user
        superadmin_password_hash = hash_password('12345678')
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, created_by)
            VALUES (?, ?, ?, ?)
        ''', ('superadmin@superadmin.com', superadmin_password_hash, 'superadmin', None))
        
        superadmin_id = cursor.lastrowid
        
        # Create admin user
        admin_password_hash = hash_password('12345678')
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, created_by)
            VALUES (?, ?, ?, ?)
        ''', ('admin@admin.com', admin_password_hash, 'admin', superadmin_id))
        
        conn.commit()
        print("Default users created:")
        print("  superadmin@superadmin.com (password: 12345678)")
        print("  admin@admin.com (password: 12345678)")
    
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_jwt_token(user_id: int, email: str, role: str) -> str:
    """Generate a JWT token for the user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'role': role,
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def verify_jwt_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
def get_user_email_from_jwt():
    """Extract user email from JWT token in Authorization header."""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return 'system'
        
        token = auth_header.split(' ')[1]  # Bearer <token>
        payload = verify_jwt_token(token)
        if payload:
            return payload.get('email', 'system')
        return 'system'
    except:
        return 'system'

def get_weather_station_info():
    """Get the actual weather station information used for all water level stations."""
    weather_station = {
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
    return weather_station

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

# ===== USER AUTHENTICATION ENDPOINTS =====

@require_role("superadmin")
@app.route('/auth/register', methods=['POST'])
def register_user():
    """Register a new user (superadmin only)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        role = data.get('role', 'user').strip().lower()
        
        # Validation
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        # Validate role
        valid_roles = ['user', 'admin', 'superadmin']
        if role not in valid_roles:
            return jsonify({'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}'}), 400
        
        # Basic email validation
        if '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'User with this email already exists'}), 409
        
        # Hash password and create user
        password_hash = hash_password(password)
        creator_email = get_user_email_from_jwt()
        
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, created_by)
            VALUES (?, ?, ?, ?)
        ''', (email, password_hash, role, creator_email))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {
                'id': user_id,
                'email': email,
                'role': role
            }
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/auth/login', methods=['POST'])
def login_user():
    """Login user with email and password."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        # Find user
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, password_hash, role, is_active
            FROM users 
            WHERE email = ?
        ''', (email,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user['is_active']:
            conn.close()
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Verify password
        if not verify_password(password, user['password_hash']):
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Update last login
        cursor.execute('''
            UPDATE users 
            SET last_login = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (user['id'],))
        
        conn.commit()
        conn.close()
        
        # Generate JWT token
        token = generate_jwt_token(user['id'], user['email'], user['role'])
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'role': user['role']
            },
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/auth/verify', methods=['GET'])
def verify_token():
    """Verify if the current token is valid."""
    return jsonify({
        'valid': True,
        'user': {
            'id': 1,
            'email': "system",
            'role': "user"
        }
    }), 200

@app.route('/auth/users', methods=['GET'])
def list_users():
    """List all users (admin and superadmin only)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, role, created_at, last_login, is_active, created_by
                   
            FROM users
            
            ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row['id'],
                'email': row['email'],
                'role': row['role'],
                'created_at': row['created_at'],
                'last_login': row['last_login'],
                'is_active': bool(row['is_active']),
                'created_by': row['created_by']
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'count': len(users),
            'users': users
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to list users: {str(e)}'}), 500
@app.route('/auth/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    """Update user details (superadmin only)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, role FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        current_user = {"user_id": 1}
        update_fields = []
        update_values = []
        
        if 'email' in data:
            new_email = data['email'].strip().lower()
            if not new_email or '@' not in new_email or '.' not in new_email:
                conn.close()
                return jsonify({'error': 'Invalid email format'}), 400
            
            cursor.execute('SELECT id FROM users WHERE email = ? AND id != ?', (new_email, user_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({'error': 'Email already exists'}), 409
            
            update_fields.append('email = ?')
            update_values.append(new_email)
        
        if 'role' in data:
            new_role = data['role'].strip().lower()
            valid_roles = ['user', 'admin', 'superadmin']
            if new_role not in valid_roles:
                conn.close()
                return jsonify({'error': f'Invalid role. Must be one of: {", ".join(valid_roles)}'}), 400
            
            update_fields.append('role = ?')
            update_values.append(new_role)
        
        if 'password' in data:
            new_password = data['password']
            if not new_password or len(new_password) < 8:
                conn.close()
                return jsonify({'error': 'Password must be at least 8 characters long'}), 400
            
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            update_fields.append('password_hash = ?')
            update_values.append(password_hash)
        
        if not update_fields:
            conn.close()
            return jsonify({'error': 'No valid fields to update'}), 400
        
        update_values.append(user_id)
        
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, update_values)
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'User updated successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to update user: {str(e)}'}), 500

@app.route('/auth/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user (superadmin only)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, role FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        if user['role'] == 'superadmin':
            cursor.execute('SELECT COUNT(*) as count FROM users WHERE role = ? AND is_active = 1', ('superadmin',))
            superadmin_count = cursor.fetchone()['count']
            if superadmin_count <= 1:
                conn.close()
                return jsonify({'error': 'Cannot delete the last superadmin user'}), 400
        
        current_user = {"user_id": 1}
        if user_id == 1:
            conn.close()
            return jsonify({'error': 'Cannot delete your own account'}), 400
        
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': f'User {user["email"]} deleted successfully',
            'deleted_user': {'id': user['id'], 'email': user['email'], 'role': user['role']}
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete user: {str(e)}'}), 500

# ===== MUNICIPALITY MANAGEMENT ENDPOINTS =====

@app.route("/municipalities", methods=["GET"])
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

@app.route("/municipalities/<int:municipality_id>", methods=["GET"])
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

@require_role("superadmin")
@app.route("/municipalities", methods=["POST"])
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

@app.route("/municipalities/<int:municipality_id>", methods=["PUT"])
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

@app.route("/municipalities/<int:municipality_id>", methods=["DELETE"])
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
# ===== MUNICIPALITY STATION MANAGEMENT ENDPOINTS =====

@app.route("/municipalities/<int:municipality_id>/stations", methods=["POST"])
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
                    "created_by": creator_email,
            "municipality_name": municipality['name'],
            "assigned_stations": assigned_count,
            "station_ids": station_ids
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to assign stations: {str(e)}"}), 500

@app.route("/municipalities/stations", methods=["GET"])
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

@app.route("/municipalities/weather-stations", methods=["GET"])
def get_weather_stations_by_municipalities():
    """Get weather station information by municipalities."""
    try:
        municipality_ids = request.args.getlist('municipality_id')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get weather station info (currently we have one weather station for all)
        
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
# ===== EXISTING API ENDPOINTS (now with optional auth) =====

@app.route('/')
def index():
    """API information endpoint."""
    return jsonify({
        "name": "Water Level Predictions API with Role-Based Authentication",
        "version": "3.0",
        "description": "API for water level data and predictions with user authentication and role-based access control",
        "endpoints": {
            "POST /auth/register": "Register a new user (superadmin only)",
            "POST /auth/login": "Login user",
            "GET /auth/verify": "Verify authentication token",
            "GET /auth/users": "List all users (admin/superadmin only)",
            "GET /municipalities": "List all municipalities (public access)",
            "GET /municipalities/<id>": "Get specific municipality (public access)",
            "POST /municipalities": "Create new municipality (superadmin only)",
            "PUT /municipalities/<id>": "Update municipality (superadmin only)",
            "DELETE /municipalities/<id>": "Delete municipality (superadmin only)",
            "POST /municipalities/<id>/stations": "Assign stations to municipality (superadmin only)",
            "GET /municipalities/stations": "Get stations by municipalities (all/specific/multiple)",
            "GET /municipalities/weather-stations": "Get weather stations by municipalities",
            "GET /": "API information",
            "GET /stations": "All stations with coordinates and weather info",
            "GET /stations/<id>": "Specific station with weather info",
            "GET /stations/<id>/minmax": "Get min/max values for a station",
            "POST /stations/<id>/minmax": "Update min/max values for a station (admin/superadmin only)",
            "POST /stations/minmax/bulk": "Update min/max values for multiple stations (admin/superadmin only)",
            "GET /water-levels": "Current water levels with weather info",
            "GET /water-levels/<id>": "Station water levels with weather info",
            "GET /predictions": "All predictions",
            "GET /predictions/<id>": "Station predictions",
            "GET /past-predictions/<id>": "Historical predictions archive for a station (public access)",
            "GET /weather-station": "Weather station information"
        },
        "roles": {
            "user": "Basic access to read water level data and predictions",
            "admin": "Can read all data and update station min/max values",
            "superadmin": "Full access including user management"
        }
    })

@app.route('/weather-station')
def get_weather_station():
    """Get the actual weather station information used for all water level stations."""
    return jsonify({
        "success": True,
        "weather_station": weather_info
    })

@app.route('/stations', methods=['GET'])
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

@app.route('/stations', methods=['POST'])
@require_role('superadmin')
def create_station():
    """Create new station - superadmin required."""
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

        # Use Vandah metadata if available (optional - you can still use provided data)
        vandah_metadata = vandah_validation['metadata']
        if vandah_metadata:
            # Use Vandah data to fill all fields
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
        
        # Start background data generation (non-blocking)
        print(f"🔄 Starting background data generation for station {station_id}...")
        try:
            import subprocess
            import threading
            
            def run_data_update():
                try:
                    result = subprocess.run([
                        'python3', 'update_new_station_data.py', station_id
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"✅ Background data generation completed for station {station_id}")
                    else:
                        print(f"❌ Background data generation failed for station {station_id}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ Background data generation timed out for station {station_id}")
                except Exception as e:
                    print(f"❌ Error during background data generation for station {station_id}: {str(e)}")
            
            # Start background thread
            thread = threading.Thread(target=run_data_update)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"❌ Error starting background data generation for station {station_id}: {str(e)}")
        
        # Return immediately with Vandah metadata
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

@app.route('/stations/<station_id>', methods=['GET'])
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

@app.route('/stations/<station_id>', methods=['DELETE'])
@require_role('superadmin')
def delete_station(station_id):
    """Delete a station and all its associated data."""
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

@app.route('/water-levels')
def get_water_levels():
    """Get current water levels for all stations with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get latest water level from either water_levels or last_30_days_historical tables
    # Get all stations first
    cursor.execute("SELECT station_id, name, latitude, longitude FROM stations ORDER BY name")
    stations = cursor.fetchall()
    
    water_levels = []
    weather_info = get_weather_station_info()
    for station in stations:
        station_id = station['station_id']
        
        # Try to get latest from last_30_days_historical first (most recent data)
        cursor.execute("""
            SELECT level_cm, level_cm/100 as water_level_m, timestamp as measurement_date, 'last_30_days_historical' as source
            FROM last_30_days_historical 
            WHERE station_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (station_id,))
        
        latest = cursor.fetchone()
        
        # If no data in last_30_days_historical, try water_levels table
        if not latest:
            cursor.execute("""
                SELECT level_cm, level_cm/100 as water_level_m, timestamp as measurement_date, 'water_levels' as source
                FROM water_levels 
                WHERE station_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (station_id,))
            latest = cursor.fetchone()
        
        # If we have data, add to results
        if latest:
            water_data = {
                "station_id": station_id,
                "name": station['name'],
                "latitude": station['latitude'],
                "longitude": station['longitude'],
                "measurement_date": latest['measurement_date'],
                "water_level_cm": latest['level_cm'],
                "water_level_m": latest['water_level_m'],
                "data_source": latest['source'],
                "weather_station_info": weather_info
            }
            water_levels.append(water_data)
    
    
    # water_levels list is now built in the query section above
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(water_levels),
        "water_levels": water_levels
    })

@app.route('/water-levels/<station_id>')
def get_station_water_levels(station_id):
    """Get water level history for a specific station with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT s.name FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id WHERE s.station_id = ?", (station_id,))
    station = cursor.fetchone()
    
    if not station:
        conn.close()
        return jsonify({"success": False, "error": "Station not found"}), 404
    
    cursor.execute("""
        SELECT level_cm, level_cm/100 as water_level_m, timestamp as measurement_date
        FROM last_30_days_historical
        WHERE station_id = ?
        ORDER BY timestamp DESC
    """, (station_id,))
    
    history = []
    for row in cursor.fetchall():
        history.append({
            "date": row['measurement_date'],
            "water_level_cm": row['level_cm'],
            "water_level_m": row['water_level_m']
        })
    weather_info = get_weather_station_info()
    
    conn.close()
    
    return jsonify({
        "success": True,
        "station_id": station_id,
        "station_name": station['name'],
        "history": history,
        "weather_station_info": weather_info
    })

@app.route('/predictions')
def get_predictions():
    """Get all predictions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.station_id, s.name, s.latitude, s.longitude, p.prediction_date, p.predicted_water_level_cm, 
               p.predicted_water_level_cm/100 as predicted_water_level_m, p.change_from_last_cm, p.forecast_date
        FROM predictions p
        JOIN stations s ON p.station_id = s.station_id
        ORDER BY s.name, p.prediction_date
    """)
    
    weather_info = get_weather_station_info()
    
    predictions = []
    for row in cursor.fetchall():
        pred_data = {
            "station_id": row['station_id'],
            "name": row['name'],
            "latitude": row['latitude'],
            "longitude": row['longitude'],
            "prediction_date": row['prediction_date'],
            "predicted_water_level_cm": row['predicted_water_level_cm'],
            "predicted_water_level_m": row['predicted_water_level_cm']/100 if row['predicted_water_level_cm'] is not None else None,
            "change_from_last_cm": row['change_from_last_cm'],
            "forecast_date": row['forecast_date'],
            "weather_station_info": weather_info
        }
        predictions.append(pred_data)
    
    conn.close()
    
    forecast_date = None
    if predictions:
        forecast_date = max(p['forecast_date'] for p in predictions)
    
    return jsonify({
        "success": True,
        "forecast_date": forecast_date,
        "count": len(predictions),
        "predictions": predictions
    })

@app.route('/predictions/<station_id>')
def get_station_predictions(station_id):
    """Get predictions for a specific station with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT s.name FROM stations s LEFT JOIN municipalities m ON s.municipality_id = m.id WHERE s.station_id = ?", (station_id,))
    station = cursor.fetchone()
    
    if not station:
        conn.close()
        return jsonify({"success": False, "error": "Station not found"}), 404
    
    cursor.execute("""
        SELECT prediction_date, predicted_water_level_cm, predicted_water_level_cm/100 as predicted_water_level_m, 
               change_from_last_cm, forecast_date
        FROM predictions
        WHERE station_id = ?
        ORDER BY prediction_date
    """, (station_id,))
    
    predictions = []
    for row in cursor.fetchall():
        predictions.append({
            "prediction_date": row['prediction_date'],
            "predicted_water_level_cm": row['predicted_water_level_cm'],
            "predicted_water_level_m": row['predicted_water_level_cm']/100 if row['predicted_water_level_cm'] is not None else None,
            "change_from_last_cm": row['change_from_last_cm'],
            "forecast_date": row['forecast_date']
        })
    
    conn.close()
    
    weather_info = get_weather_station_info()
    return jsonify({
        "success": True,
        "station_id": station_id,
        "station_name": station['name'],
        "predictions": predictions,
        "weather_station_info": weather_info
    })

@app.route('/past-predictions/<station_id>')
def get_station_past_predictions(station_id):
    """Get historical predictions for a specific station (public access)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if station exists
    cursor.execute("SELECT s.name FROM stations s WHERE s.station_id = ?", (station_id,))
    station = cursor.fetchone()
    
    if not station:
        conn.close()
        return jsonify({"success": False, "error": "Station not found"}), 404
    
    # Get all past predictions for this station, grouped by forecast date
    cursor.execute("""
        SELECT prediction_date, predicted_water_level_cm, 
               predicted_water_level_cm/100 as predicted_water_level_m,
               change_from_last_cm, forecast_created_at, created_at
        FROM past_predictions
        WHERE station_id = ?
        ORDER BY forecast_created_at DESC, prediction_date ASC
    """, (station_id,))
    
    past_predictions = []
    for row in cursor.fetchall():
        past_predictions.append({
            "prediction_date": row['prediction_date'],
            "predicted_water_level_cm": row['predicted_water_level_cm'],
            "predicted_water_level_m": row['predicted_water_level_m'],
            "change_from_last_cm": row['change_from_last_cm'],
            "forecast_created_at": row['forecast_created_at'],
            "created_at": row['created_at']
        })
    
    conn.close()
    
    return jsonify({
        "success": True,
        "station_id": station_id,
        "station_name": station['name'],
        "count": len(past_predictions),
        "past_predictions": past_predictions
    })

# ===== PROTECTED ENDPOINTS (require authentication) =====

@app.route('/stations/<station_id>/minmax', methods=['GET'])
def get_station_minmax(station_id):
    """Get current min/max water level values for a specific station."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if station exists
        cursor.execute("SELECT station_id, name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        
        if not station:
            conn.close()
            return jsonify({"error": f"Station {station_id} not found"}), 404
        
        # Get min/max values from min_max_values table
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

@app.route('/stations/<station_id>/minmax', methods=['POST'])
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
        
        # Insert or update min/max values in min_max_values table
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

@app.route('/stations/minmax/bulk', methods=['POST'])
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
        return jsonify({"error": f"Failed to update bulk station min/max values: {str(e)}"}), 500

# ============================================================================
# SUBSCRIPTION ENDPOINTS
# ============================================================================

@app.route('/stations/<station_id>/subscribe', methods=['POST'])
@require_auth
def subscribe_to_station(station_id):
    """Subscribe to water level alerts for a station."""
    try:
        # Get user email from token
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        user_email = payload['email']
        
        # Get request data
        data = request.get_json() or {}
        threshold_percentage = data.get('threshold_percentage', 0.9)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        if not station:
            conn.close()
            return jsonify({"error": "Station not found"}), 404
        
        station_name = station["name"]
        
        # Insert or update subscription
        cursor.execute("""
            INSERT OR REPLACE INTO station_subscriptions 
            (user_email, station_id, threshold_percentage, is_active, updated_at)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
        """, (user_email, station_id, threshold_percentage))
        
        conn.commit()
        conn.close()
        
        # Send confirmation email
        send_subscription_confirmation(user_email, station_name, station_id)
        
        return jsonify({
            "message": "Successfully subscribed to station alerts",
            "subscription": {
                "user_email": user_email,
                "station_id": station_id,
                "station_name": station_name,
                "threshold_percentage": threshold_percentage
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to subscribe: {str(e)}"}), 500

@app.route('/stations/<station_id>/unsubscribe', methods=['POST'])
@require_auth
def unsubscribe_from_station(station_id):
    """Unsubscribe from water level alerts for a station."""
    try:
        # Get user email from token
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        user_email = payload['email']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if subscription exists
        cursor.execute("""
            SELECT s.name FROM station_subscriptions ss
            JOIN stations s ON ss.station_id = s.station_id
            WHERE ss.user_email = ? AND ss.station_id = ? AND ss.is_active = 1
        """, (user_email, station_id))
        subscription = cursor.fetchone()
        
        if not subscription:
            conn.close()
            return jsonify({"error": "No active subscription found"}), 404
        
        # Deactivate subscription
        cursor.execute("""
            UPDATE station_subscriptions 
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE user_email = ? AND station_id = ?
        """, (user_email, station_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": "Successfully unsubscribed from station alerts",
            "subscription": {
                "user_email": user_email,
                "station_id": station_id,
                "station_name": subscription["name"]
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to unsubscribe: {str(e)}"}), 500

@app.route('/subscriptions', methods=['GET'])
@require_auth
def get_user_subscriptions():
    """Get all active subscriptions for the current user."""
    try:
        # Get user email from token
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        user_email = payload['email']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all active subscriptions
        cursor.execute("""
            SELECT ss.station_id, s.name as station_name, ss.threshold_percentage, 
                   ss.created_at, ss.updated_at
            FROM station_subscriptions ss
            JOIN stations s ON ss.station_id = s.station_id
            WHERE ss.user_email = ? AND ss.is_active = 1
            ORDER BY ss.created_at DESC
        """, (user_email,))
        
        subscriptions = cursor.fetchall()
        conn.close()
        
        return jsonify({
            "subscriptions": [
                {
                    "station_id": sub["station_id"],
                    "station_name": sub["station_name"],
                    "threshold_percentage": sub["threshold_percentage"],
                    "created_at": sub["created_at"],
                    "updated_at": sub["updated_at"]
                }
                for sub in subscriptions
            ]
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get subscriptions: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize user table and create default users
    init_user_table()
    create_default_users()
    
    print("Starting Water Level Predictions API Server with Role-Based Authentication...")
    print("Available endpoints:")
    print("  POST /auth/register - Register a new user (superadmin only)")
    print("  POST /auth/login - Login user")
    print("  GET /auth/verify - Verify authentication token")
    print("  GET /auth/users - List all users (admin/superadmin only)")
    print("  GET /municipalities - List all municipalities (public access)")
    print("  GET /municipalities/<id> - Get specific municipality (public access)")
    print("  POST /municipalities - Create new municipality (superadmin only)")
    print("  PUT /municipalities/<id> - Update municipality (superadmin only)")
    print("  DELETE /municipalities/<id> - Delete municipality (superadmin only)")
    print("  POST /municipalities/<id>/stations - Assign stations to municipality (superadmin only)")
    print("  GET /municipalities/stations - Get stations by municipalities (all/specific/multiple)")
    print("  GET /municipalities/weather-stations - Get weather stations by municipalities")
    print("  GET / - API information")
    print("  GET /stations - All stations with coordinates and weather info")
    print("  GET /stations/<id> - Specific station with weather info")
    print("  GET /stations/<id>/minmax - Get min/max values for a station")
    print("  POST /stations/<id>/minmax - Update min/max values for a station (admin/superadmin only)")
    print("  POST /stations/minmax/bulk - Update min/max values for multiple stations (admin/superadmin only)")
    print("  GET /water-levels - Current water levels with weather info")
    print("  GET /water-levels/<id> - Station water levels with weather info")
    print("  GET /predictions - All predictions with weather info")
    print("  GET /predictions/<id> - Station predictions with weather info")
    print("  GET /weather-station - Weather station information")
    print("\nDefault users created:")
    print("  superadmin@superadmin.com (password: 12345678)")
    print("  admin@admin.com (password: 12345678)")
    # Check environment variables for debug mode
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"🔧 Debug mode: {'ENABLED' if debug_mode else 'DISABLED'}")
    print(f"�� Starting server on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=debug_mode)