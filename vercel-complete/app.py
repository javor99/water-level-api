from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sqlite3
import jwt
from functools import wraps
import requests
import pandas as pd
from datetime import datetime, timedelta
import threading
import time

# Import your existing modules
try:
    from email_service import send_water_level_alert, send_subscription_confirmation
except ImportError:
    # Fallback if email service not available
    def send_water_level_alert(*args, **kwargs):
        return False
    def send_subscription_confirmation(*args, **kwargs):
        return False

app = Flask(__name__)
CORS(app)

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
DATABASE_URL = os.environ.get('DATABASE_URL', 'water_levels.db')

# Database connection
def get_db_connection():
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    return conn

# Authentication decorators
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid authorization header"}), 401
        
        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_role(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user'):
                return jsonify({"error": "Authentication required"}), 401
            
            user_role = request.user.get('role')
            if user_role != required_role:
                return jsonify({"error": f"Access denied. Required role: {required_role}"}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Water Level API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "stations": "/stations",
            "predictions": "/predictions",
            "water_levels": "/water-levels",
            "auth": "/auth/login"
        }
    })

@app.route('/stations', methods=['GET'])
def get_stations():
    """List all stations - no authentication required."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.station_id, s.name, s.latitude, s.longitude, s.location_type, 
                   s.station_owner, s.municipality_id, m.name as municipality_name
            FROM stations s 
            LEFT JOIN municipalities m ON s.municipality_id = m.id
            ORDER BY s.name
        """)
        
        stations = []
        for row in cursor.fetchall():
            station_data = {
                "station_id": row['station_id'],
                "name": row['name'],
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "location_type": row['location_type'],
                "station_owner": row["station_owner"],
                "municipality_id": row["municipality_id"],
                "municipality_name": row["municipality_name"]
            }
            stations.append(station_data)
        
        conn.close()
        return jsonify({
            "success": True,
            "count": len(stations),
            "stations": stations
        })
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stations: {str(e)}"}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    """Login endpoint for testing."""
    data = request.get_json()
    email = data.get('email', '')
    password = data.get('password', '')
    
    # Simple test credentials
    if email == 'superadmin@superadmin.com' and password == '12345678':
        token = jwt.encode({
            'user_id': 1,
            'email': email,
            'role': 'superadmin',
            'iat': datetime.utcnow()
        }, SECRET_KEY, algorithm='HS256')
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": 1,
                "email": email,
                "role": "superadmin"
            }
        })
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if test_db_connection() else "disconnected"
    })

def test_db_connection():
    """Test database connection."""
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except:
        return False

if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5001, debug=False)
