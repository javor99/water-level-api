#!/usr/bin/env python3
import sqlite3
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect("water_levels.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/auth/users', methods=['GET'])
def list_users():
    """List all users - NO AUTH REQUIRED"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, role, created_at, is_active FROM users ORDER BY created_at DESC')
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row['id'],
                'email': row['email'],
                'role': row['role'],
                'created_at': row['created_at'],
                'is_active': bool(row['is_active'])
            })
        conn.close()
        return jsonify({'success': True, 'count': len(users), 'users': users}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to list users: {str(e)}'}), 500

@app.route('/auth/register', methods=['POST'])
def register_user():
    """Register new user - NO AUTH REQUIRED"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        role = data.get('role', 'user').strip().lower()
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'User already exists'}), 409
        
        # Create user
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
        ''', (email, password, role))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'User registered successfully',
            'user': {'id': user_id, 'email': email, 'role': role}
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/auth/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user - NO AUTH REQUIRED"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, email, role FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': f'User {user["email"]} deleted successfully',
            'deleted_user': {'id': user['id'], 'email': user['email'], 'role': user['role']}
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to delete user: {str(e)}'}), 500

@app.route('/municipalities', methods=['GET'])
def list_municipalities():
    """List municipalities - NO AUTH REQUIRED"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM municipalities ORDER BY name')
        municipalities = []
        for row in cursor.fetchall():
            municipalities.append(dict(row))
        conn.close()
        return jsonify(municipalities), 200
    except Exception as e:
        return jsonify({'error': f'Failed to list municipalities: {str(e)}'}), 500

@app.route('/municipalities/stations', methods=['GET'])
def get_stations_by_municipalities():
    """Get stations by municipalities - NO AUTH REQUIRED"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.station_id, s.name, s.latitude, s.longitude, s.municipality_id, m.name as municipality_name
            FROM stations s
            LEFT JOIN municipalities m ON s.municipality_id = m.id
            ORDER BY m.name, s.name
        ''')
        stations = []
        for row in cursor.fetchall():
            stations.append(dict(row))
        conn.close()
        return jsonify({'count': len(stations), 'stations': stations}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get stations: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def api_info():
    """API information"""
    return jsonify({
        'message': 'Water Level API - NO AUTHENTICATION REQUIRED',
        'endpoints': [
            'GET /auth/users - List all users',
            'POST /auth/register - Register new user', 
            'DELETE /auth/users/<id> - Delete user',
            'GET /municipalities - List municipalities',
            'GET /municipalities/stations - Get stations by municipalities'
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Server - NO AUTHENTICATION REQUIRED")
    print("Server will be available at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
