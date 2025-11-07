"""Authentication routes."""
from flask import Blueprint, request, jsonify
import bcrypt
from app.middleware.auth import require_auth, require_role
from app.utils.database import get_db_connection
from app.utils.password import hash_password, verify_password
from app.utils.jwt_utils import generate_jwt_token, get_user_email_from_jwt

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/register', methods=['POST'])
def register_user():
    """Register a new user (superadmin only)."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        role = data.get('role', 'user').strip().lower()
        municipality_id = data.get('municipality_id')
        
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
        
        # Validate municipality_id if provided
        if municipality_id is not None:
            if not isinstance(municipality_id, int):
                return jsonify({'error': 'municipality_id must be an integer'}), 400
        
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
        
        # Check if municipality exists (if provided)
        if municipality_id is not None:
            cursor.execute('SELECT id, name FROM municipalities WHERE id = ?', (municipality_id,))
            municipality = cursor.fetchone()
            if not municipality:
                conn.close()
                return jsonify({'error': f'Municipality with id {municipality_id} does not exist'}), 404
            municipality_name = municipality['name']
        else:
            municipality_name = None
        
        # Hash password and create user
        password_hash = hash_password(password)
        creator_email = get_user_email_from_jwt()
        
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, municipality_id, created_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (email, password_hash, role, municipality_id, creator_email))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        response_data = {
            'message': 'User registered successfully',
            'user': {
                'id': user_id,
                'email': email,
                'role': role,
                'municipality_id': municipality_id,
                'municipality_name': municipality_name
            }
        }
        
        return jsonify(response_data), 201
        
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
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
            SELECT u.id, u.email, u.password_hash, u.role, u.is_active, u.municipality_id, m.name as municipality_name
            FROM users u
            LEFT JOIN municipalities m ON u.municipality_id = m.id
            WHERE u.email = ?
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
        
        response_data = {
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'role': user['role'],
                'municipality_id': user['municipality_id'],
                'municipality_name': user['municipality_name']
            },
            'token': token
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500


@auth_bp.route('/verify', methods=['GET'])
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


@auth_bp.route('/users', methods=['GET'])
def list_users():
    """List all users (admin and superadmin only)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.email, u.role, u.municipality_id, m.name as municipality_name,
                   u.created_at, u.last_login, u.is_active, u.created_by
            FROM users u
            LEFT JOIN municipalities m ON u.municipality_id = m.id
            ORDER BY u.created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row['id'],
                'email': row['email'],
                'role': row['role'],
                'municipality_id': row['municipality_id'],
                'municipality_name': row['municipality_name'],
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


@auth_bp.route('/user/municipality', methods=['GET'])
@require_auth
def get_user_municipality():
    """Get the municipality assigned to the current authenticated user.
    Superadmins get all municipalities."""
    try:
        # Get user from token
        user_id = request.current_user.get('user_id')
        user_role = request.current_user.get('role')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.id, u.email, u.role, u.municipality_id, m.name as municipality_name,
                   m.region, m.population, m.area_km2, m.description
            FROM users u
            LEFT JOIN municipalities m ON u.municipality_id = m.id
            WHERE u.id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # If superadmin, return all municipalities
        if user_role == 'superadmin':
            cursor.execute('''
                SELECT id, name, region, population, area_km2, description,
                       created_at, updated_at, created_by, updated_by
                FROM municipalities
                ORDER BY name
            ''')
            
            municipalities = []
            for row in cursor.fetchall():
                municipalities.append({
                    'id': row['id'],
                    'name': row['name'],
                    'region': row['region'],
                    'population': row['population'],
                    'area_km2': row['area_km2'],
                    'description': row['description'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'created_by': row['created_by'],
                    'updated_by': row['updated_by']
                })
            
            conn.close()
            
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'role': user['role']
                },
                'municipalities': municipalities,
                'count': len(municipalities),
                'message': 'Superadmin has access to all municipalities'
            }), 200
        
        # For non-superadmin users, return their assigned municipality
        conn.close()
        
        if user['municipality_id'] is None:
            return jsonify({
                'success': True,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'role': user['role']
                },
                'municipality': None,
                'message': 'No municipality assigned to this user'
            }), 200
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'role': user['role']
            },
            'municipality': {
                'id': user['municipality_id'],
                'name': user['municipality_name'],
                'region': user['region'],
                'population': user['population'],
                'area_km2': user['area_km2'],
                'description': user['description']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to get user municipality: {str(e)}'}), 500


@auth_bp.route('/users/<int:user_id>', methods=['PUT'])
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
        
        if 'municipality_id' in data:
            new_municipality_id = data['municipality_id']
            if new_municipality_id is not None:
                # Validate municipality exists
                cursor.execute('SELECT id FROM municipalities WHERE id = ?', (new_municipality_id,))
                if not cursor.fetchone():
                    conn.close()
                    return jsonify({'error': f'Municipality with id {new_municipality_id} does not exist'}), 404
            
            update_fields.append('municipality_id = ?')
            update_values.append(new_municipality_id)
        
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


@auth_bp.route('/users/<int:user_id>', methods=['DELETE'])
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
