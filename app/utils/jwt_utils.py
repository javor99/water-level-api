"""JWT token utilities."""
import jwt
from datetime import datetime, timedelta
from flask import request
from app.config import SECRET_KEY, JWT_EXPIRATION_HOURS


def generate_jwt_token(user_id: int, email: str, role: str) -> str:
    """Generate a JWT token for the user."""
    now = datetime.utcnow()
    payload = {
        'user_id': user_id,
        'email': email,
        'role': role,
        'iat': now,
        'exp': now + timedelta(hours=JWT_EXPIRATION_HOURS)  # Token expires after 24 hours
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

