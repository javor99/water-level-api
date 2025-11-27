# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Authentication middleware decorators."""
from functools import wraps
from flask import request, jsonify
from app.utils.jwt_utils import verify_jwt_token
import logging

logger = logging.getLogger(__name__)


def require_auth(f):
    """Decorator to require authentication for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        endpoint = request.endpoint or 'unknown'
        method = request.method
        
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logger.warning(f"🔒 {method} {request.path} - FAILED: No Authorization header")
            return jsonify({'error': 'Authorization header required'}), 401
        
        try:
            token = auth_header.split(' ')[1]  # Bearer <token>
        except IndexError:
            logger.warning(f"🔒 {method} {request.path} - FAILED: Invalid Authorization header format")
            return jsonify({'error': 'Invalid authorization header format'}), 401
        
        # Log token info (first 20 and last 10 chars for security)
        token_preview = f"{token[:20]}...{token[-10:]}" if len(token) > 30 else token
        logger.info(f"🔑 {method} {request.path} - Token: {token_preview}")
        
        payload = verify_jwt_token(token)
        if not payload:
            logger.warning(f"🔒 {method} {request.path} - FAILED: Invalid or expired token")
            logger.debug(f"   Token (full): {token}")
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Log successful authentication
        user_email = payload.get('email', 'unknown')
        user_role = payload.get('role', 'unknown')
        logger.info(f"✅ {method} {request.path} - Authenticated as: {user_email} ({user_role})")
        
        request.current_user = payload
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(required_role):
    """Decorator to require specific role for endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            method = request.method
            
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                logger.warning(f"🔒 {method} {request.path} - FAILED: No Authorization header (requires role: {required_role})")
                return jsonify({'error': 'Authorization header required'}), 401
            
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                logger.warning(f"🔒 {method} {request.path} - FAILED: Invalid Authorization header format")
                return jsonify({'error': 'Invalid authorization header format'}), 401
            
            # Log token info
            token_preview = f"{token[:20]}...{token[-10:]}" if len(token) > 30 else token
            logger.info(f"🔑 {method} {request.path} - Token: {token_preview} (requires role: {required_role})")
            
            payload = verify_jwt_token(token)
            if not payload:
                logger.warning(f"🔒 {method} {request.path} - FAILED: Invalid or expired token")
                logger.debug(f"   Token (full): {token}")
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            user_role = payload.get('role')
            user_email = payload.get('email', 'unknown')
            
            if user_role != required_role and user_role != 'superadmin':
                logger.warning(f"🔒 {method} {request.path} - FAILED: Insufficient permissions. User {user_email} has role '{user_role}', requires '{required_role}'")
                return jsonify({'error': f'Insufficient permissions. Required role: {required_role}'}), 403
            
            logger.info(f"✅ {method} {request.path} - Authenticated as: {user_email} ({user_role}) - Role check passed")
            
            request.current_user = payload
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

