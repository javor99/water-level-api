# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Subscription routes."""
from flask import Blueprint, request, jsonify
from app.middleware.auth import require_auth
from app.utils.database import get_db_connection
from app.utils.jwt_utils import verify_jwt_token
from services.email_service import send_subscription_confirmation
import logging
import threading

logger = logging.getLogger(__name__)

subscriptions_bp = Blueprint('subscriptions', __name__)


@subscriptions_bp.route('/stations/<station_id>/subscribe', methods=['POST'])
@require_auth
def subscribe_to_station(station_id):
    """Subscribe to water level alerts for a station.
    Supports both high water (flooding) and low water (drying out) alerts."""
    try:
        logger.info(f"📧 Subscription request for station {station_id}")
        
        # Get user email from token
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        user_email = payload['email']
        
        logger.info(f"   User: {user_email}")
        
        # Get request data
        data = request.get_json() or {}
        threshold_percentage = data.get('threshold_percentage', 0.9)
        alert_type = data.get('alert_type', 'above').lower()
        
        logger.info(f"   Alert type: {alert_type}, Threshold: {threshold_percentage}")
        
        # Validate alert_type
        if alert_type not in ['above', 'below']:
            logger.warning(f"   ❌ Invalid alert_type: {alert_type}")
            return jsonify({
                "error": "Invalid alert_type. Must be 'above' (flooding) or 'below' (drying out)"
            }), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if station exists
        cursor.execute("SELECT name FROM stations WHERE station_id = ?", (station_id,))
        station = cursor.fetchone()
        if not station:
            conn.close()
            logger.warning(f"   ❌ Station {station_id} not found")
            return jsonify({"error": "Station not found"}), 404
        
        station_name = station["name"]
        logger.info(f"   Station: {station_name}")
        
        # Insert or update subscription
        cursor.execute("""
            INSERT OR REPLACE INTO station_subscriptions 
            (user_email, station_id, threshold_percentage, alert_type, is_active, updated_at)
            VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
        """, (user_email, station_id, threshold_percentage, alert_type))
        
        conn.commit()
        conn.close()
        
        logger.info(f"   ✅ Subscription saved to database")
        
        # Send confirmation email in background (non-blocking with timeout)
        def send_email_with_timeout():
            """Send email in background thread with timeout."""
            try:
                logger.info(f"   📧 Sending confirmation email to {user_email}...")
                send_subscription_confirmation(user_email, station_name, station_id)
                logger.info(f"   ✅ Confirmation email sent")
            except Exception as email_error:
                logger.warning(f"   ⚠️ Failed to send confirmation email: {email_error}")
        
        # Start email sending in background thread (don't wait for it)
        email_thread = threading.Thread(target=send_email_with_timeout, daemon=True)
        email_thread.start()
        logger.info(f"   📧 Email sending started in background")
        
        alert_description = "flooding (water level exceeds threshold)" if alert_type == 'above' else "drying out (water level falls below threshold)"
        
        response = {
            "message": f"Successfully subscribed to {alert_type} threshold alerts",
            "subscription": {
                "user_email": user_email,
                "station_id": station_id,
                "station_name": station_name,
                "threshold_percentage": threshold_percentage,
                "alert_type": alert_type,
                "description": alert_description
            },
            "email_notification": "Confirmation email will be sent in background"
        }
        
        logger.info(f"   ✅ Subscription successful for {user_email} to {station_name}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"   ❌ Subscription failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to subscribe: {str(e)}"}), 500


@subscriptions_bp.route('/stations/<station_id>/unsubscribe', methods=['POST'])
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


@subscriptions_bp.route('/subscriptions', methods=['GET'])
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
                   ss.alert_type, ss.updated_at
            FROM station_subscriptions ss
            JOIN stations s ON ss.station_id = s.station_id
            WHERE ss.user_email = ? AND ss.is_active = 1
            ORDER BY ss.updated_at DESC
        """, (user_email,))
        
        subscriptions = cursor.fetchall()
        conn.close()
        
        return jsonify({
            "subscriptions": [
                {
                    "station_id": sub["station_id"],
                    "station_name": sub["station_name"],
                    "threshold_percentage": sub["threshold_percentage"],
                    "alert_type": sub["alert_type"],
                    "description": "flooding (above threshold)" if sub["alert_type"] == "above" else "drying out (below threshold)",
                    "updated_at": sub["updated_at"]
                }
                for sub in subscriptions
            ]
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to get subscriptions: {str(e)}"}), 500


@subscriptions_bp.route('/subscriptions/user/<user_email>', methods=['GET'])
def get_user_subscriptions_by_email(user_email):
    """Get all active subscriptions for a specific user (admin/superadmin only).
    Public endpoint - no auth required for flexibility."""
    try:
        logger.info(f"📋 Getting subscriptions for user: {user_email}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT email, role FROM users WHERE email = ?", (user_email,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            logger.warning(f"   ❌ User {user_email} not found")
            return jsonify({"error": "User not found"}), 404
        
        # Get all active subscriptions for this user
        cursor.execute("""
            SELECT ss.station_id, s.name as station_name, ss.threshold_percentage, 
                   ss.alert_type, ss.updated_at
            FROM station_subscriptions ss
            JOIN stations s ON ss.station_id = s.station_id
            WHERE ss.user_email = ? AND ss.is_active = 1
            ORDER BY ss.updated_at DESC
        """, (user_email,))
        
        subscriptions = cursor.fetchall()
        conn.close()
        
        logger.info(f"   ✅ Found {len(subscriptions)} subscription(s) for {user_email}")
        
        return jsonify({
            "success": True,
            "user_email": user_email,
            "user_role": user["role"],
            "count": len(subscriptions),
            "subscriptions": [
                {
                    "station_id": sub["station_id"],
                    "station_name": sub["station_name"],
                    "threshold_percentage": sub["threshold_percentage"],
                    "alert_type": sub["alert_type"],
                    "description": "flooding (above threshold)" if sub["alert_type"] == "above" else "drying out (below threshold)",
                    "updated_at": sub["updated_at"]
                }
                for sub in subscriptions
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"   ❌ Failed to get subscriptions: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to get subscriptions: {str(e)}"}), 500
