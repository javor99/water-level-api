# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Flask application factory."""
import logging
from flask import Flask
from flask_cors import CORS
from app.models.user import init_user_table, create_default_users


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('server.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize database tables and users
    with app.app_context():
        init_user_table()
        create_default_users()
    
    # NOTE: Background scheduler should be run as a SEPARATE PROCESS
    # when using gunicorn with multiple workers to avoid duplicate tasks.
    # Use: python3 background_scheduler.py
    # DO NOT start it here, or each worker will run its own scheduler!
    
    # Register blueprints
    from app.routes import auth_bp, stations_bp, municipalities_bp, predictions_bp, subscriptions_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(stations_bp)
    app.register_blueprint(municipalities_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(subscriptions_bp)
    
    # Home route
    @app.route('/')
    def index():
        """API Information endpoint."""
        return {
            "message": "Water Level Predictions API Server",
            "version": "2.0.0",
            "status": "operational",
            "description": "Real-time water level monitoring and predictions with ML-based forecasting",
            "endpoints": {
                "authentication": {
                    "POST /auth/register": "Register new user (superadmin only)",
                    "POST /auth/login": "Login and get JWT token",
                    "GET /auth/verify": "Verify JWT token",
                    "GET /auth/users": "List all users (admin/superadmin)",
                    "GET /auth/user/municipality": "Get authenticated user's municipality",
                    "PUT /auth/users/<id>": "Update user (superadmin)",
                    "DELETE /auth/users/<id>": "Delete user (superadmin)"
                },
                "municipalities": {
                    "GET /municipalities": "List all municipalities",
                    "GET /municipalities/<id>": "Get specific municipality",
                    "POST /municipalities": "Create municipality (superadmin)",
                    "PUT /municipalities/<id>": "Update municipality (superadmin)",
                    "DELETE /municipalities/<id>": "Delete municipality (superadmin)",
                    "POST /municipalities/<id>/stations": "Assign stations (superadmin)",
                    "GET /municipalities/stations": "Get stations by municipality",
                    "GET /municipalities/weather-stations": "Get weather stations"
                },
                "stations": {
                    "GET /stations": "List all stations",
                    "GET /stations/<id>": "Get specific station",
                    "POST /stations": "Create station (admin)",
                    "DELETE /stations/<id>": "Delete station (admin)",
                    "GET /stations/<id>/minmax": "Get min/max values",
                    "POST /stations/<id>/minmax": "Update min/max (admin)",
                    "POST /stations/minmax/bulk": "Bulk update min/max (admin)",
                    "GET /weather-station": "Get weather station info"
                },
                "water_levels": {
                    "GET /water-levels": "Get all current water levels",
                    "GET /water-levels/<id>": "Get station water levels"
                },
                "predictions": {
                    "GET /predictions": "Get all predictions",
                    "GET /predictions/<id>": "Get station predictions",
                    "GET /past-predictions/<id>": "Get past predictions"
                },
                "subscriptions": {
                    "POST /stations/<id>/subscribe": "Subscribe to station alerts (authenticated)",
                    "POST /stations/<id>/unsubscribe": "Unsubscribe from alerts (authenticated)",
                    "GET /subscriptions": "Get user subscriptions (authenticated)"
                }
            },
            "authentication": "Bearer token required for protected endpoints",
            "roles": ["user", "admin", "superadmin"]
        }
    
    return app



