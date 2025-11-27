# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2025 AquaMonitor/InnoTech-TaskForce
# Part of AquaMonitor/InnoTech-TaskForce. See LICENSE for license terms.

"""Prediction and water level routes."""
from flask import Blueprint, jsonify
from app.utils.database import get_db_connection
from app.models.station import get_weather_station_info

predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.route('/water-levels')
def get_water_levels():
    """Get current water levels for all stations with weather info."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all stations first
    cursor.execute("SELECT station_id, name, latitude, longitude FROM stations ORDER BY name")
    stations = cursor.fetchall()
    
    water_levels = []
    weather_info = get_weather_station_info()
    for station in stations:
        station_id = station['station_id']
        
        # Try to get latest from last_30_days_historical first
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
    
    conn.close()
    return jsonify({
        "success": True,
        "count": len(water_levels),
        "water_levels": water_levels
    })


@predictions_bp.route('/water-levels/<station_id>')
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


@predictions_bp.route('/predictions')
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


@predictions_bp.route('/predictions/<station_id>')
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


@predictions_bp.route('/past-predictions/<station_id>')
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
    
    # Get all past predictions for this station
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
        "past_predictions_count": len(past_predictions),
        "past_predictions": past_predictions
    })
