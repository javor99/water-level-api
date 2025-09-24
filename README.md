# Water Level Prediction API

A Flask-based API for water level monitoring and prediction with Vandah system integration.

## Features

- Water level data fetching from Vandah API
- Station management with validation
- User authentication and role-based access
- Background data updates
- Prediction generation
- Municipality management

## API Endpoints

- `POST /stations` - Create new station (with Vandah validation)
- `GET /stations` - List all stations
- `GET /water-levels` - Current water levels
- `GET /predictions` - Water level predictions
- Authentication endpoints for user management

## Installation

```bash
pip install -r requirements.txt
python water_level_server_with_municipalities.py
```

## Environment Variables

Set these environment variables for production:
- `SMTP_USERNAME` - Email service username
- `SMTP_PASSWORD` - Email service password
- `JWT_SECRET_KEY` - Secret key for JWT tokens
