# Water Level Subscription API Guide

## Overview
The Water Level API now includes a subscription system that allows users to receive email alerts when water level predictions exceed their specified thresholds.

## Authentication
All subscription endpoints require authentication. Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### 1. Subscribe to Station Alerts
**POST** `/stations/<station_id>/subscribe`

Subscribe to receive email alerts for a specific station when predictions exceed your threshold.

**Request Body:**
```json
{
  "threshold_percentage": 0.9
}
```

**Parameters:**
- `threshold_percentage` (optional): Alert threshold as decimal (0.9 = 90% of max historical level). Default: 0.9

**Example:**
```bash
curl -X POST http://localhost:5001/stations/70000865/subscribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"threshold_percentage": 0.8}'
```

**Response:**
```json
{
  "message": "Successfully subscribed to station alerts",
  "subscription": {
    "user_email": "user@example.com",
    "station_id": "70000865",
    "station_name": "Sengeløse å, Sengeløse mose",
    "threshold_percentage": 0.8
  }
}
```

### 2. Unsubscribe from Station Alerts
**POST** `/stations/<station_id>/unsubscribe`

Unsubscribe from email alerts for a specific station.

**Example:**
```bash
curl -X POST http://localhost:5001/stations/70000865/unsubscribe \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "message": "Successfully unsubscribed from station alerts",
  "subscription": {
    "user_email": "user@example.com",
    "station_id": "70000865",
    "station_name": "Sengeløse å, Sengeløse mose"
  }
}
```

### 3. Get User Subscriptions
**GET** `/subscriptions`

Get all active subscriptions for the current user.

**Example:**
```bash
curl -X GET http://localhost:5001/subscriptions \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "subscriptions": [
    {
      "station_id": "70000865",
      "station_name": "Sengeløse å, Sengeløse mose",
      "threshold_percentage": 0.8,
      "created_at": "2025-09-22 11:44:56",
      "updated_at": "2025-09-22 11:44:56"
    }
  ]
}
```

## How the Alert System Works

### 1. **Automatic Monitoring**
- The background scheduler runs every 5 minutes
- After updating predictions for each station, it checks for alerts
- If a prediction exceeds the user's threshold, an email alert is sent

### 2. **Alert Conditions**
- **Current Prediction** ≥ **Threshold Level**
- **Threshold Level** = **Max Historical Level** × **Threshold Percentage**
- Example: If max level is 10m and threshold is 80%, alerts trigger when prediction ≥ 8m

### 3. **Email Notifications**
When an alert is triggered, users receive an email containing:
- Station name and ID
- Current predicted water level
- Maximum historical level
- Alert threshold percentage
- Timestamp of the alert

## Complete Workflow Example

### Step 1: Register/Login
```bash
# Register a new user
curl -X POST http://localhost:5001/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123", "role": "user"}'

# Login to get token
curl -X POST http://localhost:5001/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### Step 2: Subscribe to Station
```bash
# Subscribe to station 70000865 with 80% threshold
curl -X POST http://localhost:5001/stations/70000865/subscribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"threshold_percentage": 0.8}'
```

### Step 3: Check Subscriptions
```bash
# View all your subscriptions
curl -X GET http://localhost:5001/subscriptions \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Step 4: Receive Alerts
- The system automatically monitors predictions every 5 minutes
- When predictions exceed your threshold, you'll receive an email alert
- Alerts are logged to the server console

### Step 5: Unsubscribe (Optional)
```bash
# Unsubscribe from a station
curl -X POST http://localhost:5001/stations/70000865/unsubscribe \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Available Stations

Some example station IDs you can subscribe to:
- `70000865`: Sengeløse å, Sengeløse mose
- `70000925`: Spangå, Ågesholmvej
- `70000926`: Nybølle Å, Ledøje Plantage
- `70000927`: Hakkemosegrøften, Ole Rømers Vej
- `70000923`: Enghave Å, Rolandsvej 3

## Email Configuration

To enable email notifications, set these environment variables:
```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export FROM_EMAIL="your-email@gmail.com"
export FROM_NAME="Water Level Alert System"
```

## Error Handling

### Common Error Responses:

**401 Unauthorized:**
```json
{"error": "Authorization header required"}
```

**404 Not Found:**
```json
{"error": "Station not found"}
```

**404 No Subscription:**
```json
{"error": "No active subscription found"}
```

**500 Server Error:**
```json
{"error": "Failed to subscribe: <error details>"}
```

## Testing the System

### Manual Alert Test
You can manually trigger an alert check for testing:
```python
from background_scheduler import check_and_send_alerts_for_station
result = check_and_send_alerts_for_station('70000865', 'Station Name')
print(f'Alert sent: {result}')
```

### Server Logs
Monitor the server console for alert activity:
```
     ALERT: Station Name prediction (11.44m) exceeds threshold (80% = 8.0m)
    📧 Alert email sent to user@example.com
    ✅ 1 alert(s) sent for Station Name
```

## Best Practices

1. **Threshold Settings:**
   - 0.9 (90%): High alert threshold, fewer notifications
   - 0.8 (80%): Medium alert threshold
   - 0.7 (70%): Low alert threshold, more notifications

2. **Subscription Management:**
   - Regularly check your subscriptions with GET `/subscriptions`
   - Unsubscribe from stations you no longer need to monitor
   - Use different thresholds for different stations based on their risk levels

3. **Email Management:**
   - Ensure your email service is properly configured
   - Check spam folders for alert emails
   - Use a dedicated email address for alerts

## Integration Examples

### JavaScript/Frontend
```javascript
// Subscribe to station
const subscribeToStation = async (stationId, threshold = 0.9) => {
  const response = await fetch(`/stations/${stationId}/subscribe`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ threshold_percentage: threshold })
  });
  return response.json();
};

// Get user subscriptions
const getSubscriptions = async () => {
  const response = await fetch('/subscriptions', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  return response.json();
};
```

### Python
```python
import requests

# Subscribe to station
def subscribe_to_station(token, station_id, threshold=0.9):
    response = requests.post(
        f'http://localhost:5001/stations/{station_id}/subscribe',
        headers={'Authorization': f'Bearer {token}'},
        json={'threshold_percentage': threshold}
    )
    return response.json()

# Get subscriptions
def get_subscriptions(token):
    response = requests.get(
        'http://localhost:5001/subscriptions',
        headers={'Authorization': f'Bearer {token}'}
    )
    return response.json()
```

This subscription system provides a complete solution for monitoring water levels and receiving timely alerts when conditions exceed user-defined thresholds.
