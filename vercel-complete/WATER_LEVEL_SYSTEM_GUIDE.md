# 🌊 WATER LEVEL SYSTEM - COMPLETE GUIDE

## 🎯 **System Overview**

This is a comprehensive water level monitoring and prediction system with the following components:

### **Core Programs:**
1. **Server Startup** - `start_server.py`
2. **Predictions Fetcher** - `fetch_predictions.py`
3. **Historical Min/Max Fetcher** - `fetch_historical_minmax.py`
4. **30-Day Historical Fetcher** - `fetch_historical_30days.py`
5. **New Station Fetcher** - `fetch_new_station.py`

---

## 🚀 **1. SERVER STARTUP**

### **File:** `start_server.py`
### **Purpose:** Start the Flask API server
### **Usage:**
```bash
python3 start_server.py
```

### **Features:**
- ✅ Checks for required dependencies
- ✅ Starts Flask server on port 5001
- ✅ Provides real-time server output
- ✅ Graceful shutdown with Ctrl+C

---

## 🔮 **2. PREDICTIONS FETCHER**

### **File:** `fetch_predictions.py`
### **Purpose:** Fetch today's predictions for all water level stations
### **Usage:**
```bash
python3 fetch_predictions.py
```

### **Features:**
- ✅ Processes all stations in all municipalities
- ✅ Generates 24-hour predictions (8 predictions, every 3 hours)
- ✅ Uses historical data for trend analysis
- ✅ Saves to `predictions` table
- ✅ Handles missing data gracefully

### **Database Table:** `predictions`
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    predicted_level_cm REAL NOT NULL,
    predicted_level_m REAL NOT NULL,
    confidence REAL,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station_id, prediction_time)
);
```

---

## 📊 **3. HISTORICAL MIN/MAX FETCHER**

### **File:** `fetch_historical_minmax.py`
### **Purpose:** Fetch 5-year historical min/max values for all stations
### **Usage:**
```bash
python3 fetch_historical_minmax.py
```

### **Features:**
- ✅ Fetches 5 years of historical data from external API
- ✅ Calculates true min/max values
- ✅ Updates `stations` table with min/max values
- ✅ Handles API rate limiting
- ✅ Provides detailed progress reporting

### **API Source:** `https://vandah.miljoeportal.dk/api/waterlevels/{station_id}`

---

## 📅 **4. 30-DAY HISTORICAL FETCHER**

### **File:** `fetch_historical_30days.py`
### **Purpose:** Fetch last 30 days of water level data for all stations
### **Usage:**
```bash
python3 fetch_historical_30days.py
```

### **Features:**
- ✅ Fetches 30 days of historical data
- ✅ Updates `last_30_days_historical` table
- ✅ Replaces existing data for each station
- ✅ Handles data validation and cleaning
- ✅ Provides progress tracking

### **Database Table:** `last_30_days_historical`
```sql
CREATE TABLE last_30_days_historical (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,
    measurement_date TIMESTAMP NOT NULL,
    water_level_cm REAL NOT NULL,
    water_level_m REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station_id, measurement_date)
);
```

---

## 🆕 **5. NEW STATION FETCHER**

### **File:** `fetch_new_station.py`
### **Purpose:** Fetch all data for a new water level station by ID
### **Usage:**
```bash
python3 fetch_new_station.py <station_id> [municipality_id]
```

### **Examples:**
```bash
# Add station to default municipality (ID 1)
python3 fetch_new_station.py 70000864

# Add station to specific municipality
python3 fetch_new_station.py 70000864 1
```

### **Features:**
- ✅ Fetches complete station information from API
- ✅ Converts UTM coordinates to lat/lon
- ✅ Adds station to `stations` table
- ✅ Fetches initial water level data
- ✅ Handles coordinate conversion with pyproj
- ✅ Validates and cleans data

### **API Sources:**
- Station Info: `https://vandah.miljoeportal.dk/api/stations/{station_id}`
- Water Levels: `https://vandah.miljoeportal.dk/api/waterlevels/{station_id}`

---

## 📋 **DAILY WORKFLOW**

### **Recommended Daily Tasks:**

1. **Start Server:**
   ```bash
   python3 start_server.py
   ```

2. **Fetch Today's Predictions:**
   ```bash
   python3 fetch_predictions.py
   ```

3. **Update 30-Day Historical Data:**
   ```bash
   python3 fetch_historical_30days.py
   ```

### **Weekly Tasks:**

4. **Update Historical Min/Max (Weekly):**
   ```bash
   python3 fetch_historical_minmax.py
   ```

### **As Needed:**

5. **Add New Station:**
   ```bash
   python3 fetch_new_station.py <station_id> [municipality_id]
   ```

---

## 🗄️ **DATABASE STRUCTURE**

### **Main Tables:**
- `stations` - Water level stations with coordinates and min/max
- `municipalities` - Municipalities and their information
- `predictions` - Water level predictions
- `last_30_days_historical` - Recent historical water levels
- `weather_stations` - Weather station information per municipality

### **Key Relationships:**
- Stations belong to municipalities
- Predictions are linked to stations
- Historical data is linked to stations
- Weather stations are linked to municipalities

---

## 🔧 **SYSTEM REQUIREMENTS**

### **Python Dependencies:**
- `sqlite3` (built-in)
- `requests` (for API calls)
- `pyproj` (for coordinate conversion)
- `datetime` (built-in)

### **Install Dependencies:**
```bash
pip install requests pyproj
```

### **Required Files:**
- `water_levels.db` (SQLite database)
- `water_level_server_with_municipalities.py` (Flask server)

---

## 📊 **MONITORING & LOGGING**

### **Progress Tracking:**
- All programs provide detailed progress output
- Success/failure counts for each operation
- Error handling with descriptive messages
- Processing time and statistics

### **Error Handling:**
- API timeout handling
- Database connection management
- Data validation and cleaning
- Graceful failure recovery

---

## 🎯 **API ENDPOINTS**

The system provides these main API endpoints:

- `GET /` - API information
- `GET /stations` - All stations with coordinates
- `GET /stations/<id>` - Specific station details
- `GET /stations/<id>/minmax` - Station min/max values
- `GET /water-levels` - Current water levels
- `GET /water-levels/<id>` - Station water levels
- `GET /predictions` - All predictions
- `GET /predictions/<id>` - Station predictions
- `GET /municipalities` - All municipalities
- `GET /municipalities/<id>` - Specific municipality

---

## ✅ **SYSTEM STATUS**

### **Completed Features:**
- ✅ Server startup program
- ✅ Predictions fetching system
- ✅ Historical min/max fetching
- ✅ 30-day historical data fetching
- ✅ New station addition system
- ✅ Database schema and relationships
- ✅ API endpoints and documentation
- ✅ Error handling and logging
- ✅ Progress tracking and monitoring

### **Ready for Production:**
The system is complete and ready for daily operation. All programs are tested and include proper error handling, logging, and progress tracking.

**🌊 Your water level monitoring system is fully operational!**
