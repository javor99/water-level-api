#!/usr/bin/env python3
import sqlite3
import subprocess
import threading
import time

def create_municipality_and_station():
    """Create Struer municipality and station 70000600, then trigger data update."""
    
    print("🏛️ Creating municipality and station...")
    
    # Connect to database
    conn = sqlite3.connect('water_levels.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Create municipality Struer
        print("1. Creating municipality 'Struer'...")
        cursor.execute("""
            INSERT OR REPLACE INTO municipalities (name, region, population, area_km2)
            VALUES (?, ?, ?, ?)
        """, ("Struer", "Midtjylland", 22000, 250.0))
        
        municipality_id = cursor.lastrowid
        print(f"   ✅ Municipality created with ID: {municipality_id}")
        
        # Create station 70000600
        print("2. Creating station '70000600'...")
        station_id = "70000600"
        station_name = "Struer Water Station"
        latitude = 56.5
        longitude = 8.6
        
        cursor.execute("""
            INSERT OR REPLACE INTO stations 
            (station_id, name, latitude, longitude, location_type, station_owner, municipality_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (station_id, station_name, latitude, longitude, "stream", "Struer Municipality", municipality_id))
        
        conn.commit()
        print(f"   ✅ Station created: {station_name} ({station_id})")
        
        conn.close()
        
        # Now trigger the automatic data update
        print(f"\n🚀 Starting automatic data update for station {station_id}...")
        print("This will:")
        print("  📊 Fetch 30 days of recent water level data")
        print("  📈 Calculate min/max values from 5 years of historical data")
        print("  🤖 Generate predictions using the trained models")
        
        def run_update():
            try:
                result = subprocess.run([
                    'python3', 'update_new_station_data.py', station_id
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"\n✅ Automatic data update completed successfully for station {station_id}")
                    print("The station is now fully operational with:")
                    print("  • Recent water level data")
                    print("  • Historical min/max values")
                    print("  • AI-generated predictions")
                else:
                    print(f"\n❌ Data update failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"\n⏰ Data update timed out for station {station_id}")
            except Exception as e:
                print(f"\n❌ Error during data update: {e}")
        
        # Start the update in a separate thread
        update_thread = threading.Thread(target=run_update, daemon=True)
        update_thread.start()
        
        # Wait for it to complete
        update_thread.join()
        
        print(f"\n🎉 Station {station_id} has been created and updated successfully!")
        print("You can now use it in your water level monitoring system.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_municipality_and_station()
