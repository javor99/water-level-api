#!/usr/bin/env python3
"""Debug station creation to find the exact error"""

# Import necessary modules
import sys
sys.path.append('.')

# Mock the Flask request object
class MockRequest:
    def get_json(self):
        return {
            "station_id": "70000618",
            "municipality_id": 1
        }

# Import and test the create_station function
import water_level_server_with_municipalities as server

# Replace the request object temporarily
original_request = server.request
server.request = MockRequest()

try:
    # Call the create_station function directly
    print("🔍 Testing create_station function directly...")
    result = server.create_station()
    print(f"✅ Success: {result}")
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    print("📋 Full traceback:")
    traceback.print_exc()
finally:
    # Restore original request
    server.request = original_request
