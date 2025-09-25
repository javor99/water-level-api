#!/usr/bin/env python3
"""Fix Vandah metadata access to handle missing fields"""

# Read the file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Fix the Vandah metadata access to use .get() method
old_code = '''            if not name:
                name = vandah_metadata['name']
            if not latitude:
                latitude = vandah_metadata['latitude']
            if not longitude:
                longitude = vandah_metadata['longitude']
            if not location_type or location_type == "stream":
                location_type = vandah_metadata['location_type']
            if not station_owner:
                station_owner = vandah_metadata['station_owner']'''

new_code = '''            if not name:
                name = vandah_metadata.get('name', '')
            if not latitude:
                latitude = vandah_metadata.get('latitude')
            if not longitude:
                longitude = vandah_metadata.get('longitude')
            if not location_type or location_type == "stream":
                location_type = vandah_metadata.get('location_type', 'stream')
            if not station_owner:
                station_owner = vandah_metadata.get('station_owner', '')'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Fixed Vandah metadata access to use .get() method")
else:
    print("❌ Could not find the Vandah metadata access code to fix")

# Write the fixed content back
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Vandah metadata access now handles missing fields safely")
