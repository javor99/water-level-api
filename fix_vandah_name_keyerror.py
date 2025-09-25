#!/usr/bin/env python3
"""Fix KeyError for 'name' in Vandah validation function"""

# Read the file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Fix the Vandah API response access to use .get() method
old_line = "                            'name': s['name'],"
new_line = "                            'name': s.get('name', ''),"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Fixed Vandah 'name' field access to use .get() method")
else:
    print("❌ Could not find the line to fix")

# Write the fixed content back
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Vandah validation now handles missing 'name' field safely")
