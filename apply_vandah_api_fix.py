#!/usr/bin/env python3
"""Apply Vandah API KeyError fix to main server file"""

# Read the file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Fix the Vandah API response access to use .get() method
old_line = "                            'name': s['name'],"
new_line = "                            'name': s.get('name', ''),"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Applied Vandah API KeyError fix")
else:
    print("❌ Could not find the line to fix")

# Write the fixed content back
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Vandah API now handles missing 'name' field safely")
